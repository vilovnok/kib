
import argparse
import json
import logging
import threading

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from classifier.llm import GigaChatClient


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_PATH = Path("data/gt/pool1.json")
OUTPUT_PATH = Path("data/predictions/pool1_predictions.json")
ERRORS_PATH = Path("data/predictions/pool1_errors.json")
SYSTEM_PROMPT_PATH = Path("prompts/system_prompt.md")

LOG_EVERY = 10

NO_TOOL_REASONS = {
    "missing_data",
    "clarification_needed",
    "unclear_request",
    "no_suitable_tool",
    "config_request",
    "no_tool_needed",
}

worker_local = threading.local()


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ============================================================
# SYSTEM PROMPT
# ============================================================

def load_system_prompt() -> str:
    """Загружает системный промпт из файла."""

    if not SYSTEM_PROMPT_PATH.exists():
        raise FileNotFoundError(
            f"Не найден системный промпт: {SYSTEM_PROMPT_PATH}"
        )

    prompt = SYSTEM_PROMPT_PATH.read_text(
        encoding="utf-8"
    )

    if not prompt.strip():
        raise ValueError("Системный промпт пустой.")

    return prompt


# ============================================================
# WORKER INITIALIZATION
# ============================================================

def get_worker_client() -> GigaChatClient:
    """
    Возвращает GigaChatClient текущего потока.

    Каждый поток создаёт собственный клиент
    и использует его для последующих запросов.
    """

    if not hasattr(worker_local, "client"):

        worker_local.client = GigaChatClient()

        logger.info(
            "Создан GigaChatClient для потока %s",
            threading.current_thread().name,
        )

    return worker_local.client


# ============================================================
# INPUT PREPARATION
# ============================================================

def prepare_user_prompt(
    data: dict[str, Any],
) -> str:
    """
    Формирует входное сообщение для модели.

    id и role не передаются модели.
    tools переименовывается в available_tools.
    """

    question = data.get("question")
    tools = data.get("tools")

    if not isinstance(question, str) or not question.strip():
        raise ValueError("Поле question отсутствует или пустое.")

    if not isinstance(tools, str) or not tools.strip():
        raise ValueError("Поле tools отсутствует или пустое.")

    user_data = {
        "question": question,
        "available_tools": tools,
    }

    return json.dumps(
        user_data,
        ensure_ascii=False,
        indent=4,
    )


def parse_available_tools(tools: str) -> set[str]:
    """Преобразует строку доступных инструментов в множество."""

    return {
        tool.strip()
        for tool in tools.split(",")
        if tool.strip()
    }


# ============================================================
# RESPONSE PARSING
# ============================================================

def extract_json(content: str) -> dict[str, Any]:
    """
    Извлекает JSON-объект из ответа модели.

    Поддерживает JSON без обёртки и JSON,
    помещённый в Markdown-блок кода.
    """

    if not isinstance(content, str) or not content.strip():
        raise ValueError("Модель вернула пустой ответ.")

    content = content.strip()

    # Убираем Markdown-обёртку, если она присутствует.
    if content.startswith("```"):
        lines = content.splitlines()

        if lines[-1].strip() == "```":
            content = "\n".join(lines[1:-1]).strip()

    # Сначала пробуем разобрать весь ответ.
    try:
        parsed = json.loads(content)

    except json.JSONDecodeError:
        # Если модель добавила текст вокруг JSON,
        # извлекаем первый JSON-объект.
        decoder = json.JSONDecoder()

        start = content.find("{")

        if start == -1:
            raise ValueError(
                f"В ответе модели не найден JSON: {content}"
            )

        parsed, _ = decoder.raw_decode(content[start:])

    if not isinstance(parsed, dict):
        raise ValueError("Ответ модели должен быть JSON-объектом.")

    return parsed


# ============================================================
# RESULT VALIDATION
# ============================================================

def validate_result(
    response: dict[str, Any],
    available_tools: set[str],
) -> dict[str, Any]:
    """
    Проверяет и нормализует результат классификации.

    Возвращает только поля целевой схемы.
    """

    if "result" not in response:
        raise ValueError(
            "В ответе модели отсутствует поле result."
        )

    result = response["result"]

    if not isinstance(result, dict):
        raise ValueError("Поле result должно быть JSON-объектом.")

    tool_needed = result.get("tool_needed")
    agent_code = result.get("agent_code")

    if tool_needed == "yes":

        if not isinstance(agent_code, str) or not agent_code.strip():
            raise ValueError(
                "При tool_needed='yes' должен быть указан agent_code."
            )

        agent_code = agent_code.strip()

        if agent_code not in available_tools:
            raise ValueError(
                f"Выбран недоступный инструмент: {agent_code}"
            )

        return {
            "agent_code": agent_code,
            "tool_needed": "yes",
        }

    if tool_needed == "no":

        if agent_code is not None:
            raise ValueError(
                "При tool_needed='no' agent_code должен быть null."
            )

        no_tool_reason = result.get("no_tool_reason")

        if no_tool_reason not in NO_TOOL_REASONS:
            raise ValueError(
                f"Недопустимое значение no_tool_reason: "
                f"{no_tool_reason}"
            )

        return {
            "agent_code": None,
            "tool_needed": "no",
            "no_tool_reason": no_tool_reason,
        }

    if not isinstance(no_tool_reason, str) or no_tool_reason not in NO_TOOL_REASONS:
        raise ValueError(
            f"Недопустимое значение no_tool_reason: {no_tool_reason}"
        )


# ============================================================
# SINGLE TASK PROCESSING
# ============================================================

def process_task(
    task: dict[str, Any],
    system_prompt: str,
) -> dict[str, Any]:

    if not isinstance(task, dict):
        raise ValueError("Задание должно быть JSON-объектом.")

    data = task.get("data")
    gt = task.get("result")

    if not isinstance(data, dict):
        raise ValueError("В задании отсутствует объект data.")

    if not isinstance(gt, dict):
        raise ValueError("В задании отсутствует объект result (GT).")

    if not isinstance(data.get("id"), str) or not data["id"].strip():
        raise ValueError("Отсутствует корректный id задания.")

    user_prompt = prepare_user_prompt(data)
    available_tools = parse_available_tools(data["tools"])

    client = get_worker_client()

    content = client.invoke(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )

    response = extract_json(content)
    prediction = validate_result(response, available_tools)

    return {
        "data": data,
        "gt": gt,
        "prediction": prediction,
    }


# ============================================================
# PARALLEL PROCESSING
# ============================================================

def evaluate_task(
    task: dict[str, Any],
    system_prompt: str,
) -> dict[str, Any]:
    """Обрабатывает задание и перехватывает ошибки."""

    data = task.get("data")

    task_id = (
        data.get("id")
        if isinstance(data, dict)
        else None
    )

    try:
        result = process_task(
            task=task,
            system_prompt=system_prompt,
        )

        return {
            "success": True,
            "result": result,
        }

    except Exception as exc:
        logger.exception(
            "Ошибка обработки задания %s",
            task_id,
        )

        return {
            "success": False,
            "error": {
                "id": task_id,
                "error": str(exc),
            },
        }


# ============================================================
# SAVE RESULTS
# ============================================================

def save_json(
    data: list[dict[str, Any]],
    path: Path,
) -> None:
    """Безопасно сохраняет JSON через временный файл."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    temp_path.write_text(
        json.dumps(
            data,
            ensure_ascii=False,
            indent=4,
        ),
        encoding="utf-8",
    )

    temp_path.replace(path)


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Параллельная классификация запросов через LLM."
    )

    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=1,
        help="Количество параллельных потоков.",
    )

    args = parser.parse_args()

    if args.workers < 1:
        parser.error("--workers должен быть не меньше 1.")

    return args


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    args = parse_args()
    workers = args.workers

    # 1. Проверяем и загружаем входные данные.

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Не найден входной файл: {INPUT_PATH}"
        )

    with INPUT_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        tasks = json.load(file)

    if not isinstance(tasks, list):
        raise ValueError(
            "Входной JSON должен содержать список заданий."
        )

    total_tasks = len(tasks)

    logger.info("Загружено заданий: %s", total_tasks)
    logger.info("Количество воркеров: %s", workers)

    # 2. Загружаем системный промпт.

    system_prompt = load_system_prompt()

    # 3. Подготавливаем хранилища результатов.

    results = [None] * total_tasks
    errors = [None] * total_tasks

    # 4. Запускаем параллельную обработку.

    with ThreadPoolExecutor(
        max_workers=workers,
        thread_name_prefix="llm-worker",
    ) as executor:

        futures = {
            executor.submit(
                evaluate_task,
                task,
                system_prompt,
            ): position

            for position, task in enumerate(tasks)
        }

        completed = 0

        for future in as_completed(futures):

            position = futures[future]

            try:
                outcome = future.result()

                if outcome["success"]:
                    results[position] = outcome["result"]
                else:
                    errors[position] = outcome["error"]

            except Exception as exc:

                task_id = tasks[position].get(
                    "data", {}
                ).get("id")

                errors[position] = {
                    "id": task_id,
                    "error": str(exc),
                }

                logger.exception(
                    "Необработанная ошибка задания %s",
                    task_id,
                )

            completed += 1

            if (
                completed % LOG_EVERY == 0
                or completed == total_tasks
            ):

                successful_results = [
                    item
                    for item in results
                    if item is not None
                ]

                failed_results = [
                    item
                    for item in errors
                    if item is not None
                ]

                save_json(
                    successful_results,
                    OUTPUT_PATH,
                )

                save_json(
                    failed_results,
                    ERRORS_PATH,
                )

                logger.info(
                    "Обработано %s/%s | Успешно: %s | Ошибки: %s",
                    completed,
                    total_tasks,
                    len(successful_results),
                    len(failed_results),
                )

    # 5. Финальное сохранение.

    successful_results = [
        item
        for item in results
        if item is not None
    ]

    failed_results = [
        item
        for item in errors
        if item is not None
    ]

    save_json(
        successful_results,
        OUTPUT_PATH,
    )

    save_json(
        failed_results,
        ERRORS_PATH,
    )

    # 6. Итоговая статистика.

    logger.info("Обработка завершена.")
    logger.info("Всего заданий: %s", total_tasks)
    logger.info("Успешно обработано: %s", len(successful_results))
    logger.info("Ошибок: %s", len(failed_results))

    logger.info("Предсказания: %s", OUTPUT_PATH)
    logger.info("Ошибки: %s", ERRORS_PATH)


if __name__ == "__main__":
    main()
