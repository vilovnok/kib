
import requests


# ============================================================
# CONFIGURATION
# ============================================================

API_URL = "https://api.aitunnel.ru/v1/chat/completions"

API_KEY = "sk-aitunnel-952OyNguzQqPU6Esiay4Bo0YioSi1SXj"

MODEL = "deepseek-v4.1-flash"

REASONING_EFFORT = "medium"

TIMEOUT = (60, 300)


# ============================================================
# AITUNNEL CLIENT
# ============================================================

class AITunnelClient:
    """
    Клиент для обращения к DeepSeek через AITunnel API.

    Использует OpenAI-compatible Chat Completions API.
    """

    def __init__(self):
        if not API_KEY or API_KEY == "ВСТАВЬ_СЮДА_СВОЙ_API_ТОКЕН":
            raise ValueError(
                "Необходимо указать API_KEY для AITunnel."
            )

        self.url = API_URL
        self.model = MODEL

        self.session = requests.Session()

        self.headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        }

    def invoke(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        """
        Отправляет запрос модели и возвращает текст ответа.
        """

        payload = {
            "model": self.model,
            "reasoning": {
                "effort": REASONING_EFFORT,
            },
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
        }

        response = self.session.post(
            self.url,
            headers=self.headers,
            json=payload,
            timeout=TIMEOUT,
        )

        response.raise_for_status()

        data = response.json()

        choices = data.get("choices") or []

        if not choices:
            raise ValueError(
                f"Модель вернула пустой choices: {data}"
            )

        message = choices[0].get("message") or {}

        content = message.get("content", "")

        if isinstance(content, list):
            content = " ".join(
                part.get("text", "")
                for part in content
                if isinstance(part, dict)
            )

        if not isinstance(content, str) or not content.strip():
            raise ValueError(
                f"Модель вернула пустой ответ: {data}"
            )

        return content
