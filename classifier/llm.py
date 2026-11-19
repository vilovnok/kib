import os
import requests

from crowd_sdk.tagme import TagmeClientAdvanced


PROVIDER = "aihf" # "vsellm" # aihf direct

ENDPOINTS = {
    "direct": "https://tagme.sberdevices.ru/ai/llm/v1/chat/completions",
    "aihf": "https://tagme.sberdevices.ru/x/aihf/llm/v1/chat/completions",
    "vsellm": "https://tagme.sberdevices.ru/x/ai/llm/v1/chat/completions",
}

CFG = {
    "direct": "~/configs/.ml_config.cfg",
    "aihf": "~/configs/.ml_config.cfg",
    "vsellm": "~/configs/.ml_config.cfg",
}

MODEL = "DeepSeek-V4.1-Flash" # "openai/gpt-5.6-luna" # "DeepSeek-V4.1-Flash" # "GigaChat-3.5-432B-A28B" 



# REPETITION_PENALTY = 1.05
# MAX_RETRIES = 0

TEMPERATURE = 0.1
MAX_TOKENS = 8000 # 512
TIMEOUT = (60, 120)


class GigaChatClient:

    def __init__(self):
        self.url = ENDPOINTS[PROVIDER]
        self.model = MODEL
        self.token = self._get_token()

        self.session = requests.Session()

    @staticmethod
    def _get_token() -> str:
        cfg_path = os.path.expanduser(CFG[PROVIDER])

        client = TagmeClientAdvanced(cfg_path)

        return client.client.token

    def invoke(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:

        payload = {
            "model": self.model,
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
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
            "profanity_check": False,
        }

        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

        response = self.session.post(
            self.url,
            headers=headers,
            json=payload,
            verify=False,
            timeout=TIMEOUT,
        )

        # Если токен истёк, обновляем его и повторяем запрос один раз.
        if response.status_code == 401:
            self.token = self._get_token()
            headers["Authorization"] = f"Bearer {self.token}"

            response = self.session.post(
                self.url,
                headers=headers,
                json=payload,
                verify=False,
                timeout=TIMEOUT,
            )

        response.raise_for_status()

        data = response.json()

        choices = data.get("choices") or []

        if not choices:
            raise ValueError(
                f"Модель вернула пустой choices: {data}"
            )

        content = (
            choices[0].get("message") or {}
        ).get("content", "")

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
