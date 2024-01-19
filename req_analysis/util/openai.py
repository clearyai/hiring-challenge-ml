import logging
from config import (
    OPENAI_API_TYPE,
    OPENAI_API_VERSION,
    GPT4_TURBO,
    GPT35_TURBO,
)

from openai import AzureOpenAI
from openai import OpenAIError

logger = logging.getLogger(__name__)


class OpenAIClient:
    allowed_deployments = [GPT4_TURBO]

    def __init__(self, api_key, api_base):
        # no default values since deployment id and keys need to match

        self.api_key = api_key
        self.api_base = api_base
        self.api_type = OPENAI_API_TYPE
        self.api_version = OPENAI_API_VERSION

        self.client = AzureOpenAI(
            api_version=self.api_version,
            azure_endpoint=self.api_base,
            api_key=self.api_key,
        )

        self.num_retries = 4
        self.temp = 0.0

    def call_llm(self, messages, deployment_id):
        if deployment_id not in self.allowed_deployments:
            raise ValueError(f"Deployment {deployment_id} not in allowed deployments")

        for i in range(self.num_retries):
            try:
                logger.info("Calling OpenAI API")
                res = self.client.chat.completions.create(
                    messages=messages,
                    model=deployment_id,
                    temperature=self.temp,
                )
                res = res.choices[0].message
            except OpenAIError as e:
                if i < self.num_retries - 1:
                    logger.warning("Openai API call failed. Retrying...", exc_info=True)
                else:
                    logger.error(
                        "Openai API call failed. No more retries.", exc_info=True
                    )
                    raise e
            break

        return res

    def make_messages(self, system_prompt, initial_prompt, deployment_id=GPT4_TURBO):
        messages = [
            {"content": system_prompt, "role": "system"},
            {"content": initial_prompt, "role": "user"},
        ]

        if deployment_id == GPT35_TURBO:
            messages[0]["role"] = "user"

        return messages
