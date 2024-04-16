import os
from dotenv import load_dotenv

load_dotenv()

GPT4_TURBO = "GPT-4-TURBO"
GPT35_TURBO = "GPT-35T"

OPENAI_API_TYPE = "azure"
OPENAI_API_BASE = "https://clr-openai-hiring.openai.azure.com/"
OPENAI_API_VERSION = "2023-05-15"
OPENAI_API_KEY = os.getenv("AZ_OPENAI_API_KEY")
