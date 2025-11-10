import os
from dotenv import load_dotenv

load_dotenv()  # ← ЭТО НУЖНО!

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
PROMPT_ROLE = os.getenv(
    "PROMPT_ROLE",
    "Ты — продавец детских велосипедов. Отвечай как компетентный консультант по выбору и покупке детских велосипедов."
    " Сценарии: 1) Помогай подобрать велосипед по возрасту, росту и условиям использования."
    " 2) Просвещай о доставке, оплате, акциях и базовых рекомендациях по уходу."
)
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-3.5-turbo")