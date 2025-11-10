import os
from dotenv import load_dotenv
from aiogram import Bot, Dispatcher
from aiogram.types import Message
import asyncio
import openai
import logging
from config import TELEGRAM_TOKEN, OPENROUTER_API_KEY, OPENROUTER_BASE_URL, OPENROUTER_MODEL, PROMPT_ROLE

load_dotenv()

# Минимальный логгинг
logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
#print(f"TELEGRAM_TOKEN: '{TELEGRAM_TOKEN}'")
print(f"OPENROUTER_MODEL: '{OPENROUTER_MODEL}'")
client = openai.AsyncOpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url=OPENROUTER_BASE_URL
)

bot = Bot(token=TELEGRAM_TOKEN)
dp = Dispatcher()

MAX_TELEGRAM_MESSAGE_LENGTH = 4000

def chunk_message(text: str, chunk_size: int = MAX_TELEGRAM_MESSAGE_LENGTH):
    for start in range(0, len(text), chunk_size):
        yield text[start:start + chunk_size]

async def ask_llm(user_message: str) -> str:
    try:
        response = await client.chat.completions.create(
            model=OPENROUTER_MODEL,
            messages=[
                {"role": "system", "content": PROMPT_ROLE},
                {"role": "user", "content": user_message},
            ],
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"LLM error: {e!s}")
        return f"Ошибка: {e!s}"

@dp.message()
async def reply_as_seller(message: Message):
    answer = await ask_llm(message.text)
    for chunk in chunk_message(answer):
        await message.answer(chunk)

async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
