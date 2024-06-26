import asyncio
import logging
import sys

from aiogram import Bot, Dispatcher, F, html
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.types import Message, CallbackQuery, BotCommand, ContentType

from aigrammy.types.response import GptResponse
from aigrammy.models import GPTModel
from aigrammy.types.assistant import GptAssistantRepo
from aigrammy.middleware import GptAssistantMiddleware
from environs import Env
from openai import AsyncOpenAI

dp = Dispatcher()
_THREAD_ID = ""


@dp.message(F.text)
async def assistant_text_example(message: Message, assistant: GptAssistantRepo):
    to_delete = await message.answer("Asking GPT...")

    # _THREAD_ID = await database.users.get_thread_id()  # ideally do this

    response = await assistant.push_prompt_to_thread(content=message.text,
                                                     thread_id=_THREAD_ID)

    full_response = f"{html.italic(f'Response: {response.text}')}" \
                    f"\n\n" \
                    f"Finish reason: {response.finish_reason}\n" \
                    f"Prompt tokens: {response.prompt_tokens}\n" \
                    f"Completion tokens: {response.completion_tokens}\n" \
                    f"Total tokens consumed: {response.total_tokens_used}"

    await message.answer(text=full_response)

    await to_delete.delete()


@dp.message(F.photo)
async def assistant_photo_example(message: Message, assistant: GptAssistantRepo):
    to_delete = await message.answer("Asking GPT... (uploading image)")
    # _THREAD_ID = await database.users.get_thread_id()  # ideally do this

    # get file_id of sent photo
    file_id = message.photo.pop().file_id
    # get file_path of sent photo. This path represents a link to photo inside telegram database
    file = await message.bot.get_file(file_id=file_id)

    # create url so OpenAI can download it
    base_url = f'https://api.telegram.org/file/bot' \
               f'{message.bot.token}' \
               f'/{file.file_path}'

    prompt = message.caption
    if not prompt:
        prompt = "no prompt"

    # push new message to thread with given image and prompt
    response = await assistant.push_image_url_to_thread(uri=base_url,
                                                        thread_id=_THREAD_ID,
                                                        content=prompt,
                                                        detail="low")
    # create user-friendly readable response
    full_response = f"{html.italic(f'Response: {response.text}')}" \
                    f"\n\n" \
                    f"Finish reason: {response.finish_reason}\n" \
                    f"Prompt tokens: {response.prompt_tokens}\n" \
                    f"Completion tokens: {response.completion_tokens}\n" \
                    f"Total tokens consumed: {response.total_tokens_used}"

    await message.answer(text=full_response)
    await to_delete.delete()


def setup_assistant(openai_token: str,
                    assistant_id: str,
                    run_instructions: str) -> GptAssistantRepo:
    client = AsyncOpenAI(api_key=openai_token)

    repo = GptAssistantRepo(client=client,
                            assistant_id=assistant_id,
                            run_instructions=run_instructions)

    return repo


async def main():
    # region: Setup environmental variables
    # Initialize Bot instance with default bot properties which will be passed to all API calls
    env = Env()
    env.read_env("./.env")
    # endregion:

    # region: Setup aiogram bot
    _TOKEN = env.str("BOT_TOKEN")
    bot = Bot(token=_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
    # endregion:

    # region: Setup GPT
    _GPT_TOKEN = env.str("GPT_TOKEN")
    _ASSISTANT_ID = env.str("ASSISTANT_ID")
    global _THREAD_ID  # ideally, parse thread_id inside handler.
    _THREAD_ID = env.str("THREAD_ID")
    run_instructions = "End each message with OwO UwU"
    assistant_repo = setup_assistant(openai_token=_GPT_TOKEN,
                                     assistant_id=_ASSISTANT_ID,
                                     run_instructions=run_instructions)

    # endregion:

    # region: Middleware setup
    dp.message.outer_middleware(GptAssistantMiddleware(client=assistant_repo))

    await dp.start_polling(bot)
    # endregion:


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    asyncio.run(main())
