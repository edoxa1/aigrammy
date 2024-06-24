import asyncio
import logging
import sys
import typing

from environs import Env

from aiogram import Bot, Dispatcher, html, F
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart
from aiogram.types import Message

from openai import AsyncOpenAI

# pip install --index-url https://test.pypi.org/simple/ --no-deps aigrammy
from aigrammy.types.chat_completion import GptChatCompletionRepo
from aigrammy.types.response import GptResponse

from aigrammy.middleware import GptChatCompletionMiddleware
from aigrammy.models import GPTModel

"""
Majority of code was copied from aiogram's examples directory: 
Source: https://github.com/aiogram/aiogram/blob/dev-3.x/examples/echo_bot.py
"""


# All handlers should be attached to the Router (or Dispatcher)

dp = Dispatcher()


@dp.message(CommandStart())
async def command_start_handler(message: Message) -> None:
    """
    This handler receives messages with `/start` command
    """
    await message.answer(f"Hello, {html.bold(message.from_user.full_name)}!")


@dp.message(F.text)
async def chat_gpt_example(message: Message, gpt: GptChatCompletionRepo):
    # placeholder text so user will know that bot actually works and not crashed
    to_delete = await message.answer("Asking GPT...")

    # Request to ChatGPT begins here:
    response: GptResponse = await gpt.ask_text(prompt=message.text, max_tokens=500)
    # delete 'Asking GPT...'
    await message.bot.delete_message(chat_id=message.chat.id, message_id=to_delete.message_id)
    # response text. Shows all available fields in `GptResponse` class
    full_response = f"Response: {response.text}\n" \
                    f"Finish reason: {response.finish_reason}\n" \
                    f"Prompt tokens: {response.prompt_tokens}\n" \
                    f"Completion tokens: {response.completion_tokens}\n" \
                    f"Total tokens consumed: {response.total_tokens_used}"

    # answer to user:
    await message.answer(text=full_response)


@dp.message(F.photo)
async def chat_gpt_photo_example(message: Message, gpt: GptChatCompletionRepo):
    # placeholder text so user will know that bot actually works and not crashed
    to_delete = await message.answer("Asking GPT...")

    # get file_id of sent photo
    file_id = message.photo.pop().file_id
    # get file_path of sent photo. This path represents a link to photo inside telegram database
    file = await message.bot.get_file(file_id=file_id)
    # download file using provided link. Then, save it to binary stream
    # it was done to avoid saving photo in host system, then reading it again, then by deleting it.
    binary_file: typing.BinaryIO = await message.bot.download_file(file_path=file.file_path)

    # check if prompt was given in message.caption, otherwise, no prompt given.
    prompt = message.caption
    if not prompt:
        prompt = "No prompt given."

    # make request to ChatCompletion and get response.
    response: GptResponse = await gpt.ask_telegram_image(binary_file=binary_file,
                                                         max_tokens=1000,
                                                         prompt=prompt)
    # create user-friendly readable response
    full_response = f"{html.italic(f'Response: {response.text}')}" \
                    f"\n\n" \
                    f"Finish reason: {response.finish_reason}\n" \
                    f"Prompt tokens: {response.prompt_tokens}\n" \
                    f"Completion tokens: {response.completion_tokens}\n" \
                    f"Total tokens consumed: {response.total_tokens_used}"
    await message.answer(text=full_response)


def setup_gpt(gpt_token: str, model: str, system_prompt: str = None) -> GptChatCompletionRepo:
    # create an instance of Asynchronous OpenAI client wih given API key:
    open_ai_client = AsyncOpenAI(api_key=gpt_token)
    # create an instance of ChatCompletionRepo
    aigrammy_client = GptChatCompletionRepo(open_ai_client, model=model, system_prompt=system_prompt)
    return aigrammy_client


async def main() -> None:
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

    model = GPTModel.four_turbo  # feel free to check out all models in that class!
    model = "gpt-4o"  # also feel free to indicate your own model!

    # system prompt will be called every time before creating a chat completion
    system_prompt = "End every message with 'meow:3'"

    # create client, so we can wrap it in middleware
    client = setup_gpt(gpt_token=_GPT_TOKEN,
                       model=model,
                       system_prompt=system_prompt)
    # endregion:

    # region: setup middlewares of aiogram bot
    dp.message.outer_middleware(GptChatCompletionMiddleware(client))
    # # middleware allows us to access GPT client from ANY handler!
    # # In our case only `dp.message` handlers, but we can do it in callback as well!
    # # uncomment following to enable GPT client for any `dp.callback_query`:
    # dp.callback_query.outer_middleware(GptMiddleware(client))

    # # uncomment following to enable GPT client for any `dp.business_message`:
    # dp.business_message.outer_middleware(GptMiddleware(client))
    # endregion:

    # And then run events dispatching
    await dp.start_polling(bot)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    asyncio.run(main())
