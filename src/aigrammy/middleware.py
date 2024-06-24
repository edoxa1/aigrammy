from typing import Callable, Dict, Any, Awaitable

from aiogram import BaseMiddleware
from aiogram.types import Message, TelegramObject

from .types.response import GptResponse
from .types.chat_completion import GptChatCompletionRepo
from .types.assistant import GptAssistantRepo


class GptChatCompletionMiddleware(BaseMiddleware):
    def __init__(self, client: GptChatCompletionRepo) -> None:
        self.client = client  # GPT client instance
        # note that this constructor will be called only once

    async def __call__(
        self,
        handler: Callable[[Message, Dict[str, Any]], Awaitable[Any]],
        event: Message,
        data: Dict[str, Any],
    ) -> Any:
        data["gpt"] = self.client  # link client to middleware, so we can access it inside handlers;
        return await handler(event, data)


class GptAssistantMiddleware(BaseMiddleware):
    def __init__(self, client: GptAssistantRepo):
        self.client = client

    async def __call__(
            self,
            handler: Callable[[TelegramObject, Dict[str, Any]], Awaitable[Any]],
            event: TelegramObject,
            data: Dict[str, Any]
            ) -> Any:
        data["assistant"] = self.client
        return await handler(event, data)


# todo: multiple assistants with unique names