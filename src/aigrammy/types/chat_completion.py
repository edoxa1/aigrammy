import logging

from typing import BinaryIO
from base64 import b64encode
from openai import AsyncOpenAI

from ..exceptions import NoGptPromptSpecifiedException
from .response import GptResponse


class GptChatCompletionRepo:
    """
    Asynchronous ChatGPT wrapper for `telegram` bots based on `aiogram`\n
    Useful and flexible requests/responses, with only relevant fields!
    """

    def __init__(self,
                 client: AsyncOpenAI,
                 model: str,
                 system_prompt: str = ""
                 ):
        """
        :param client: instance of `AsyncOpenAI`
        :param model: `aigrammy.models.GPT instance, or `str` according to https://platform.openai.com/docs/models
        :param system_prompt: system prompt which will be used in message generation
        """
        self.model = model
        self.client = client
        self.system_prompt = system_prompt

    def setup_logging(self, log_level=logging.INFO) -> None:
        """
        Setup logging. Overrides internal
        :param log_level: Logging level. Enum can be accessed from `logging` package
        :return:
        """
        logger = logging.getLogger(__name__)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        pass

    async def ask_text(
            self,
            prompt: str,
            max_tokens=1000
    ) -> GptResponse:
        """ Sends message to ChatGPT with given prompt.
        :param prompt: Given prompt
        :param max_tokens: Maximum number of tokens which ChatGPT can generate, `default=1000`

        :return: Returns the `aiogpt.models.GptResponse` instance
        """
        if not prompt:
            raise NoGptPromptSpecifiedException("Given prompt is `empty` or `None`!")

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": f"System instructions: {self.system_prompt}"
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=max_tokens,
        )

        return GptResponse(text=response.choices[0].message.content,
                           finish_reason=response.choices[0].finish_reason,
                           completion_tokens=response.usage.completion_tokens,
                           prompt_tokens=response.usage.prompt_tokens)

    async def ask_from_binaryio_image(
            self,
            binary_file: BinaryIO,
            max_tokens=500,
            content: str = "(no additional info was specified)"
    ) -> GptResponse:
        """
        Sends message to ChatGPT with given image in `BinaryIO` format, as well as certain given prompt.
        Will automatically convert it to b64 string, provide only `io.BinaryIO` instance.
        :param binary_file: Binary from `aiogram.bot.download_file(file_id: str)`. Automatically closes stream!
            How to use: https://docs.aiogram.dev/en/latest/api/download_file.html#download-file-to-binary-i-o-object
        :param max_tokens: Maximum number of tokens which ChatGPT can generate, `default=500`
        :param content: prompt which will be sent with given image

        :return: Returns the `aiogpt.models.GptResponse` instance
        """
        try:
            b64_str = self._encode_image(binary_file)
        except Exception as e:
            logging.exception(f"Failed to convert BinaryIO to b64 while sending image to OpenAI. Error: {e}")
            binary_file.close()
            raise e  # try-except used here to close the binary file and avoid potential memory leak

        img_url = f"data:image/jpeg;base64,{b64_str}"
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": f"System instructions: {self.system_prompt}"
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": content
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": img_url,
                            },
                        },
                    ],
                }
            ],
            max_tokens=max_tokens,
        )

        binary_file.close()
        return GptResponse(text=response.choices[0].message.content,
                           finish_reason=response.choices[0].finish_reason,
                           completion_tokens=response.usage.completion_tokens,
                           prompt_tokens=response.usage.prompt_tokens)

    async def ask_telegram_image_url(
            self,
            uri: str,
            max_tokens=500,
            content: str = "(no additional info was specified)"
    ) -> GptResponse:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": f"System instructions: {self.system_prompt}"
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": content
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": uri,
                            },
                        },
                    ],
                }
            ],
            max_tokens=max_tokens,
        )

        return GptResponse(text=response.choices[0].message.content,
                           finish_reason=response.choices[0].finish_reason,
                           completion_tokens=response.usage.completion_tokens,
                           prompt_tokens=response.usage.prompt_tokens)

    def change_model(self, new_model: str):
        """ Changes the default model of ChatGPT"""
        old_model = self.model
        self.model = new_model

        logging.warning(f"Changed `ChatCompletionRepo.model` from {old_model} to {new_model}")

    @staticmethod
    def _encode_image(file: BinaryIO):
        """ Private method used to convert `io.BinaryIO` to b64 string """
        file.seek(0)
        return b64encode(file.read()).decode('utf-8')
