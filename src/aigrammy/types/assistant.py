import logging
from base64 import b64encode
from typing import BinaryIO, Literal

from openai import AsyncOpenAI
from openai.types.beta import Thread
from openai.types.beta.threads import Run

from ..types.response import GptResponse


class GptAssistantRepo:
    def __init__(self, client: AsyncOpenAI,
                 assistant_id: str | None = None,
                 run_instructions: str | None = None):
        self.client = client
        self.assistant_id = assistant_id
        self.run_instructions = run_instructions

    async def create_thread(self) -> Thread:
        thr = await self.client.beta.threads.create()
        return thr

    async def push_prompt_to_thread(
            self,
            content: str,
            thread_id: str,
            max_prompt_tokens: int | None = None,
            max_completion_tokens: int | None = None
    ) -> GptResponse:
        message = await self.client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=content
        )
        run = await self._execute_run(max_completion_tokens, max_prompt_tokens, thread_id)
        response = await self._match_status(thread_id=thread_id, run=run)
        return response

    async def push_image_url_to_thread(
            self,
            thread_id: int,
            uri: str,
            content: str = "(no additional info was specified)",
            detail: str = Literal["auto", "low", "high"],
            max_prompt_tokens: int | None = None,
            max_completion_tokens: int | None = None
    ) -> GptResponse:
        await self.client.beta.threads.messages.create(
            thread_id=thread_id,
            role='user',
            content=[
                {
                    "type": "image_url",
                    "image_url": {
                        "url": uri,
                        "detail": detail
                    }
                },
                {
                    "type": "text",
                    "text": content
                }
            ]
        )

        run = await self._execute_run(max_completion_tokens, max_prompt_tokens, thread_id)
        response = await self._match_status(thread_id=thread_id, run=run)
        return response

    async def push_image_binaryio_to_thread(
            self,
            thread_id: str,
            binary_file: BinaryIO,
            content: str = "(no additional info was specified)",
            detail: str = Literal["auto", "low", "high"],
            max_prompt_tokens: int | None = None,
            max_completion_tokens: int | None = None
    ) -> GptResponse:
        try:
            b64_str = self._encode_image(binary_file)
        except Exception as e:
            logging.exception(f"Failed to convert BinaryIO to b64 while sending image to OpenAI. Error: {e}")
            binary_file.close()
            raise e  # try-except used here to close the binary file and avoid potential memory leak

        img_url = f"data:image/jpeg;base64,{b64_str}"
        await self.client.beta.threads.messages.create(
            thread_id=thread_id,
            role='user',
            content=[
                {
                    "type": "image_url",
                    "image_url": {
                        "url": img_url,
                        "detail": detail
                    }
                },
                {
                    "type": "text",
                    "text": content
                }
            ]
        )

        run = await self._execute_run(max_completion_tokens, max_prompt_tokens, thread_id)
        response = await self._match_status(thread_id=thread_id, run=run)
        return response

    async def _execute_run(self, max_completion_tokens, max_prompt_tokens, thread_id):
        run = await self.client.beta.threads.runs.create_and_poll(
            assistant_id=self.assistant_id,
            thread_id=thread_id,
            instructions=self.run_instructions,
            max_prompt_tokens=max_prompt_tokens,
            max_completion_tokens=max_completion_tokens
        )
        return run

    async def _parse_answer(self, thread_id: str, run_id: str):
        resp = await self.client.beta.threads.messages.list(
            thread_id=thread_id,
            run_id=run_id
        )

        return resp

    async def _match_status(
            self,
            thread_id: str,
            run: Run,
    ) -> GptResponse:
        """ Match-case for `run.status` """
        answer = await self._parse_answer(thread_id=thread_id, run_id=run.id)
        match run.status:
            case 'completed':
                return GptResponse(text=answer.data.pop().content.pop().text.value,
                                   finish_reason="end",
                                   completion_tokens=run.usage.completion_tokens,
                                   prompt_tokens=run.usage.prompt_tokens)

            case 'incomplete':
                logging.warning("aigrammy: INCOMPLETE REQUEST DETECTED. Check reason in return value. "
                                "`GptResponse.finish_reason`")
                return GptResponse(text=answer.data.pop().content.pop().text.value,
                                   finish_reason=run.incomplete_details.reason,
                                   completion_tokens=run.usage.completion_tokens,
                                   prompt_tokens=run.usage.prompt_tokens)

            case _:
                return GptResponse(text=None,
                                   finish_reason="failed",
                                   completion_tokens=run.usage.completion_tokens,
                                   prompt_tokens=run.usage.prompt_tokens)

    @staticmethod
    def _encode_image(file: BinaryIO):
        """ Private method used to convert `io.BinaryIO` to b64 string """
        file.seek(0)
        return b64encode(file.read()).decode('utf-8')
