import logging

from openai import AsyncOpenAI
from openai.types.beta.threads import Message

from ..types.response import GptResponse


class GptAssistantRepo:
    def __init__(self, client: AsyncOpenAI,
                 assistant_id: str | None = None,
                 run_instructions: str | None = None):
        self.client = client
        self.assistant_id = assistant_id
        self.run_instructions = run_instructions

    async def create_thread(self) -> str:
        thr = await self.client.beta.threads.create()
        return thr.id

    async def ask_text(self,
                       content: str,
                       thread_id: str,
                       max_prompt_tokens: int = 5000,
                       max_completion_tokens: int = 5000) -> GptResponse | None:
        message = await self.client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=content
        )
        run = await self.client.beta.threads.runs.create_and_poll(
            assistant_id=self.assistant_id,
            thread_id=thread_id,
            instructions=self.run_instructions,
            max_prompt_tokens=max_prompt_tokens,
            max_completion_tokens=max_completion_tokens
        )

        answer = await self._parse_answer(thread_id=thread_id, run_id=run.id)
        match run.status:
            case 'completed':
                return GptResponse(text=answer.data.pop().content.pop().text.value,
                                   finish_reason="end",
                                   completion_tokens=run.usage.completion_tokens,
                                   prompt_tokens=run.usage.prompt_tokens)

            case 'incomplete':
                logging.warning("aigrammy: INCOMPLETE REQUEST DETECTED. Check reason in return value. "
                                "(GptResponse.finish_reason)")
                return GptResponse(text=answer.data.pop().content.pop().text.value,
                                   finish_reason=run.incomplete_details.reason,
                                   completion_tokens=run.usage.completion_tokens,
                                   prompt_tokens=run.usage.prompt_tokens)

            case _:
                return GptResponse(text=None,
                                   finish_reason="failed",
                                   completion_tokens=run.usage.completion_tokens,
                                   prompt_tokens=run.usage.prompt_tokens)

    async def _parse_answer(self, thread_id: str, run_id: str):
        resp = await self.client.beta.threads.messages.list(
            thread_id=thread_id,
            run_id=run_id
        )

        return resp

    async def ASS(self):
        pass
