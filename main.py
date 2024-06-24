import asyncio
from environs import Env
from openai import AsyncOpenAI

from src.aigrammy.types.assistant import GptAssistantRepo


def parse_config():
    CONFIG: dict = {}
    env = Env()
    env.read_env(path="./.env")
    gpt_token = env.str("GPT_TOKEN")
    assistant_id = env.str("ASSISTANT_ID")
    thread_id = env.str("THREAD_ID")
    CONFIG.update({
        "GPT_TOKEN": gpt_token,
        "ASSISTANT_ID": assistant_id,
        "THREAD_ID": thread_id
    })

    return CONFIG


async def main():
    config = parse_config()

    client = AsyncOpenAI(api_key=config.get("GPT_TOKEN"))
    assistant_id: str = config.get("ASSISTANT_ID")

    assistant_repo = GptAssistantRepo(client=client,
                                      assistant_id=assistant_id,
                                      run_instructions="End each answer with 'meow:3'")
    # thread_id = await assistant_repo.create_thread()  # -> UNCOMMENT FOR FIRST TIME,
    # print(thread_id)
    # THEN SAVE IN ENV(in reality save in database)

    # TODO: SAVE THREAD_ID in database FOR EACH user, so we don't have to create UNIQUE thread every time
    thread_id = config.get("THREAD_ID")
    response = await assistant_repo.ask_text("Hello?", thread_id=thread_id)
    print(response.text)
    print(f"PT: {response.prompt_tokens}; CT: {response.completion_tokens}")

if __name__ == '__main__':
    asyncio.run(main())
