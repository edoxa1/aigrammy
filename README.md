# aigrammy — simple ChatGPT wrapper for your telegram bot!


## Description
Features:
* Utilizes only most relevant data from requests
* Text prompts
* Automatically prepares telegram photo to be sent to OpenAI

Supports:
* Chat Completions
* Assistant

## Install
Work in progress

## Usage in real project:
1. Import aigrammy.middlewares
2. Wrap middleware in any handler (e.g. message, callback_query)
3. Access instance of Repo inside handler
4. Call methods inside instance

Refer to `examples/chat_completions/basic_with_middleware.py`
