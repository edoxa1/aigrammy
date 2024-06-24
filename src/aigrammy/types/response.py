class GptResponse:
    def __init__(self,
                 text: str,
                 finish_reason: str,
                 completion_tokens: int,
                 prompt_tokens: int):
        self.text = text
        self.finish_reason = finish_reason
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens_used = prompt_tokens + completion_tokens
