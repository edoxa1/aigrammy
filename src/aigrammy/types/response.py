class GptResponse:
    def __init__(self,
                 text: str,
                 finish_reason: str,
                 prompt_tokens: int = 0,
                 completion_tokens: int = 0
                 ):
        self.text = text
        self.finish_reason = finish_reason
        self.completion_tokens = completion_tokens
        self.prompt_tokens = prompt_tokens
        self.total_tokens_used = prompt_tokens + completion_tokens
