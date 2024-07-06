from dataclasses import dataclass
from typing import Literal, Dict


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


class FailedGptResponse(GptResponse):
    def __init__(self, fail_reason: str = "unknown"):
        super().__init__(text=None,
                         finish_reason=fail_reason,
                         prompt_tokens=0,
                         completion_tokens=0)

#
# MODERATION_LITERAL = Literal["sexual", "hate", "harassment",
#                              "self-harm", "sexual/minors", "hate/threatening",
#                              "violence/graphic", "self-harm/intent", "self-harm/instructions",
#                              "harassment/threatening", "violence"]


class ModerationResponse:
    def __init__(
            self,
            is_flagged: bool = False,
            category_names: list[str] | None = None,
            category_scores: list[float] | None = None
    ):
        self.is_flagged = is_flagged
        self.category_name = category_names
        self.category_score = category_scores

