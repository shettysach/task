from __future__ import annotations

from mjlab_textop.robotmdar.feedback import FeedbackObservation

from .base import PlannerContext


class ConstantPromptSelector:
    def __init__(self, prompt: str) -> None:
        if not prompt:
            raise ValueError("prompt must not be empty")
        self.prompt = prompt

    def choose_prompt(
        self,
        *,
        observation: FeedbackObservation | None,
        context: PlannerContext,
        current_prompt: str,
    ) -> str:
        return self.prompt
