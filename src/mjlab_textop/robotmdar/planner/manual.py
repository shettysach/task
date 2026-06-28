from __future__ import annotations

import threading
from dataclasses import dataclass

from .base import PlannerContext


@dataclass
class PromptState:
    text: str
    stop: bool = False
    input_active: bool = False


class ManualPromptPlanner:
    def __init__(self, initial_prompt: str) -> None:
        self.prompt = PromptState(text=initial_prompt)
        self._thread: threading.Thread | None = None

    @property
    def should_stop(self) -> bool:
        return self.prompt.stop

    @property
    def input_active(self) -> bool:
        return self.prompt.input_active

    @property
    def log_suffix(self) -> str:
        return "\nEnter text prompt (or q to exit): "

    def start(self) -> None:
        self._thread = threading.Thread(
            target=_prompt_loop,
            args=(self.prompt,),
            daemon=True,
        )
        self._thread.start()

    def request_stop(self) -> None:
        self.prompt.stop = True

    def choose_prompt(self, context: PlannerContext) -> str:
        return self.prompt.text


def _prompt_loop(prompt: PromptState) -> None:
    while not prompt.stop:
        try:
            prompt.input_active = True
            text = input("Enter text prompt (or q to exit): ").strip()
        except (EOFError, KeyboardInterrupt):
            prompt.stop = True
            return
        finally:
            prompt.input_active = False
        if text.lower() in {"q", "quit", "exit"}:
            prompt.stop = True
        elif text:
            prompt.text = text
