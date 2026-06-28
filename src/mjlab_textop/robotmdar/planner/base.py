from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from mjlab_textop.robotmdar.feedback import FeedbackObservation


@dataclass(frozen=True)
class PlannerContext:
    frame_index: int
    block_count: int


class PromptPlanner(Protocol):
    @property
    def should_stop(self) -> bool: ...

    @property
    def input_active(self) -> bool: ...

    @property
    def log_suffix(self) -> str: ...

    def start(self) -> None: ...

    def request_stop(self) -> None: ...

    def choose_prompt(self, context: PlannerContext) -> str: ...


class PromptSelector(Protocol):
    def choose_prompt(self, observation: FeedbackObservation | None) -> str: ...


class FeedbackObservationProvider(Protocol):
    def start(self) -> None: ...

    def close(self) -> None: ...

    def latest(self) -> FeedbackObservation | None: ...

    def latest_age_seconds(self) -> float | None: ...
