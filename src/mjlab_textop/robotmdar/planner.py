from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Protocol

from mjlab_textop.robotmdar.feedback import FeedbackObservation


@dataclass
class PromptState:
    text: str
    stop: bool = False
    input_active: bool = False


@dataclass(frozen=True)
class PlannerContext:
    frame_index: int
    block_count: int


@dataclass(frozen=True)
class GeneratedBlockInfo:
    prompt: str
    start_frame: int
    frames: int
    block_count: int


class PromptPlanner(Protocol):
    @property
    def should_stop(self) -> bool:
        ...

    @property
    def input_active(self) -> bool:
        ...

    @property
    def log_suffix(self) -> str:
        ...

    def start(self) -> None:
        ...

    def request_stop(self) -> None:
        ...

    def choose_prompt(self, context: PlannerContext) -> str:
        ...

    def on_block_sent(self, info: GeneratedBlockInfo) -> None:
        ...


class PromptSelector(Protocol):
    def choose_prompt(self, observation: FeedbackObservation | None) -> str:
        ...


class FeedbackObservationProvider(Protocol):
    def start(self) -> None:
        ...

    def close(self) -> None:
        ...

    def latest(self) -> FeedbackObservation | None:
        ...

    def latest_age_seconds(self) -> float | None:
        ...


class ConstantPromptSelector:
    def __init__(self, prompt: str) -> None:
        if not prompt:
            raise ValueError("prompt must not be empty")
        self.prompt = prompt

    def choose_prompt(self, observation: FeedbackObservation | None) -> str:
        return self.prompt


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

    def on_block_sent(self, info: GeneratedBlockInfo) -> None:
        return


class FeedbackPlanner:
    def __init__(
        self,
        *,
        observation_provider: FeedbackObservationProvider,
        selector: PromptSelector,
        initial_prompt: str,
        query_every_blocks: int,
        fallback_prompt: str,
        stale_steps_threshold: int,
        feedback_timeout_sec: float | None = None,
    ) -> None:
        if query_every_blocks <= 0:
            raise ValueError(
                f"query_every_blocks must be positive, got {query_every_blocks}"
            )
        if stale_steps_threshold < 0:
            raise ValueError(
                "stale_steps_threshold must be non-negative, "
                f"got {stale_steps_threshold}"
            )
        self.observation_provider = observation_provider
        self.selector = selector
        self.current_prompt = initial_prompt
        self.query_every_blocks = query_every_blocks
        self.fallback_prompt = fallback_prompt
        self.stale_steps_threshold = stale_steps_threshold
        self.feedback_timeout_sec = feedback_timeout_sec
        self._stop = False
        self._last_query_block: int | None = None
        self._last_override_reason: str | None = None

    @property
    def should_stop(self) -> bool:
        return self._stop

    @property
    def input_active(self) -> bool:
        return False

    @property
    def log_suffix(self) -> str:
        if self._last_override_reason is None:
            return ""
        return f" planner_override={self._last_override_reason}"

    def start(self) -> None:
        self.observation_provider.start()

    def request_stop(self) -> None:
        self._stop = True
        self.observation_provider.close()

    def choose_prompt(self, context: PlannerContext) -> str:
        self._last_override_reason = None
        observation = self.observation_provider.latest()

        if self._feedback_is_stale():
            return self.current_prompt

        if (
            observation is not None
            and observation.consecutive_stale_steps >= self.stale_steps_threshold
        ):
            self.current_prompt = self.fallback_prompt
            self._last_override_reason = "stale_tracking"
            return self.current_prompt

        if self._should_query_selector(context):
            self.current_prompt = self.selector.choose_prompt(observation)
            self._last_query_block = context.block_count

        return self.current_prompt

    def on_block_sent(self, info: GeneratedBlockInfo) -> None:
        return

    def _should_query_selector(self, context: PlannerContext) -> bool:
        if self._last_query_block is None:
            return True
        return context.block_count - self._last_query_block >= self.query_every_blocks

    def _feedback_is_stale(self) -> bool:
        if self.feedback_timeout_sec is None:
            return False
        latest_age = self.observation_provider.latest_age_seconds()
        if latest_age is None:
            return False
        return latest_age > self.feedback_timeout_sec


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
