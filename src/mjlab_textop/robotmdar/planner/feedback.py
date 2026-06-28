from __future__ import annotations

from .base import FeedbackObservationProvider, PlannerContext, PromptSelector


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
        fall_recovery_blocks: int = 8,
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
        if fall_recovery_blocks < 0:
            raise ValueError(
                f"fall_recovery_blocks must be non-negative, got {fall_recovery_blocks}"
            )
        self.observation_provider = observation_provider
        self.selector = selector
        self.current_prompt = initial_prompt
        self.query_every_blocks = query_every_blocks
        self.fallback_prompt = fallback_prompt
        self.stale_steps_threshold = stale_steps_threshold
        self.fall_recovery_blocks = fall_recovery_blocks
        self.feedback_timeout_sec = feedback_timeout_sec
        self._stop = False
        self._last_query_block: int | None = None
        self._last_override_reason: str | None = None
        self._fall_recovery_until_block: int | None = None

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

        if observation is not None and observation.fallen:
            self._last_override_reason = (
                "fallen"
                if observation.fall_reason is None
                else f"fallen:{observation.fall_reason}"
            )
            self._fall_recovery_until_block = (
                context.block_count + self.fall_recovery_blocks
            )
            return self.fallback_prompt

        if (
            self._fall_recovery_until_block is not None
            and context.block_count < self._fall_recovery_until_block
        ):
            self._last_override_reason = "fall_recovery"
            return self.fallback_prompt

        if (
            observation is not None
            and observation.consecutive_stale_steps >= self.stale_steps_threshold
        ):
            self._last_override_reason = "stale_tracking"
            return self.fallback_prompt

        if self._should_query_selector(context):
            self.current_prompt = self.selector.choose_prompt(observation)
            self._last_query_block = context.block_count

        return self.current_prompt

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
