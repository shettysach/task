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
    ) -> None:
        if query_every_blocks <= 0:
            raise ValueError(
                f"query_every_blocks must be positive, got {query_every_blocks}"
            )
        self.observation_provider = observation_provider
        self.selector = selector
        self.current_prompt = initial_prompt
        self.query_every_blocks = query_every_blocks
        self._stop = False
        self._last_query_block: int | None = None

    @property
    def should_stop(self) -> bool:
        return self._stop

    @property
    def input_active(self) -> bool:
        return False

    @property
    def log_suffix(self) -> str:
        return ""

    def start(self) -> None:
        self.observation_provider.start()

    def request_stop(self) -> None:
        self._stop = True
        self.observation_provider.close()

    def choose_prompt(self, context: PlannerContext) -> str:
        if self._should_query_selector(context):
            self._last_query_block = context.block_count
            self.current_prompt = self.selector.choose_prompt(
                observation=self.observation_provider.latest(),
                current_prompt=self.current_prompt,
            )

        return self.current_prompt

    def _should_query_selector(self, context: PlannerContext) -> bool:
        if self._last_query_block is None:
            return True
        return context.block_count - self._last_query_block >= self.query_every_blocks
