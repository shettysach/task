from __future__ import annotations

from mjlab_textop.robotmdar.feedback import (
    FeedbackObservation,
    parse_feedback_observation,
)
from mjlab_textop.robotmdar.planner import (
    ConstantPromptSelector,
    FeedbackPlanner,
    ManualPromptPlanner,
    PlannerContext,
)


class _FakeObservationProvider:
    def __init__(self, observation: FeedbackObservation | None = None) -> None:
        self.observation = observation
        self.age_seconds: float | None = None
        self.started = False
        self.closed = False

    def start(self) -> None:
        self.started = True

    def close(self) -> None:
        self.closed = True

    def latest(self) -> FeedbackObservation | None:
        return self.observation

    def latest_age_seconds(self) -> float | None:
        return self.age_seconds


def _observation(
    *,
    consecutive_stale_steps: int = 0,
    fallen: bool = False,
    fall_reason: str | None = None,
) -> FeedbackObservation:
    return FeedbackObservation(
        frame=10,
        started=True,
        current_frame=10,
        latest_frame=18,
        lag_frames=8,
        buffer_frames=32,
        stale_steps=0,
        consecutive_stale_steps=consecutive_stale_steps,
        fallen=fallen,
        fall_reason=fall_reason,
        robot_anchor_pos_w=(1.0, 2.0, 3.0),
        robot_anchor_quat_w=(1.0, 0.0, 0.0, 0.0),
    )


def test_parse_feedback_observation() -> None:
    observation = parse_feedback_observation(
        {
            "frame": 10,
            "started": True,
            "current_frame": 10,
            "latest_frame": 18,
            "lag_frames": 8,
            "buffer_frames": 32,
            "stale_steps": 0,
            "consecutive_stale_steps": 0,
            "fallen": True,
            "fall_reason": "anchor_height_below_0.35",
            "robot_anchor_pos_w": [1.0, 2.0, 3.0],
            "robot_anchor_quat_w": [1.0, 0.0, 0.0, 0.0],
        }
    )

    assert observation.robot_anchor_pos_w == (1.0, 2.0, 3.0)
    assert observation.robot_anchor_quat_w == (1.0, 0.0, 0.0, 0.0)
    assert observation.latest_frame == 18
    assert observation.fallen is True
    assert observation.fall_reason == "anchor_height_below_0.35"


def test_manual_prompt_planner_uses_current_prompt_without_starting_thread() -> None:
    planner = ManualPromptPlanner("walk forward")

    assert planner.choose_prompt(PlannerContext(frame_index=0, block_count=0)) == (
        "walk forward"
    )

    planner.prompt.text = "turn left"

    assert planner.choose_prompt(PlannerContext(frame_index=30, block_count=1)) == (
        "turn left"
    )
    assert planner.should_stop is False
    assert planner.input_active is False
    assert "Enter text prompt" in planner.log_suffix


def test_feedback_planner_queries_selector_on_cadence() -> None:
    provider = _FakeObservationProvider(_observation())
    planner = FeedbackPlanner(
        observation_provider=provider,
        selector=ConstantPromptSelector("turn left"),
        initial_prompt="walk forward",
        query_every_blocks=2,
        fallback_prompt="stand still",
        stale_steps_threshold=5,
    )

    planner.start()

    assert provider.started is True
    assert planner.choose_prompt(PlannerContext(frame_index=0, block_count=0)) == (
        "turn left"
    )
    assert planner.choose_prompt(PlannerContext(frame_index=30, block_count=1)) == (
        "turn left"
    )
    assert planner.choose_prompt(PlannerContext(frame_index=60, block_count=2)) == (
        "turn left"
    )

    planner.request_stop()

    assert provider.closed is True


def test_feedback_planner_falls_back_on_stale_tracking() -> None:
    provider = _FakeObservationProvider(_observation())
    planner = FeedbackPlanner(
        observation_provider=provider,
        selector=ConstantPromptSelector("turn left"),
        initial_prompt="walk forward",
        query_every_blocks=10,
        fallback_prompt="stand still",
        stale_steps_threshold=5,
    )

    assert planner.choose_prompt(PlannerContext(frame_index=0, block_count=0)) == (
        "turn left"
    )

    provider.observation = _observation(consecutive_stale_steps=5)

    assert planner.choose_prompt(PlannerContext(frame_index=30, block_count=1)) == (
        "stand still"
    )
    assert "stale_tracking" in planner.log_suffix

    provider.observation = _observation()

    assert planner.choose_prompt(PlannerContext(frame_index=60, block_count=2)) == (
        "turn left"
    )


def test_feedback_planner_falls_back_during_fall_recovery() -> None:
    provider = _FakeObservationProvider(_observation())
    planner = FeedbackPlanner(
        observation_provider=provider,
        selector=ConstantPromptSelector("turn left"),
        initial_prompt="walk forward",
        query_every_blocks=10,
        fallback_prompt="stand still",
        stale_steps_threshold=5,
        fall_recovery_blocks=3,
    )

    assert planner.choose_prompt(PlannerContext(frame_index=0, block_count=0)) == (
        "turn left"
    )

    provider.observation = _observation(
        fallen=True,
        fall_reason="anchor_height_below_0.35",
    )

    assert planner.choose_prompt(PlannerContext(frame_index=30, block_count=1)) == (
        "stand still"
    )
    assert "fallen:anchor_height_below_0.35" in planner.log_suffix

    provider.observation = _observation()

    assert planner.choose_prompt(PlannerContext(frame_index=60, block_count=2)) == (
        "stand still"
    )
    assert "fall_recovery" in planner.log_suffix
    assert planner.choose_prompt(PlannerContext(frame_index=90, block_count=4)) == (
        "turn left"
    )


def test_feedback_planner_keeps_current_prompt_when_feedback_is_old() -> None:
    provider = _FakeObservationProvider(_observation())
    provider.age_seconds = 10.0
    planner = FeedbackPlanner(
        observation_provider=provider,
        selector=ConstantPromptSelector("turn left"),
        initial_prompt="walk forward",
        query_every_blocks=1,
        fallback_prompt="stand still",
        stale_steps_threshold=5,
        feedback_timeout_sec=1.0,
    )

    assert planner.choose_prompt(PlannerContext(frame_index=0, block_count=0)) == (
        "walk forward"
    )
