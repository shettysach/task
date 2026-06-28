from __future__ import annotations

import json

from mjlab_textop.robotmdar.feedback import (
    FeedbackObservation,
    parse_feedback_observation,
)
from mjlab_textop.robotmdar.planner import (
    ConstantPromptSelector,
    FeedbackPlanner,
    ManualPromptPlanner,
    OpenAIChatPromptSelector,
    PlannerContext,
)
from mjlab_textop.robotmdar.planner.vlm import sanitize_motion_prompt


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


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


class _FailingSelector:
    def __init__(self) -> None:
        self.calls = 0

    def choose_prompt(self, **kwargs) -> str:
        del kwargs
        self.calls += 1
        raise TimeoutError("vlm timed out")


def _observation(
    *,
    consecutive_stale_steps: int = 0,
    fallen: bool = False,
    fall_reason: str | None = None,
    image_data_base64: str | None = None,
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
        image_mime_type="image/jpeg" if image_data_base64 is not None else None,
        image_data_base64=image_data_base64,
        image_frame=9 if image_data_base64 is not None else None,
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
            "image": {
                "mime_type": "image/jpeg",
                "data_base64": "abc123",
                "frame": 9,
            },
        }
    )

    assert observation.robot_anchor_pos_w == (1.0, 2.0, 3.0)
    assert observation.robot_anchor_quat_w == (1.0, 0.0, 0.0, 0.0)
    assert observation.latest_frame == 18
    assert observation.fallen is True
    assert observation.fall_reason == "anchor_height_below_0.35"
    assert observation.image_mime_type == "image/jpeg"
    assert observation.image_data_base64 == "abc123"
    assert observation.image_frame == 9


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


def test_sanitize_motion_prompt_accepts_allowed_prompts() -> None:
    assert sanitize_motion_prompt("turn left", fallback="stand stable") == "turn left"
    assert (
        sanitize_motion_prompt('"walk forward"\nextra text', fallback="stand stable")
        == "walk forward"
    )


def test_sanitize_motion_prompt_maps_aliases() -> None:
    assert sanitize_motion_prompt("recover", fallback="walk forward") == "stand stable"
    assert sanitize_motion_prompt("please walk now", fallback="stand stable") == (
        "walk forward"
    )


def test_sanitize_motion_prompt_rejects_garbage() -> None:
    assert sanitize_motion_prompt("import math463i[y>3]?", fallback="stand stable") == (
        "stand stable"
    )
    assert sanitize_motion_prompt("x" * 80, fallback="walk forward") == (
        "walk forward"
    )


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


def test_feedback_planner_throttles_failed_selector_queries() -> None:
    provider = _FakeObservationProvider(_observation())
    selector = _FailingSelector()
    planner = FeedbackPlanner(
        observation_provider=provider,
        selector=selector,
        initial_prompt="walk forward",
        query_every_blocks=3,
        fallback_prompt="stand still",
        stale_steps_threshold=5,
    )

    assert planner.choose_prompt(PlannerContext(frame_index=0, block_count=0)) == (
        "walk forward"
    )
    assert planner.choose_prompt(PlannerContext(frame_index=8, block_count=1)) == (
        "walk forward"
    )
    assert planner.choose_prompt(PlannerContext(frame_index=16, block_count=2)) == (
        "walk forward"
    )
    assert selector.calls == 1

    assert planner.choose_prompt(PlannerContext(frame_index=24, block_count=3)) == (
        "walk forward"
    )
    assert selector.calls == 2


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


def test_http_vlm_prompt_selector_posts_context_and_observation(monkeypatch) -> None:
    posted = {}

    def fake_urlopen(request, timeout):
        posted["url"] = request.full_url
        posted["timeout"] = timeout
        posted["payload"] = json.loads(request.data.decode("utf-8"))
        posted["content_type"] = request.headers["Content-type"]
        return _FakeResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": "turn left",
                        }
                    }
                ]
            }
        )

    monkeypatch.setattr(
        "mjlab_textop.robotmdar.planner.vlm.urllib.request.urlopen",
        fake_urlopen,
    )
    selector = OpenAIChatPromptSelector(
        base_url="http://127.0.0.1:9379",
        model="gemma-4-e2b-it",
        system_prompt="You are a motion planner.",
        timeout_sec=1.5,
        max_completion_tokens=16,
    )

    prompt = selector.choose_prompt(
        observation=_observation(image_data_base64="abc123"),
        context=PlannerContext(frame_index=64, block_count=8),
        current_prompt="walk forward",
    )

    assert prompt == "turn left"
    assert posted["url"] == "http://127.0.0.1:9379/v1/chat/completions"
    assert posted["timeout"] == 1.5
    assert posted["content_type"] == "application/json"
    assert posted["payload"]["model"] == "gemma-4-e2b-it"
    assert posted["payload"]["max_tokens"] == 16
    assert posted["payload"]["max_completion_tokens"] == 16
    assert posted["payload"]["temperature"] == 0
    assert posted["payload"]["messages"][0]["role"] == "system"
    assert posted["payload"]["messages"][0]["content"][0]["text"] == (
        "You are a motion planner."
    )
    content = posted["payload"]["messages"][1]["content"]
    assert content[0]["type"] == "text"
    assert "Choose exactly one command from this list" in content[0]["text"]
    assert "stand stable" in content[0]["text"]
    assert "Return only the command text" in content[0]["text"]
    assert '"frame_index":64' in content[0]["text"]
    assert '"current_prompt":"walk forward"' in content[0]["text"]
    assert '"lag_frames":8' in content[0]["text"]
    assert '"has_image":true' in content[0]["text"]
    assert content[1] == {
        "type": "image_url",
        "image_url": {"url": "data:image/jpeg;base64,abc123"},
    }


def test_http_vlm_prompt_selector_sanitizes_response(monkeypatch) -> None:
    def fake_urlopen(request, timeout):
        del request, timeout
        return _FakeResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": 'STOP. Clear location near pose.g39g}<|"|>',
                        }
                    }
                ]
            }
        )

    monkeypatch.setattr(
        "mjlab_textop.robotmdar.planner.vlm.urllib.request.urlopen",
        fake_urlopen,
    )
    selector = OpenAIChatPromptSelector(
        base_url="http://127.0.0.1:9379",
        model="gemma-4-e2b-it",
    )

    assert selector.choose_prompt(
        observation=_observation(),
        context=PlannerContext(frame_index=64, block_count=8),
        current_prompt="stand stable",
    ) == "stand stable"
