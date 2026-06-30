from __future__ import annotations

import json
import threading
import time

from mjlab_textop.robotmdar.feedback import (
    FeedbackObservation,
    parse_feedback_observation,
)
from mjlab_textop.robotmdar.planner import (
    ManualPromptPlanner,
    OpenAIChatPromptSelector,
    VlmPromptPlanner,
)
from mjlab_textop.robotmdar.planner.vlm import sanitize_motion_prompt


class _FakeObservationProvider:
    def __init__(self, observation: FeedbackObservation | None = None) -> None:
        self.observation = observation
        self.started = False
        self.closed = False

    def start(self) -> None:
        self.started = True

    def close(self) -> None:
        self.closed = True

    def latest(self) -> FeedbackObservation | None:
        return self.observation


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
        self.finished = threading.Event()

    def choose_prompt(self, **kwargs) -> str:
        del kwargs
        self.calls += 1
        self.finished.set()
        raise TimeoutError("vlm timed out")


class _FixedSelector:
    def __init__(self, prompt: str) -> None:
        self.prompt = prompt
        self.calls = 0
        self.finished = threading.Event()

    def choose_prompt(self, **kwargs) -> str:
        del kwargs
        self.calls += 1
        self.finished.set()
        return self.prompt


class _BlockingSelector:
    def __init__(self, prompt: str) -> None:
        self.prompt = prompt
        self.calls = 0
        self.started = threading.Event()
        self.release = threading.Event()
        self.finished = threading.Event()

    def choose_prompt(self, **kwargs) -> str:
        del kwargs
        self.calls += 1
        self.started.set()
        self.release.wait(timeout=1)
        self.finished.set()
        return self.prompt


def _wait_for(condition) -> None:
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        if condition():
            return
        time.sleep(0.01)
    raise AssertionError("condition did not become true")


def _observation(
    *,
    consecutive_stale_steps: int = 0,
    image_path: str | None = None,
    image_frame: int | None = None,
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
        robot_anchor_pos_w=(1.0, 2.0, 3.0),
        robot_anchor_quat_w=(1.0, 0.0, 0.0, 0.0),
        image_path=image_path,
        image_frame=image_frame,
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
            "robot_anchor_pos_w": [1.0, 2.0, 3.0],
            "robot_anchor_quat_w": [1.0, 0.0, 0.0, 0.0],
            "image_path": "/tmp/mjlab_textop_latest.png",
            "image_frame": 10,
        }
    )

    assert observation.robot_anchor_pos_w == (1.0, 2.0, 3.0)
    assert observation.robot_anchor_quat_w == (1.0, 0.0, 0.0, 0.0)
    assert observation.latest_frame == 18
    assert observation.image_path == "/tmp/mjlab_textop_latest.png"
    assert observation.image_frame == 10


def test_manual_prompt_planner_uses_current_prompt_without_starting_thread() -> None:
    planner = ManualPromptPlanner("walk forward")

    assert planner.choose_prompt(block_count=0) == "walk forward"

    planner.prompt.text = "turn left"

    assert planner.choose_prompt(block_count=1) == "turn left"
    assert planner.should_stop is False
    assert planner.input_active is False
    assert "Enter text prompt" in planner.log_suffix


def test_sanitize_motion_prompt_accepts_allowed_prompts() -> None:
    assert sanitize_motion_prompt("wave", fallback="stand still") == "wave"
    assert (
        sanitize_motion_prompt('"walk forward"\nextra text', fallback="stand still")
        == "walk forward"
    )


def test_sanitize_motion_prompt_maps_aliases() -> None:
    assert sanitize_motion_prompt("recover", fallback="walk forward") == "walk forward"
    assert sanitize_motion_prompt("please walk now", fallback="stand stable") == (
        "stand stable"
    )


def test_sanitize_motion_prompt_rejects_garbage() -> None:
    assert sanitize_motion_prompt("import math463i[y>3]?", fallback="stand stable") == (
        "stand stable"
    )
    assert sanitize_motion_prompt("x" * 80, fallback="walk forward") == (
        "walk forward"
    )


def test_vlm_planner_queries_selector_on_cadence() -> None:
    provider = _FakeObservationProvider(_observation())
    selector = _FixedSelector("turn left")
    planner = VlmPromptPlanner(
        feedback=provider,
        selector=selector,
        initial_prompt="walk forward",
        query_every_blocks=2,
    )

    planner.start()

    assert provider.started is True
    assert planner.choose_prompt(block_count=0) == "walk forward"
    assert selector.finished.wait(timeout=1)
    assert planner.choose_prompt(block_count=1) == "turn left"
    assert planner.choose_prompt(block_count=2) == "turn left"
    _wait_for(lambda: selector.calls == 2)

    planner.request_stop()

    assert provider.closed is True


def test_vlm_planner_does_not_block_while_selector_runs() -> None:
    provider = _FakeObservationProvider(_observation())
    selector = _BlockingSelector("turn right")
    planner = VlmPromptPlanner(
        feedback=provider,
        selector=selector,
        initial_prompt="walk forward",
        query_every_blocks=10,
    )

    assert planner.choose_prompt(block_count=0) == "walk forward"
    assert selector.started.wait(timeout=1)
    assert planner.log_suffix == " vlm=inflight"
    assert planner.choose_prompt(block_count=1) == "walk forward"
    assert selector.calls == 1

    selector.release.set()
    assert selector.finished.wait(timeout=1)
    assert planner.choose_prompt(block_count=2) == "turn right"
    assert planner.log_suffix == " vlm=idle"

    planner.request_stop()


def test_vlm_planner_keeps_current_prompt_on_selector_errors() -> None:
    provider = _FakeObservationProvider(_observation())
    selector = _FailingSelector()
    planner = VlmPromptPlanner(
        feedback=provider,
        selector=selector,
        initial_prompt="walk forward",
        query_every_blocks=3,
    )

    assert planner.choose_prompt(block_count=0) == "walk forward"
    assert selector.finished.wait(timeout=1)
    assert planner.choose_prompt(block_count=1) == "walk forward"
    assert selector.calls == 1
    assert planner.last_error == "TimeoutError"
    assert planner.log_suffix == " vlm=idle vlm_error=TimeoutError"

    planner.request_stop()


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
                            "content": "wave",
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
        observation=_observation(),
        current_prompt="walk forward",
    )

    assert prompt == "wave"
    assert posted["url"] == "http://127.0.0.1:9379/v1/chat/completions"
    assert posted["timeout"] == 1.5
    assert posted["content_type"] == "application/json"
    assert posted["payload"]["model"] == "gemma-4-e2b-it"
    assert posted["payload"]["max_completion_tokens"] == 16
    assert posted["payload"]["temperature"] == 0
    assert posted["payload"]["messages"][0]["role"] == "system"
    assert posted["payload"]["messages"][0]["content"][0]["text"] == (
        "You are a motion planner."
    )
    content = posted["payload"]["messages"][1]["content"]
    assert content[0]["type"] == "text"
    assert "Choose exactly one command from this list" in content[0]["text"]
    assert "stand still" in content[0]["text"]
    assert "Return only the command text" in content[0]["text"]
    assert '"current_frame":10' in content[0]["text"]
    assert '"latest_frame":18' in content[0]["text"]
    assert '"lag_frames":8' in content[0]["text"]
    assert '"robot_anchor_pos_w":[1.0,2.0,3.0]' in content[0]["text"]
    assert '"has_image":false' in content[0]["text"]
    assert len(content) == 1


def test_http_vlm_prompt_selector_posts_image_from_feedback_path(
    monkeypatch,
    tmp_path,
) -> None:
    posted = {}
    image_path = tmp_path / "latest.png"
    image_path.write_bytes(b"png bytes")

    def fake_urlopen(request, timeout):
        del timeout
        posted["payload"] = json.loads(request.data.decode("utf-8"))
        return _FakeResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": "punch",
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

    prompt = selector.choose_prompt(
        observation=_observation(image_path=str(image_path), image_frame=10),
        current_prompt="walk forward",
    )

    content = posted["payload"]["messages"][0]["content"]
    assert prompt == "punch"
    assert content[0]["type"] == "text"
    assert '"has_image":true' in content[0]["text"]
    assert '"image_frame":10' in content[0]["text"]
    assert content[1] == {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,cG5nIGJ5dGVz"},
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
        current_prompt="stand stable",
    ) == "stand stable"


def test_http_vlm_prompt_selector_can_skip_sanitization(monkeypatch) -> None:
    def fake_urlopen(request, timeout):
        del request, timeout
        return _FakeResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": "turn left\nextra text",
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
        sanitize_response=False,
    )

    assert selector.choose_prompt(
        observation=_observation(),
        current_prompt="stand stable",
    ) == "turn left\nextra text"
