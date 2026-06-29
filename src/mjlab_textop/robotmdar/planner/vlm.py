from __future__ import annotations

import json
import urllib.request
from base64 import b64encode
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

from mjlab_textop.robotmdar.feedback import FeedbackObservation, UdpFeedbackReceiver

ALLOWED_MOTION_PROMPTS = (
    "stand still",
    "walk forward",
    "step left",
    "step right",
    "wave",
    "punch",
    "dance",
    "sit",
    "squat",
    "stop",
)

_ALLOWED_MOTION_PROMPT_SET = set(ALLOWED_MOTION_PROMPTS)


class VlmPromptPlanner:
    def __init__(
        self,
        *,
        feedback: UdpFeedbackReceiver,
        selector: "OpenAIChatPromptSelector",
        initial_prompt: str,
        query_every_blocks: int,
    ) -> None:
        if query_every_blocks <= 0:
            raise ValueError(
                f"query_every_blocks must be positive, got {query_every_blocks}"
            )
        self.feedback = feedback
        self.selector = selector
        self.current_prompt = initial_prompt
        self.query_every_blocks = query_every_blocks
        self.last_error: str | None = None
        self._stop = False
        self._future: Future[str] | None = None
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._last_query_block: int | None = None

    @property
    def should_stop(self) -> bool:
        return self._stop

    @property
    def input_active(self) -> bool:
        return False

    @property
    def log_suffix(self) -> str:
        state = "inflight" if self._future is not None else "idle"
        suffix = f" vlm={state}"
        if self.last_error is not None:
            suffix += f" vlm_error={self.last_error}"
        return suffix

    def start(self) -> None:
        self.feedback.start()

    def request_stop(self) -> None:
        self._stop = True
        self.feedback.close()
        if self._future is not None:
            self._future.cancel()
        self._executor.shutdown(wait=False, cancel_futures=True)

    def choose_prompt(self, *, block_count: int) -> str:
        self._collect_finished_request()
        observation = self.feedback.latest()
        if (
            not self._stop
            and self._future is None
            and observation is not None
            and self._should_query_selector(block_count)
        ):
            self._last_query_block = block_count
            self._future = self._executor.submit(
                self.selector.choose_prompt,
                observation=observation,
                current_prompt=self.current_prompt,
            )
        return self.current_prompt

    def _collect_finished_request(self) -> None:
        if self._future is None or not self._future.done():
            return

        try:
            self.current_prompt = self._future.result()
            self.last_error = None
        except Exception as exc:
            self.last_error = type(exc).__name__
        self._future = None

    def _should_query_selector(self, block_count: int) -> bool:
        if self._last_query_block is None:
            return True
        return block_count - self._last_query_block >= self.query_every_blocks


class OpenAIChatPromptSelector:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        system_prompt: str | None = None,
        sanitize_response: bool = True,
        timeout_sec: float = 30.0,
        max_completion_tokens: int = 32,
    ) -> None:
        if not model:
            raise ValueError("model must be a non-empty string")
        if timeout_sec <= 0:
            raise ValueError(f"timeout_sec must be positive, got {timeout_sec}")
        if max_completion_tokens <= 0:
            raise ValueError(
                f"max_completion_tokens must be positive, got {max_completion_tokens}"
            )
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.system_prompt = system_prompt
        self.sanitize_response = sanitize_response
        self.timeout_sec = timeout_sec
        self.max_completion_tokens = max_completion_tokens

    def choose_prompt(
        self,
        *,
        observation: FeedbackObservation | None,
        current_prompt: str,
    ) -> str:
        response = self._post_json(
            _make_chat_completions_payload(
                state=_make_state_payload(observation=observation),
                image_path=None if observation is None else observation.image_path,
                model=self.model,
                system_prompt=self.system_prompt,
                max_completion_tokens=self.max_completion_tokens,
            )
        )
        raw_prompt = str(response["choices"][0]["message"]["content"])
        if not self.sanitize_response:
            return raw_prompt
        return sanitize_motion_prompt(raw_prompt, fallback=current_prompt)

    def _post_json(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_sec) as response:
            return json.loads(response.read().decode("utf-8"))


def _make_state_payload(
    *,
    observation: FeedbackObservation | None,
) -> dict[str, Any]:
    if observation is None:
        return {"has_observation": False}

    return {
        "has_observation": True,
        "frame": observation.frame,
        "started": observation.started,
        "current_frame": observation.current_frame,
        "latest_frame": observation.latest_frame,
        "lag_frames": observation.lag_frames,
        "buffer_frames": observation.buffer_frames,
        "stale_steps": observation.stale_steps,
        "consecutive_stale_steps": observation.consecutive_stale_steps,
        "fallen": observation.fallen,
        "fall_reason": observation.fall_reason,
        "robot_anchor_pos_w": observation.robot_anchor_pos_w,
        "robot_anchor_quat_w": observation.robot_anchor_quat_w,
        "has_image": observation.image_path is not None,
        "image_frame": observation.image_frame,
    }


def _make_chat_completions_payload(
    *,
    state: dict[str, Any],
    image_path: str | None,
    model: str,
    system_prompt: str | None,
    max_completion_tokens: int,
) -> dict[str, Any]:
    text = (
        "Choose exactly one command from this list:\n"
        f"{_allowed_prompt_text()}\n\n"
        "Return only the command text. No punctuation. No explanation.\n"
        f"State: {json.dumps(state, separators=(',', ':'))}"
    )
    content: list[dict[str, Any]] = [{"type": "text", "text": text}]
    if image_path is not None:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": _image_data_url(Path(image_path))},
            }
        )
    messages: list[dict[str, Any]] = (
        [{"role": "system", "content": [{"type": "text", "text": system_prompt}]}]
        if system_prompt is not None
        else []
    )
    messages.append({"role": "user", "content": content})
    return {
        "model": model,
        "messages": messages,
        "max_completion_tokens": max_completion_tokens,
        "temperature": 0,
    }


def sanitize_motion_prompt(raw: str, *, fallback: str) -> str:
    text = raw.strip()
    text = text.splitlines()[0].strip()
    text = text.strip("\"'`").strip()
    text = " ".join(text.lower().split())
    return text if text in _ALLOWED_MOTION_PROMPT_SET else fallback


def _image_data_url(path: Path) -> str:
    data = b64encode(path.read_bytes()).decode("ascii")
    return f"{_image_mime_type(path)};base64,{data}"


def _image_mime_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in (".jpg", ".jpeg"):
        return "data:image/jpeg"
    return "data:image/png"


def _allowed_prompt_text() -> str:
    return "\n".join(ALLOWED_MOTION_PROMPTS)
