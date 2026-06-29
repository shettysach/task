from __future__ import annotations

import json
import urllib.request
from typing import Any

from mjlab_textop.robotmdar.feedback import FeedbackObservation

ALLOWED_MOTION_PROMPTS = (
    "stand still",
    "stand stable",
    "walk forward",
    "turn left",
    "turn right",
    "step left",
    "step right",
    "stop",
)

_ALLOWED_MOTION_PROMPT_SET = set(ALLOWED_MOTION_PROMPTS)


class OpenAIChatPromptSelector:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        system_prompt: str | None = None,
        timeout_sec: float = 2.0,
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
                model=self.model,
                system_prompt=self.system_prompt,
                max_completion_tokens=self.max_completion_tokens,
            )
        )
        raw_prompt = str(response["choices"][0]["message"]["content"])
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
    }


def _make_chat_completions_payload(
    *,
    state: dict[str, Any],
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
    messages: list[dict[str, Any]] = (
        [{"role": "system", "content": [{"type": "text", "text": system_prompt}]}]
        if system_prompt is not None
        else []
    )
    messages.append({"role": "user", "content": [{"type": "text", "text": text}]})
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


def _allowed_prompt_text() -> str:
    return "\n".join(ALLOWED_MOTION_PROMPTS)
