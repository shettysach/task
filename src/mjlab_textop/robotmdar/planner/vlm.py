from __future__ import annotations

import json
import re
import urllib.request
from typing import Any

from mjlab_textop.robotmdar.feedback import FeedbackObservation

from .base import PlannerContext

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
_PROMPT_ALIASES = {
    "stand": "stand stable",
    "stable": "stand stable",
    "balance": "stand stable",
    "recover": "stand stable",
    "walk": "walk forward",
    "forward": "walk forward",
}
_MOTION_PROMPT_PATTERN = re.compile(r"[a-z0-9 ,.-]+")


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
        context: PlannerContext,
        current_prompt: str,
    ) -> str:
        payload = {
            "state": _make_state_payload(
                observation=observation,
                context=context,
                current_prompt=current_prompt,
            ),
        }
        if observation is not None and observation.image_data_base64 is not None:
            payload["image"] = {
                "mime_type": observation.image_mime_type or "image/jpeg",
                "data_base64": observation.image_data_base64,
            }
        response = self._post_json(
            _make_chat_completions_payload(
                payload=payload,
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
    context: PlannerContext,
    current_prompt: str,
) -> dict[str, Any]:
    state: dict[str, Any] = {
        "frame_index": context.frame_index,
        "block_count": context.block_count,
        "current_prompt": current_prompt,
    }
    if observation is None:
        state["has_observation"] = False
        return state

    state.update(
        {
            "has_observation": True,
            "fallen": observation.fallen,
            "fall_reason": observation.fall_reason,
            "lag_frames": observation.lag_frames,
            "buffer_frames": observation.buffer_frames,
            "stale_steps": observation.stale_steps,
            "consecutive_stale_steps": observation.consecutive_stale_steps,
            "has_image": observation.image_data_base64 is not None,
            "image_frame": observation.image_frame,
        }
    )
    return state


def _make_chat_completions_payload(
    *,
    payload: dict[str, Any],
    model: str,
    system_prompt: str | None,
    max_completion_tokens: int,
) -> dict[str, Any]:
    state = payload["state"]
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
    content = [{"type": "text", "text": text}]
    image = payload.get("image")
    if image is not None:
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": (
                        f"data:{image['mime_type']};base64,"
                        f"{image['data_base64']}"
                    )
                },
            }
        )
    messages.append({"role": "user", "content": content})
    return {
        "model": model,
        "messages": messages,
        "max_tokens": max_completion_tokens,
        "max_completion_tokens": max_completion_tokens,
        "temperature": 0,
    }


def sanitize_motion_prompt(raw: str, *, fallback: str) -> str:
    text = raw.strip()
    text = text.splitlines()[0].strip()
    text = text.strip("\"'`").strip()
    text = re.sub(r"\s+", " ", text).lower()

    if len(text) > 48:
        return fallback
    if not _MOTION_PROMPT_PATTERN.fullmatch(text):
        return fallback
    if text in _ALLOWED_MOTION_PROMPT_SET:
        return text
    if text in _PROMPT_ALIASES:
        return _PROMPT_ALIASES[text]

    for key, value in _PROMPT_ALIASES.items():
        if key in text:
            return value
    return fallback


def _allowed_prompt_text() -> str:
    return "\n".join(ALLOWED_MOTION_PROMPTS)
