from __future__ import annotations

import json
import urllib.request
from typing import Any

from mjlab_textop.robotmdar.feedback import FeedbackObservation

from .base import PlannerContext


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
        return str(response["choices"][0]["message"]["content"])

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
        "Return only one short RobotMDAR motion prompt\n"
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
