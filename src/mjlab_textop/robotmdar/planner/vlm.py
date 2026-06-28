from __future__ import annotations

import json
import urllib.request
from dataclasses import asdict
from typing import Any

from mjlab_textop.robotmdar.feedback import FeedbackObservation

from .base import PlannerContext


class HttpVlmPromptSelector:
    def __init__(
        self,
        *,
        endpoint: str,
        timeout_sec: float = 2.0,
    ) -> None:
        if timeout_sec <= 0:
            raise ValueError(f"timeout_sec must be positive, got {timeout_sec}")
        self.endpoint = endpoint
        self.timeout_sec = timeout_sec

    def choose_prompt(
        self,
        *,
        observation: FeedbackObservation | None,
        context: PlannerContext,
        current_prompt: str,
    ) -> str:
        payload = {
            "frame_index": context.frame_index,
            "block_count": context.block_count,
            "current_prompt": current_prompt,
            "observation": _observation_payload(observation),
        }
        response = self._post_json(payload)
        return str(response["prompt"])

    def _post_json(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_sec) as response:
            return json.loads(response.read().decode("utf-8"))


def _observation_payload(
    observation: FeedbackObservation | None,
) -> dict[str, Any] | None:
    if observation is None:
        return None
    return asdict(observation)
