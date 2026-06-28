from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from mjlab_textop.core.feedback.observation import TextOpObservationPublisher
from mjlab_textop.core.mdp.online_commands import TextOpOnlineSourceMode
from mjlab_textop.core.online.source import TextOpOnlineSource
from mjlab_textop.core.task import (
    register_online_textop_onnx_task,
    register_online_textop_task,
)


def verify_resolved(resolved: Path, label: str) -> Path:
    if not resolved.exists():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} is not a file: {resolved}")
    return resolved


@dataclass(frozen=True)
class ResolvedPolicy:
    kind: Literal["checkpoint", "onnx"]
    file: Path


def resolve_policy(
    checkpoint_file: str | Path | None,
    onnx_file: str | Path | None,
) -> ResolvedPolicy:
    if checkpoint_file is not None and onnx_file is not None:
        raise ValueError("Pass exactly one of --checkpoint-file or --onnx-file")

    if checkpoint_file is not None:
        return ResolvedPolicy(
            "checkpoint",
            verify_resolved(
                Path(checkpoint_file).expanduser().resolve(),
                "Checkpoint file",
            ),
        )

    if onnx_file is not None:
        return ResolvedPolicy(
            "onnx",
            verify_resolved(
                Path(onnx_file).expanduser().resolve(),
                "ONNX policy file",
            ),
        )

    raise ValueError("Pass exactly one of --checkpoint-file or --onnx-file")


def register_textop_play_task(
    *,
    policy: ResolvedPolicy,
    source: TextOpOnlineSource | None = None,
    source_key: str | None = None,
    source_mode: TextOpOnlineSourceMode,
    future_steps: int,
    num_envs: int,
    anchor_alignment: Literal["align_to_robot_start", "direct_world"] = (
        "align_to_robot_start"
    ),
    reset_robot_to_reference: bool = True,
    observation_publisher: TextOpObservationPublisher | None = None,
    observation_publish_interval: int = 1,
) -> str:
    if policy.kind == "onnx":
        return register_online_textop_onnx_task(
            source=source,
            source_key=source_key,
            source_mode=source_mode,
            future_steps=future_steps,
            num_envs=num_envs,
            anchor_alignment=anchor_alignment,
            reset_robot_to_reference=reset_robot_to_reference,
            observation_publisher=observation_publisher,
            observation_publish_interval=observation_publish_interval,
        )
    else:
        return register_online_textop_task(
            source=source,
            source_key=source_key,
            source_mode=source_mode,
            future_steps=future_steps,
            num_envs=num_envs,
            anchor_alignment=anchor_alignment,
            reset_robot_to_reference=reset_robot_to_reference,
            observation_publisher=observation_publisher,
            observation_publish_interval=observation_publish_interval,
        )
