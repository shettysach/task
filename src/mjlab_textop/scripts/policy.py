from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from mjlab.scripts.play import PlayConfig, run_play


@dataclass(frozen=True)
class ResolvedPolicy:
    kind: Literal["checkpoint", "onnx"]
    file: Path


def resolve_path(path: str) -> Path:
    return Path(path).expanduser().resolve()


def verify_resolved(resolved: Path, label: str) -> Path:
    if not resolved.exists():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} is not a file: {resolved}")
    return resolved


def verify_path(path: str, label: str) -> Path:
    return verify_resolved(resolve_path(path), label)


def resolve_policy(
    *,
    checkpoint_file: str | None,
    onnx_file: str | None,
) -> ResolvedPolicy:
    if (checkpoint_file is None) == (onnx_file is None):
        raise ValueError("Pass exactly one of --checkpoint-file or --onnx-file")

    if checkpoint_file is not None:
        return ResolvedPolicy(
            "checkpoint",
            verify_path(checkpoint_file, "Checkpoint file"),
        )

    assert onnx_file is not None
    return ResolvedPolicy("onnx", verify_path(onnx_file, "ONNX policy file"))


def run_textop_play(task_name: str, policy_file: Path, cfg) -> None:
    play_cfg = PlayConfig(
        agent="trained",
        checkpoint_file=str(policy_file),
        num_envs=cfg.num_envs,
        device=cfg.device,
        viewer=cfg.viewer,
    )
    run_play(task_name, play_cfg)
