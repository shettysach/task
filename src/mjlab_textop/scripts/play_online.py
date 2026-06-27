from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import tyro

from mjlab_textop.core.contract import TEXTOP_FUTURE_STEPS
from mjlab_textop.core.online.replay import make_mjlab_npz_replay_source
from mjlab_textop.core.task import (
    ensure_textop_task_registered,
    register_online_textop_onnx_task,
    register_online_textop_replay_task,
)
from mjlab_textop.scripts.policy import ResolvedPolicy, run_textop_play


@dataclass(kw_only=True)
class PlayOnlineCommand:
    motion_file: str = field(default=tyro.MISSING)
    checkpoint_file: str | None = None
    onnx_file: str | None = None
    device: str = "cuda:0"
    num_envs: int = 1
    viewer: Literal["auto", "native", "viser"] = "auto"
    future_steps: int = TEXTOP_FUTURE_STEPS
    block_size: int = 8
    max_stale_steps: int = 25
    reset_robot_to_reference: bool = True
    anchor_alignment: Literal["align_to_robot_start", "direct_world"] = (
        "align_to_robot_start"
    )


def play_online_textop_motion(
    cfg: PlayOnlineCommand,
    *,
    motion_file: Path,
    policy: ResolvedPolicy,
) -> None:
    ensure_textop_task_registered()
    source = make_mjlab_npz_replay_source(motion_file, block_size=cfg.block_size)
    if policy.kind == "onnx":
        task_name = register_online_textop_onnx_task(
            source=source,
            source_mode="replay",
            future_steps=cfg.future_steps,
            num_envs=cfg.num_envs,
            anchor_alignment=cfg.anchor_alignment,
            max_stale_steps=cfg.max_stale_steps,
            reset_robot_to_reference=cfg.reset_robot_to_reference,
        )
    else:
        task_name = register_online_textop_replay_task(
            source=source,
            future_steps=cfg.future_steps,
            num_envs=cfg.num_envs,
            anchor_alignment=cfg.anchor_alignment,
            max_stale_steps=cfg.max_stale_steps,
            reset_robot_to_reference=cfg.reset_robot_to_reference,
        )

    run_textop_play(task_name, policy.file, cfg)
