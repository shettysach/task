from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import tyro
from mjlab.scripts.play import PlayConfig, run_play

from mjlab_vla.tracking import TASK_NAME


@dataclass(kw_only=True)
class PlayCommand:
    normalized_motion_file: str = "/tmp/textop_walk_mjlab.npz"
    checkpoint_file: str = field(default=tyro.MISSING)
    device: str = "cuda:0"
    num_envs: int = 1
    viewer: Literal["auto", "native", "viser"] = "auto"


def play_textop_motion(
    cfg: PlayCommand,
    *,
    motion_file: Path,
    checkpoint_file: Path,
) -> None:
    play_cfg = PlayConfig(
        agent="trained",
        checkpoint_file=str(checkpoint_file),
        motion_file=str(motion_file),
        num_envs=cfg.num_envs,
        device=cfg.device,
        viewer=cfg.viewer,
    )
    run_play(TASK_NAME, play_cfg)
