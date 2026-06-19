from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from mjlab.scripts.train import TrainConfig, launch_training

from mjlab_vla.scripts.config import NormalizedMotionConfig
from mjlab_vla.tracking import TASK_NAME, set_motion_file


@dataclass(kw_only=True)
class TrainCommand(NormalizedMotionConfig):
    num_envs: int = 4096
    max_iterations: int = 10000
    logger: Literal["tensorboard", "wandb"] = "wandb"
    experiment_name: str = "textop_tracking"
    run_name: str = "walk_scratch"
    resume: bool = False
    load_run: str = ".*"
    load_checkpoint: str = "model_.*.pt"


def train_textop_motion(
    cfg: TrainCommand,
    *,
    motion_file: Path,
) -> None:
    train_cfg = TrainConfig.from_task(TASK_NAME)
    set_motion_file(train_cfg.env, motion_file)

    train_cfg.env.scene.num_envs = cfg.num_envs
    train_cfg.agent.max_iterations = cfg.max_iterations
    train_cfg.agent.logger = cfg.logger
    train_cfg.agent.experiment_name = cfg.experiment_name
    train_cfg.agent.run_name = cfg.run_name
    train_cfg.agent.resume = cfg.resume
    train_cfg.agent.load_run = cfg.load_run
    train_cfg.agent.load_checkpoint = cfg.load_checkpoint

    launch_training(TASK_NAME, train_cfg)
