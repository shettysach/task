from __future__ import annotations

import copy
from dataclasses import dataclass, fields

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.tracking.mdp.commands import MotionCommand, MotionCommandCfg

from mjlab_textop.core.schema import TEXTOP_FUTURE_STEPS


def make_future_time_steps(
    time_steps: torch.Tensor,
    *,
    future_steps: int,
    time_step_total: int,
) -> torch.Tensor:
    offsets = torch.arange(
        future_steps,
        dtype=torch.long,
        device=time_steps.device,
    )
    future = time_steps[:, None] + offsets[None, :]
    return torch.clamp(future, max=time_step_total - 1)


@dataclass(kw_only=True)
class TextOpMotionCommandCfg(MotionCommandCfg):
    future_steps: int = TEXTOP_FUTURE_STEPS

    def __post_init__(self) -> None:
        if self.future_steps <= 0:
            raise ValueError(f"future_steps must be positive, got {self.future_steps}")

    def build(self, env: ManagerBasedRlEnv) -> TextOpMotionCommand:
        return TextOpMotionCommand(self, env)


class TextOpMotionCommand(MotionCommand):
    cfg: TextOpMotionCommandCfg

    @property
    def future_time_steps(self) -> torch.Tensor:
        return make_future_time_steps(
            self.time_steps,
            future_steps=self.cfg.future_steps,
            time_step_total=self.motion.time_step_total,
        )

    @property
    def future_joint_pos(self) -> torch.Tensor:
        return self.motion.joint_pos[self.future_time_steps]

    @property
    def future_joint_vel(self) -> torch.Tensor:
        return self.motion.joint_vel[self.future_time_steps]

    @property
    def future_anchor_pos_w(self) -> torch.Tensor:
        return (
            self.motion.body_pos_w[
                self.future_time_steps,
                self.motion_anchor_body_index,
            ]
            + self._env.scene.env_origins[:, None, :]
        )

    @property
    def future_anchor_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[
            self.future_time_steps,
            self.motion_anchor_body_index,
        ]


def textop_motion_command_cfg_from(
    cfg: MotionCommandCfg,
    *,
    future_steps: int = TEXTOP_FUTURE_STEPS,
) -> TextOpMotionCommandCfg:
    kwargs = {
        field.name: copy.deepcopy(getattr(cfg, field.name))
        for field in fields(MotionCommandCfg)
    }
    return TextOpMotionCommandCfg(**kwargs, future_steps=future_steps)


def use_textop_motion_command(
    env_cfg,
    *,
    command_name: str = "motion",
    future_steps: int = TEXTOP_FUTURE_STEPS,
) -> None:
    motion_cfg = env_cfg.commands[command_name]
    if not isinstance(motion_cfg, MotionCommandCfg):
        raise TypeError(
            f"Expected env_cfg.commands[{command_name!r}] to be MotionCommandCfg, "
            f"got {type(motion_cfg).__name__}"
        )

    env_cfg.commands[command_name] = textop_motion_command_cfg_from(
        motion_cfg,
        future_steps=future_steps,
    )
