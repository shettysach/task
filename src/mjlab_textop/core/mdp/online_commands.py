from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

from mjlab_textop.core.contract import TEXTOP_FUTURE_STEPS, TEXTOP_G1_JOINT_COUNT
from mjlab_textop.core.online.buffer import (
    TextOpRollingMotionBuffer,
)
from mjlab_textop.core.online.live_registry import get_live_textop_source
from mjlab_textop.core.online.source import (
    QueueTextOpOnlineSource,
    ResettableTextOpOnlineSource,
    TextOpOnlineSource,
)

TextOpOnlineSourceMode = Literal["replay", "live"]


@dataclass(kw_only=True)
class OnlineTextOpMotionCommandCfg(CommandTermCfg):
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
    entity_name: str = "robot"
    anchor_body_name: str
    future_steps: int = TEXTOP_FUTURE_STEPS
    source: TextOpOnlineSource = field(default_factory=QueueTextOpOnlineSource)
    source_key: str | None = None
    source_mode: TextOpOnlineSourceMode = "live"
    start_frame: int = 0
    startup_timeout_steps: int = 250
    max_stale_steps: int = 25
    max_poll_blocks: int = 16
    max_buffer_frames: int | None = 512
    clear_buffer_on_reset: bool = True
    reset_robot_to_reference: bool = True
    anchor_alignment: Literal["align_to_robot_start", "direct_world"] = (
        "align_to_robot_start"
    )

    def __post_init__(self) -> None:
        if self.future_steps <= 0:
            raise ValueError(f"future_steps must be positive, got {self.future_steps}")
        if self.source_mode not in ("replay", "live"):
            raise ValueError(f"Unknown source_mode: {self.source_mode}")
        if self.source_mode == "replay" and not isinstance(
            self.source,
            ResettableTextOpOnlineSource,
        ):
            raise TypeError("Replay online source must implement reset()")

    def build(self, env: ManagerBasedRlEnv) -> OnlineTextOpMotionCommand:
        return OnlineTextOpMotionCommand(self, env)


class OnlineTextOpMotionCommand(CommandTerm):
    cfg: OnlineTextOpMotionCommandCfg

    def __init__(self, cfg: OnlineTextOpMotionCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        if self.cfg.source_mode == "live" and self.cfg.source_key:
            self.cfg.source = get_live_textop_source(self.cfg.source_key)
        if self.num_envs != 1:
            raise ValueError(
                f"Online TextOp supports one environment in v1, got {self.num_envs}"
            )
        if self.cfg.start_frame < 0:
            raise ValueError(
                f"start_frame must be non-negative, got {self.cfg.start_frame}"
            )
        self._validate_source_fps(env)

        self.robot = env.scene[cfg.entity_name]
        self.robot_anchor_body_index = self.robot.body_names.index(cfg.anchor_body_name)
        max_buffer_frames = (
            None if self.cfg.source_mode == "replay" else self.cfg.max_buffer_frames
        )
        self.buffer = TextOpRollingMotionBuffer(
            device=self.device,
            max_frames=max_buffer_frames,
        )
        self.current_frame = int(self.cfg.start_frame)
        self._started = False
        self._startup_wait_steps = 0
        self._last_stale_steps = 0
        self._consecutive_stale_steps = 0
        self._last_stale_frame: int | None = None
        self._anchor_pos_offset_w = torch.zeros(3, device=self.device)

        self.metrics["online_buffer_frames"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self.metrics["online_stale_steps"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self.metrics["online_consecutive_stale_steps"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self.metrics["online_current_frame"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self.metrics["online_latest_frame"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self.metrics["online_lag_frames"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self.metrics["online_started"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["online_queue_depth"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self.metrics["online_blocks_received"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self.metrics["online_blocks_dropped"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self.metrics["online_bad_messages"] = torch.zeros(
            self.num_envs, device=self.device
        )

    @property
    def command(self) -> torch.Tensor:
        return torch.cat([self.joint_pos, self.joint_vel], dim=-1)

    @property
    def joint_pos(self) -> torch.Tensor:
        return self.future_joint_pos[:, 0]

    @property
    def joint_vel(self) -> torch.Tensor:
        return self.future_joint_vel[:, 0]

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        return self.future_anchor_pos_w[:, 0]

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        return self.future_anchor_quat_w[:, 0]

    @property
    def future_joint_pos(self) -> torch.Tensor:
        return self._future()[0].unsqueeze(0)

    @property
    def future_joint_vel(self) -> torch.Tensor:
        return self._future()[1].unsqueeze(0)

    @property
    def future_anchor_pos_w(self) -> torch.Tensor:
        return self._future()[2].unsqueeze(0)

    @property
    def future_anchor_quat_w(self) -> torch.Tensor:
        return self._future()[3].unsqueeze(0)

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_link_pos_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_link_quat_w[:, self.robot_anchor_body_index]

    def _update_metrics(self) -> None:
        self.metrics["online_buffer_frames"][:] = float(self.buffer.frame_count)
        self.metrics["online_stale_steps"][:] = float(self._last_stale_steps)
        self.metrics["online_consecutive_stale_steps"][:] = float(
            self._consecutive_stale_steps
        )
        latest_index = self.buffer.latest_index
        self.metrics["online_current_frame"][:] = float(self.current_frame)
        self.metrics["online_latest_frame"][:] = float(
            -1 if latest_index is None else latest_index
        )
        self.metrics["online_lag_frames"][:] = float(
            0 if latest_index is None else latest_index - self.current_frame
        )
        self.metrics["online_started"][:] = float(self._started)
        diagnostics = getattr(self.cfg.source, "diagnostics", None)
        if diagnostics is not None:
            self.metrics["online_queue_depth"][:] = float(
                getattr(diagnostics, "queue_depth", 0)
            )
            self.metrics["online_blocks_received"][:] = float(
                getattr(diagnostics, "blocks_received", 0)
            )
            self.metrics["online_blocks_dropped"][:] = float(
                getattr(diagnostics, "blocks_dropped", 0)
            )
            self.metrics["online_bad_messages"][:] = float(
                getattr(diagnostics, "bad_messages", 0)
            )

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        if len(env_ids) == 0:
            return
        if self.cfg.clear_buffer_on_reset:
            self.buffer.clear()
            if self.cfg.source_mode == "replay":
                assert isinstance(self.cfg.source, ResettableTextOpOnlineSource)
                self.cfg.source.reset()
                self._poll_source()
        self.current_frame = int(self.cfg.start_frame)
        self._started = False
        self._startup_wait_steps = 0
        self._last_stale_steps = 0
        self._consecutive_stale_steps = 0
        self._last_stale_frame = None
        self._anchor_pos_offset_w.zero_()
        if self.cfg.source_mode == "replay" and self.buffer.can_start(
            self.current_frame,
            self.cfg.future_steps,
        ):
            if self.cfg.reset_robot_to_reference:
                self._reset_robot_to_reference(env_ids)
            self._started = True

    def _update_command(self) -> None:
        self._poll_source()

        if not self._started:
            start_frame = self._startup_start_frame()
            if start_frame is not None:
                self.current_frame = start_frame
                self._align_reference_anchor()
                if self.cfg.reset_robot_to_reference:
                    env_ids = torch.arange(self.num_envs, device=self.device)
                    self._reset_robot_to_reference(env_ids)
                self._started = True
                return

            self._startup_wait_steps += 1
            if self._startup_wait_steps > self.cfg.startup_timeout_steps:
                raise RuntimeError(
                    "Online TextOp buffer did not receive enough contiguous "
                    f"frames for future_steps={self.cfg.future_steps}"
                )
            return

        # V1 assumes one MJLab command update corresponds to one TextOp source
        # frame. RobotMDAR/TextOpDeploy commonly runs at 50 Hz; add explicit
        # source-FPS resampling before using streams at a different control rate.
        self.current_frame += 1

    def _poll_source(self) -> None:
        for _ in range(self.cfg.max_poll_blocks):
            block = self.cfg.source.poll()
            if block is None:
                return
            self.buffer.append_block(block)

    def _startup_start_frame(self) -> int | None:
        if self.cfg.source_mode == "live":
            return self.buffer.earliest_start_frame(self.cfg.future_steps)
        if self.buffer.can_start(self.current_frame, self.cfg.future_steps):
            return self.current_frame
        return None

    def _future(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self._started:
            return self._startup_future()

        joint_pos, joint_vel, anchor_pos_w, anchor_quat_w, stale_steps = (
            self.buffer.get_future(self.current_frame, self.cfg.future_steps)
        )
        anchor_pos_w = anchor_pos_w + self._anchor_pos_offset_w[None, :]
        self._last_stale_steps = stale_steps
        if self._last_stale_frame != self.current_frame:
            if stale_steps > 0:
                self._consecutive_stale_steps += 1
            else:
                self._consecutive_stale_steps = 0
            self._last_stale_frame = self.current_frame
        # Clamp stale future frames for now. Keep tracking consecutive stale
        # windows so live deployments can surface underruns without aborting.
        return joint_pos, joint_vel, anchor_pos_w, anchor_quat_w

    def _startup_future(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dtype = self.robot_anchor_pos_w.dtype
        joint_shape = (self.cfg.future_steps, TEXTOP_G1_JOINT_COUNT)
        joint_pos = torch.zeros(joint_shape, device=self.device, dtype=dtype)
        joint_vel = torch.zeros(joint_shape, device=self.device, dtype=dtype)
        anchor_pos_w = self.robot_anchor_pos_w[0].expand(self.cfg.future_steps, -1)
        anchor_quat_w = self.robot_anchor_quat_w[0].expand(self.cfg.future_steps, -1)
        return joint_pos, joint_vel, anchor_pos_w, anchor_quat_w

    def _align_reference_anchor(self) -> None:
        if self.cfg.anchor_alignment == "direct_world":
            self._anchor_pos_offset_w.zero_()
            return
        if self.cfg.anchor_alignment != "align_to_robot_start":
            raise ValueError(f"Unknown anchor alignment: {self.cfg.anchor_alignment}")

        _, _, anchor_pos_w, _, _ = self.buffer.get_future(self.current_frame, 1)
        self._anchor_pos_offset_w = self.robot_anchor_pos_w[0] - anchor_pos_w[0]

    def _reset_robot_to_reference(self, env_ids: torch.Tensor) -> None:
        joint_pos, joint_vel, anchor_pos_w, anchor_quat_w, _ = self.buffer.get_future(
            self.current_frame,
            1,
        )
        joint_pos = joint_pos[0].repeat(len(env_ids), 1)
        joint_vel = joint_vel[0].repeat(len(env_ids), 1)
        soft_limits = self.robot.data.soft_joint_pos_limits[env_ids]
        joint_pos = torch.clip(joint_pos, soft_limits[:, :, 0], soft_limits[:, :, 1])
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        root_pos = (anchor_pos_w[0] + self._anchor_pos_offset_w).repeat(len(env_ids), 1)
        root_quat = anchor_quat_w[0].repeat(len(env_ids), 1)
        root_vel = torch.zeros(
            len(env_ids), 6, device=self.device, dtype=root_pos.dtype
        )
        root_state = torch.cat([root_pos, root_quat, root_vel], dim=-1)
        self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)
        self.robot.reset(env_ids=env_ids)

    def _validate_source_fps(self, env: ManagerBasedRlEnv) -> None:
        fps = getattr(self.cfg.source, "fps", None)
        if fps is None:
            return
        expected_fps = 1.0 / float(env.step_dt)
        if abs(float(fps) - expected_fps) > 1.0e-4:
            raise ValueError(
                "Replay TextOp source FPS must match env control rate: "
                f"{float(fps):g} != {expected_fps:g}"
            )


def use_online_textop_motion_command(
    env_cfg,
    *,
    command_name: str = "motion",
    future_steps: int = TEXTOP_FUTURE_STEPS,
    source: TextOpOnlineSource | None = None,
    source_key: str | None = None,
    source_mode: TextOpOnlineSourceMode = "live",
    anchor_alignment: Literal["align_to_robot_start", "direct_world"] = (
        "align_to_robot_start"
    ),
    max_stale_steps: int = 25,
    reset_robot_to_reference: bool = True,
) -> None:
    motion_cfg = env_cfg.commands[command_name]
    entity_name = getattr(motion_cfg, "entity_name", "robot")
    anchor_body_name = motion_cfg.anchor_body_name
    source = source if source is not None else QueueTextOpOnlineSource()

    env_cfg.commands[command_name] = OnlineTextOpMotionCommandCfg(
        entity_name=entity_name,
        anchor_body_name=anchor_body_name,
        future_steps=future_steps,
        source=source,
        source_key=source_key,
        source_mode=source_mode,
        anchor_alignment=anchor_alignment,
        max_stale_steps=max_stale_steps,
        reset_robot_to_reference=reset_robot_to_reference,
    )
