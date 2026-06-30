from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import tyro
from mjlab.scripts.play import PlayConfig, run_play

from mjlab_textop.core.feedback.observation import (
    OnlineTextOpObservationCfg,
    UdpObservationPublisher,
    UdpObservationPublisherCfg,
)
from mjlab_textop.core.online.live import SocketTextOpSourceCfg
from mjlab_textop.core.online.replay import make_mjlab_npz_replay_source
from mjlab_textop.core.schema import TEXTOP_FUTURE_STEPS
from mjlab_textop.core.task import ensure_textop_task_registered
from mjlab_textop.scripts.utils import ResolvedPolicy, register_textop_play_task


@dataclass(kw_only=True)
class NormalizeCommand:
    input_motion_file: str = field(default=tyro.MISSING)
    output_motion_file: str = field(default=tyro.MISSING)
    device: str = "cuda:0"
    max_frames: int | None = None


# --


@dataclass(kw_only=True)
class PlayLiveCommand:
    checkpoint_file: str | None = None
    onnx_file: str | None = None
    host: str = "127.0.0.1"
    port: int = 8765
    device: str = "cuda:0"
    num_envs: int = 1
    viewer: Literal["auto", "native", "viser"] = "auto"
    future_steps: int = TEXTOP_FUTURE_STEPS
    fps: float = 50.0
    max_queue_blocks: int = 32
    anchor_alignment: Literal["align_to_robot_start", "direct_world"] = (
        "align_to_robot_start"
    )
    reset_robot_to_reference: bool = True
    feedback_host: str = "127.0.0.1"
    feedback_port: int | None = None
    feedback_every_frames: int = 5
    feedback_image_path: str | None = None
    feedback_image_every_frames: int = 5
    feedback_image_width: int | None = 320
    feedback_image_height: int | None = 240


def play_live_textop_motion(
    cfg: PlayLiveCommand,
    *,
    policy: ResolvedPolicy,
) -> None:
    ensure_textop_task_registered()
    live_source_cfg = SocketTextOpSourceCfg(
        host=cfg.host,
        port=cfg.port,
        fps=cfg.fps,
        max_queue_blocks=cfg.max_queue_blocks,
    )
    observation_publisher_cfg = (
        UdpObservationPublisherCfg(
            host=cfg.feedback_host,
            port=cfg.feedback_port,
        )
        if cfg.feedback_port is not None
        else None
    )
    observation_publisher = (
        UdpObservationPublisher(observation_publisher_cfg)
        if observation_publisher_cfg is not None
        else None
    )
    observation = OnlineTextOpObservationCfg(
        publisher=observation_publisher,
        publish_interval=cfg.feedback_every_frames,
        image_path=cfg.feedback_image_path,
        image_publish_interval=cfg.feedback_image_every_frames,
    )
    task_name = register_textop_play_task(
        policy=policy,
        live_source_cfg=live_source_cfg,
        source_mode="live",
        future_steps=cfg.future_steps,
        num_envs=cfg.num_envs,
        anchor_alignment=cfg.anchor_alignment,
        observation=observation,
        reset_robot_to_reference=cfg.reset_robot_to_reference,
    )
    play_cfg = PlayConfig(
        agent="trained",
        checkpoint_file=str(policy.file),
        num_envs=cfg.num_envs,
        device=cfg.device,
        viewer=cfg.viewer,
        video_width=cfg.feedback_image_width if cfg.feedback_image_path else None,
        video_height=cfg.feedback_image_height if cfg.feedback_image_path else None,
    )
    run_play(task_name, play_cfg)


# --


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
    task_name = register_textop_play_task(
        policy=policy,
        source=source,
        source_mode="replay",
        future_steps=cfg.future_steps,
        num_envs=cfg.num_envs,
        anchor_alignment=cfg.anchor_alignment,
        reset_robot_to_reference=cfg.reset_robot_to_reference,
    )
    play_cfg = PlayConfig(
        agent="trained",
        checkpoint_file=str(policy.file),
        num_envs=cfg.num_envs,
        device=cfg.device,
        viewer=cfg.viewer,
    )
    run_play(task_name, play_cfg)
