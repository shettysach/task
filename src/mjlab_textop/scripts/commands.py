from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
from uuid import uuid4

import tyro
from mjlab.scripts.play import PlayConfig, run_play

from mjlab_textop.core.feedback.fall import FallDetectionCfg
from mjlab_textop.core.feedback.image import (
    ObservationImageStore,
    register_observation_image_store,
    unregister_observation_image_store,
)
from mjlab_textop.core.feedback.observation import UdpObservationPublisherCfg
from mjlab_textop.core.online.live import (
    SocketTextOpOnlineSource,
    SocketTextOpSourceCfg,
)
from mjlab_textop.core.online.live_registry import (
    register_live_textop_source,
    unregister_live_textop_source,
)
from mjlab_textop.core.online.replay import make_mjlab_npz_replay_source
from mjlab_textop.core.schema import TEXTOP_FUTURE_STEPS
from mjlab_textop.core.task import ensure_textop_task_registered
from mjlab_textop.scripts.live_play import FeedbackImageCaptureCfg, run_live_play
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
    feedback_image: bool = False
    feedback_image_every_frames: int = 20
    feedback_image_width: int = 320
    feedback_image_height: int = 240
    feedback_image_jpeg_quality: int = 60
    fall_min_anchor_height: float | None = 0.35
    fall_min_anchor_up_z: float | None = 0.3623577544766736


def play_live_textop_motion(
    cfg: PlayLiveCommand,
    *,
    policy: ResolvedPolicy,
) -> None:
    ensure_textop_task_registered()
    source = SocketTextOpOnlineSource(
        SocketTextOpSourceCfg(
            host=cfg.host,
            port=cfg.port,
            fps=cfg.fps,
            max_queue_blocks=cfg.max_queue_blocks,
        )
    )
    source.start()
    if cfg.feedback_image and cfg.feedback_port is None:
        raise ValueError("--feedback-image requires --feedback-port")
    image_store_key = f"play-live-feedback-image-{uuid4().hex}"
    image_store = ObservationImageStore() if cfg.feedback_image else None
    if image_store is not None:
        register_observation_image_store(image_store_key, image_store)
    observation_publisher_cfg = (
        UdpObservationPublisherCfg(
            host=cfg.feedback_host,
            port=cfg.feedback_port,
            image_store_key=image_store_key if image_store is not None else None,
        )
        if cfg.feedback_port is not None
        else None
    )
    source_key = register_live_textop_source(source)
    try:
        task_name = register_textop_play_task(
            policy=policy,
            source_key=source_key,
            source_mode="live",
            future_steps=cfg.future_steps,
            num_envs=cfg.num_envs,
            anchor_alignment=cfg.anchor_alignment,
            observation_publisher_cfg=observation_publisher_cfg,
            observation_publish_interval=cfg.feedback_every_frames,
            reset_robot_to_reference=cfg.reset_robot_to_reference,
            fall_detection=FallDetectionCfg(
                min_anchor_height=cfg.fall_min_anchor_height,
                min_anchor_up_z=cfg.fall_min_anchor_up_z,
            ),
        )
        play_cfg = PlayConfig(
            agent="trained",
            checkpoint_file=str(policy.file),
            num_envs=cfg.num_envs,
            device=cfg.device,
            viewer=cfg.viewer,
        )
        if image_store is None:
            run_play(task_name, play_cfg)
        else:
            run_live_play(
                task_name,
                play_cfg,
                image_capture_cfg=FeedbackImageCaptureCfg(
                    store=image_store,
                    every_steps=cfg.feedback_image_every_frames,
                    width=cfg.feedback_image_width,
                    height=cfg.feedback_image_height,
                    jpeg_quality=cfg.feedback_image_jpeg_quality,
                ),
            )
    finally:
        unregister_observation_image_store(image_store_key)
        unregister_live_textop_source(source_key)
        source.close()


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
