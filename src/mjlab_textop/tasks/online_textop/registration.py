from __future__ import annotations

from typing import Literal
from uuid import uuid4

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from mjlab_textop.core.feedback.observation import OnlineTextOpObservationCfg
from mjlab_textop.core.mdp.online_commands import TextOpOnlineSourceMode
from mjlab_textop.core.online.live import SocketTextOpSourceCfg
from mjlab_textop.core.online.source import TextOpOnlineSource
from mjlab_textop.core.onnx_policy import CustomOnnxPolicyRunner
from mjlab_textop.core.schema import TEXTOP_FUTURE_STEPS
from mjlab_textop.core.task import StaticTaskSpec
from mjlab_textop.tasks.online_textop.env_cfg import (
    make_online_textop_g1_flat_tracking_env_cfg,
    make_online_textop_onnx_g1_flat_tracking_env_cfg,
)
from mjlab_textop.tasks.online_textop.ppo_cfg import (
    unitree_g1_tracking_ppo_runner_cfg,
)

ONLINE_TEXTOP_TASK_NAME = "Mjlab-OnlineTextOp-Flat-Unitree-G1"
ONLINE_TEXTOP_ONNX_TASK_NAME = "Mjlab-OnlineTextOpOnnx-Flat-Unitree-G1"

STATIC_TASK_SPECS = [
    StaticTaskSpec(
        task_id=ONLINE_TEXTOP_TASK_NAME,
        # Online TextOp currently runs as a single-env play task; train and play
        # configs intentionally share the same factory until batching exists.
        make_env_cfg=lambda: make_online_textop_g1_flat_tracking_env_cfg(play=True),
        make_play_env_cfg=lambda: make_online_textop_g1_flat_tracking_env_cfg(
            play=True
        ),
        make_rl_cfg=unitree_g1_tracking_ppo_runner_cfg,
        runner_cls=MotionTrackingOnPolicyRunner,
    ),
    StaticTaskSpec(
        task_id=ONLINE_TEXTOP_ONNX_TASK_NAME,
        make_env_cfg=lambda: make_online_textop_onnx_g1_flat_tracking_env_cfg(
            play=True
        ),
        make_play_env_cfg=lambda: make_online_textop_onnx_g1_flat_tracking_env_cfg(
            play=True
        ),
        make_rl_cfg=unitree_g1_tracking_ppo_runner_cfg,
        runner_cls=CustomOnnxPolicyRunner,
    ),
]


def register_online_textop_task(
    *,
    source: TextOpOnlineSource | None = None,
    live_source_cfg: SocketTextOpSourceCfg | None = None,
    source_mode: TextOpOnlineSourceMode,
    future_steps: int = TEXTOP_FUTURE_STEPS,
    num_envs: int = 1,
    anchor_alignment: Literal["align_to_robot_start", "direct_world"] = (
        "align_to_robot_start"
    ),
    reset_robot_to_reference: bool = True,
    observation: OnlineTextOpObservationCfg | None = None,
) -> str:
    mode_name = source_mode.capitalize()
    task_name = f"{ONLINE_TEXTOP_TASK_NAME}-{mode_name}-{uuid4().hex}"
    env_cfg = make_online_textop_g1_flat_tracking_env_cfg(
        play=True,
        future_steps=future_steps,
        source=source,
        live_source_cfg=live_source_cfg,
        source_mode=source_mode,
        anchor_alignment=anchor_alignment,
        reset_robot_to_reference=reset_robot_to_reference,
        observation=observation,
    )
    env_cfg.scene.num_envs = num_envs

    register_mjlab_task(
        task_id=task_name,
        env_cfg=env_cfg,
        play_env_cfg=env_cfg,
        rl_cfg=unitree_g1_tracking_ppo_runner_cfg(),
        runner_cls=MotionTrackingOnPolicyRunner,
    )
    return task_name


def register_online_textop_onnx_task(
    *,
    source: TextOpOnlineSource | None = None,
    live_source_cfg: SocketTextOpSourceCfg | None = None,
    source_mode: TextOpOnlineSourceMode,
    future_steps: int = TEXTOP_FUTURE_STEPS,
    num_envs: int = 1,
    anchor_alignment: Literal["align_to_robot_start", "direct_world"] = (
        "align_to_robot_start"
    ),
    reset_robot_to_reference: bool = True,
    observation: OnlineTextOpObservationCfg | None = None,
) -> str:
    mode_name = source_mode.capitalize()
    task_name = f"{ONLINE_TEXTOP_ONNX_TASK_NAME}-{mode_name}-{uuid4().hex}"
    env_cfg = make_online_textop_onnx_g1_flat_tracking_env_cfg(
        play=True,
        future_steps=future_steps,
        source=source,
        live_source_cfg=live_source_cfg,
        source_mode=source_mode,
        anchor_alignment=anchor_alignment,
        reset_robot_to_reference=reset_robot_to_reference,
        observation=observation,
    )
    env_cfg.scene.num_envs = num_envs

    register_mjlab_task(
        task_id=task_name,
        env_cfg=env_cfg,
        play_env_cfg=env_cfg,
        rl_cfg=unitree_g1_tracking_ppo_runner_cfg(),
        runner_cls=CustomOnnxPolicyRunner,
    )
    return task_name
