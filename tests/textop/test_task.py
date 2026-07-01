from __future__ import annotations

import numpy as np
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_runner_cls
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from mjlab_textop.core.feedback.observation import OnlineTextOpObservationCfg
from mjlab_textop.core.mdp.offline_commands import TextOpMotionCommandCfg
from mjlab_textop.core.mdp.online_commands import OnlineTextOpMotionCommandCfg
from mjlab_textop.core.online.source import QueueTextOpOnlineSource, TextOpMotionBlock
from mjlab_textop.core.onnx_policy import CustomOnnxPolicyRunner
from mjlab_textop.core.schema import TEXTOP_FUTURE_STEPS
from mjlab_textop.tasks import ensure_textop_task_registered
from mjlab_textop.tasks.online_textop.registration import (
    ONLINE_TEXTOP_ONNX_TASK_NAME,
    ONLINE_TEXTOP_TASK_NAME,
    register_online_textop_onnx_task,
    register_online_textop_task,
)
from mjlab_textop.tasks.textop_tracking.registration import TEXTOP_TASK_NAME


def test_textop_task_registers_once() -> None:
    ensure_textop_task_registered()
    ensure_textop_task_registered()

    assert TEXTOP_TASK_NAME in list_tasks()
    assert ONLINE_TEXTOP_TASK_NAME in list_tasks()
    assert ONLINE_TEXTOP_ONNX_TASK_NAME in list_tasks()
    assert load_runner_cls(TEXTOP_TASK_NAME) is MotionTrackingOnPolicyRunner
    assert load_runner_cls(ONLINE_TEXTOP_TASK_NAME) is MotionTrackingOnPolicyRunner
    assert load_runner_cls(ONLINE_TEXTOP_ONNX_TASK_NAME) is CustomOnnxPolicyRunner


def test_textop_task_uses_textop_motion_command() -> None:
    ensure_textop_task_registered()
    env_cfg = load_env_cfg(TEXTOP_TASK_NAME)
    motion_cmd = env_cfg.commands["motion"]

    assert isinstance(motion_cmd, TextOpMotionCommandCfg)
    assert motion_cmd.future_steps == TEXTOP_FUTURE_STEPS
    assert motion_cmd.anchor_body_name == "torso_link"


def test_online_textop_task_uses_online_motion_command() -> None:
    ensure_textop_task_registered()
    env_cfg = load_env_cfg(ONLINE_TEXTOP_TASK_NAME)
    motion_cmd = env_cfg.commands["motion"]

    assert isinstance(motion_cmd, OnlineTextOpMotionCommandCfg)
    assert motion_cmd.future_steps == TEXTOP_FUTURE_STEPS
    assert motion_cmd.anchor_body_name == "pelvis"
    assert motion_cmd.source_mode == "live"


def test_online_textop_onnx_task_uses_online_motion_command() -> None:
    ensure_textop_task_registered()
    env_cfg = load_env_cfg(ONLINE_TEXTOP_ONNX_TASK_NAME)
    motion_cmd = env_cfg.commands["motion"]

    assert isinstance(motion_cmd, OnlineTextOpMotionCommandCfg)
    assert motion_cmd.future_steps == TEXTOP_FUTURE_STEPS
    assert motion_cmd.anchor_body_name == "pelvis"
    assert motion_cmd.source_mode == "live"


def test_online_textop_replay_task_uses_replay_source_mode() -> None:
    source = QueueTextOpOnlineSource(
        [
            TextOpMotionBlock(
                index=0,
                joint_pos=np.zeros((5, 29), dtype=np.float32),
                joint_vel=np.zeros((5, 29), dtype=np.float32),
                anchor_pos_w=np.zeros((5, 3), dtype=np.float32),
                anchor_quat_w=np.tile(
                    np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                    (5, 1),
                ),
            )
        ]
    )

    task_name = register_online_textop_task(source=source, source_mode="replay")
    env_cfg = load_env_cfg(task_name, play=True)

    assert env_cfg.commands["motion"].source_mode == "replay"


def test_online_textop_onnx_replay_task_uses_replay_source_mode() -> None:
    source = QueueTextOpOnlineSource(
        [
            TextOpMotionBlock(
                index=0,
                joint_pos=np.zeros((5, 29), dtype=np.float32),
                joint_vel=np.zeros((5, 29), dtype=np.float32),
                anchor_pos_w=np.zeros((5, 3), dtype=np.float32),
                anchor_quat_w=np.tile(
                    np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                    (5, 1),
                ),
            )
        ]
    )

    task_name = register_online_textop_onnx_task(source=source, source_mode="replay")
    env_cfg = load_env_cfg(task_name, play=True)

    assert env_cfg.commands["motion"].source_mode == "replay"
    assert load_runner_cls(task_name) is CustomOnnxPolicyRunner


def test_online_textop_replay_task_can_disable_reference_reset() -> None:
    source = QueueTextOpOnlineSource(
        [
            TextOpMotionBlock(
                index=0,
                joint_pos=np.zeros((5, 29), dtype=np.float32),
                joint_vel=np.zeros((5, 29), dtype=np.float32),
                anchor_pos_w=np.zeros((5, 3), dtype=np.float32),
                anchor_quat_w=np.tile(
                    np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                    (5, 1),
                ),
            )
        ]
    )

    task_name = register_online_textop_task(
        source=source,
        source_mode="replay",
        reset_robot_to_reference=False,
    )
    env_cfg = load_env_cfg(task_name, play=True)

    assert env_cfg.commands["motion"].reset_robot_to_reference is False


def test_online_textop_live_task_uses_live_source_mode() -> None:
    source = QueueTextOpOnlineSource([], fps=50.0)

    task_name = register_online_textop_task(
        source=source,
        source_mode="live",
        observation=OnlineTextOpObservationCfg(
            image_publish_interval=7,
        ),
    )
    env_cfg = load_env_cfg(task_name, play=True)

    assert env_cfg.commands["motion"].source_mode == "live"
    assert env_cfg.commands["motion"].observation.image_publish_interval == 7


def test_online_textop_task_removes_full_body_tracking_terms() -> None:
    ensure_textop_task_registered()
    env_cfg = load_env_cfg(ONLINE_TEXTOP_TASK_NAME)

    assert "body_pos" not in env_cfg.observations["critic"].terms
    assert "body_ori" not in env_cfg.observations["critic"].terms
    assert "motion_global_root_pos" in env_cfg.rewards
    assert "motion_global_root_ori" in env_cfg.rewards
    assert "motion_body_pos" not in env_cfg.rewards
    assert "motion_body_ori" not in env_cfg.rewards
    assert "motion_body_lin_vel" not in env_cfg.rewards
    assert "motion_body_ang_vel" not in env_cfg.rewards
    assert "ee_body_pos" not in env_cfg.terminations


def test_textop_actor_observation_order() -> None:
    ensure_textop_task_registered()
    env_cfg = load_env_cfg(TEXTOP_TASK_NAME)

    assert list(env_cfg.observations["actor"].terms) == [
        "future_joint_window",
        "future_anchor_pos_b",
        "future_anchor_ori_b",
        "projected_gravity",
        "base_lin_vel",
        "base_ang_vel",
        "joint_pos",
        "joint_vel",
        "actions",
    ]


def test_textop_onnx_actor_observation_order_and_no_corruption() -> None:
    ensure_textop_task_registered()
    env_cfg = load_env_cfg(ONLINE_TEXTOP_ONNX_TASK_NAME)

    assert list(env_cfg.observations["actor"].terms) == [
        "future_joint_window",
        "future_anchor_pos_b",
        "future_anchor_ori_b",
        "projected_gravity",
        "base_lin_vel",
        "base_ang_vel",
        "joint_pos",
        "joint_vel",
        "actions",
    ]
    assert env_cfg.observations["actor"].enable_corruption is False


def test_textop_critic_observation_order_keeps_privileged_terms() -> None:
    ensure_textop_task_registered()
    env_cfg = load_env_cfg(TEXTOP_TASK_NAME)

    assert list(env_cfg.observations["critic"].terms) == [
        "future_joint_window",
        "future_anchor_pos_b",
        "future_anchor_ori_b",
        "body_pos",
        "body_ori",
        "base_lin_vel",
        "base_ang_vel",
        "joint_pos",
        "joint_vel",
        "actions",
    ]


def test_textop_play_env_uses_start_sampling_and_no_actor_corruption() -> None:
    ensure_textop_task_registered()
    env_cfg = load_env_cfg(TEXTOP_TASK_NAME, play=True)

    assert env_cfg.commands["motion"].sampling_mode == "start"
    assert env_cfg.observations["actor"].enable_corruption is False
