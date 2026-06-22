from __future__ import annotations

from mjlab.tasks.registry import list_tasks, load_env_cfg, load_runner_cls
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from mjlab_vla.textop.contract import TEXTOP_FUTURE_STEPS
from mjlab_vla.textop.mdp.offline_commands import TextOpMotionCommandCfg
from mjlab_vla.textop.mdp.online_commands import OnlineTextOpMotionCommandCfg
from mjlab_vla.textop.task import (
    ONLINE_TEXTOP_TASK_NAME,
    TEXTOP_TASK_NAME,
    ensure_textop_task_registered,
)


def test_textop_task_registers_once() -> None:
    ensure_textop_task_registered()
    ensure_textop_task_registered()

    assert TEXTOP_TASK_NAME in list_tasks()
    assert ONLINE_TEXTOP_TASK_NAME in list_tasks()
    assert load_runner_cls(TEXTOP_TASK_NAME) is MotionTrackingOnPolicyRunner
    assert load_runner_cls(ONLINE_TEXTOP_TASK_NAME) is MotionTrackingOnPolicyRunner


def test_textop_task_uses_textop_motion_command() -> None:
    ensure_textop_task_registered()
    env_cfg = load_env_cfg(TEXTOP_TASK_NAME)
    motion_cmd = env_cfg.commands["motion"]

    assert isinstance(motion_cmd, TextOpMotionCommandCfg)
    assert motion_cmd.future_steps == TEXTOP_FUTURE_STEPS
    assert motion_cmd.anchor_body_name == "pelvis"


def test_online_textop_task_uses_online_motion_command() -> None:
    ensure_textop_task_registered()
    env_cfg = load_env_cfg(ONLINE_TEXTOP_TASK_NAME)
    motion_cmd = env_cfg.commands["motion"]

    assert isinstance(motion_cmd, OnlineTextOpMotionCommandCfg)
    assert motion_cmd.future_steps == TEXTOP_FUTURE_STEPS
    assert motion_cmd.anchor_body_name == "pelvis"


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
