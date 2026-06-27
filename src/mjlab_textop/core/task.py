from __future__ import annotations

from typing import Literal
from uuid import uuid4

from mjlab.envs.mdp.observations import projected_gravity
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.tasks.registry import list_tasks, register_mjlab_task
from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg
from mjlab.tasks.tracking.config.g1.rl_cfg import unitree_g1_tracking_ppo_runner_cfg
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from mjlab_textop.core.mdp.observations import (
    future_anchor_ori_b,
    future_anchor_pos_b,
    future_joint_window,
    future_joint_window_textop_order,
    joint_pos_rel_textop_order,
    joint_vel_rel_textop_order,
    last_action_textop_order,
)
from mjlab_textop.core.mdp.offline_commands import use_textop_motion_command
from mjlab_textop.core.mdp.online_commands import (
    TextOpOnlineSourceMode,
    use_online_textop_motion_command,
)
from mjlab_textop.core.online.source import TextOpOnlineSource
from mjlab_textop.core.onnx_policy import TextOpOnnxPolicyRunner
from mjlab_textop.core.schema import TEXTOP_FUTURE_STEPS

TEXTOP_TASK_NAME = "Mjlab-TextOp-Flat-Unitree-G1"
ONLINE_TEXTOP_TASK_NAME = "Mjlab-OnlineTextOp-Flat-Unitree-G1"
ONLINE_TEXTOP_ONNX_TASK_NAME = "Mjlab-OnlineTextOpOnnx-Flat-Unitree-G1"


def make_textop_g1_flat_tracking_env_cfg(
    *,
    play: bool = False,
    future_steps: int = TEXTOP_FUTURE_STEPS,
):
    cfg = unitree_g1_flat_tracking_env_cfg(play=play)

    use_textop_motion_command(
        cfg,
        command_name="motion",
        future_steps=future_steps,
    )
    _configure_textop_actor_observations(cfg)
    _configure_textop_critic_observations(cfg)

    return cfg


def make_online_textop_g1_flat_tracking_env_cfg(
    *,
    play: bool = True,
    future_steps: int = TEXTOP_FUTURE_STEPS,
    source: TextOpOnlineSource | None = None,
    source_key: str | None = None,
    source_mode: TextOpOnlineSourceMode = "live",
    anchor_alignment: Literal["align_to_robot_start", "direct_world"] = (
        "align_to_robot_start"
    ),
    reset_robot_to_reference: bool = True,
):
    cfg = unitree_g1_flat_tracking_env_cfg(play=play)

    use_online_textop_motion_command(
        cfg,
        command_name="motion",
        future_steps=future_steps,
        source=source,
        source_key=source_key,
        source_mode=source_mode,
        anchor_alignment=anchor_alignment,
        reset_robot_to_reference=reset_robot_to_reference,
    )
    cfg.commands["motion"].anchor_body_name = "pelvis"  # ty:ignore[unresolved-attribute]
    _configure_textop_actor_observations(cfg)
    _configure_textop_critic_observations(cfg)
    _configure_online_textop_tracking_terms(cfg)

    return cfg


def make_online_textop_onnx_g1_flat_tracking_env_cfg(
    *,
    play: bool = True,
    future_steps: int = TEXTOP_FUTURE_STEPS,
    source: TextOpOnlineSource | None = None,
    source_key: str | None = None,
    source_mode: TextOpOnlineSourceMode = "live",
    anchor_alignment: Literal["align_to_robot_start", "direct_world"] = (
        "align_to_robot_start"
    ),
    reset_robot_to_reference: bool = True,
):
    cfg = unitree_g1_flat_tracking_env_cfg(play=play)

    use_online_textop_motion_command(
        cfg,
        command_name="motion",
        future_steps=future_steps,
        source=source,
        source_key=source_key,
        source_mode=source_mode,
        anchor_alignment=anchor_alignment,
        reset_robot_to_reference=reset_robot_to_reference,
    )
    cfg.commands["motion"].anchor_body_name = "pelvis"  # ty:ignore[unresolved-attribute]
    _configure_textop_onnx_actor_observations(cfg)
    _configure_online_textop_tracking_terms(cfg)

    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)

    return cfg


def register_online_textop_task(
    *,
    source: TextOpOnlineSource | None = None,
    source_key: str | None = None,
    source_mode: TextOpOnlineSourceMode,
    future_steps: int = TEXTOP_FUTURE_STEPS,
    num_envs: int = 1,
    anchor_alignment: Literal["align_to_robot_start", "direct_world"] = (
        "align_to_robot_start"
    ),
    reset_robot_to_reference: bool = True,
) -> str:
    mode_name = source_mode.capitalize()
    task_name = f"{ONLINE_TEXTOP_TASK_NAME}-{mode_name}-{uuid4().hex}"
    env_cfg = make_online_textop_g1_flat_tracking_env_cfg(
        play=True,
        future_steps=future_steps,
        source=source,
        source_key=source_key,
        source_mode=source_mode,
        anchor_alignment=anchor_alignment,
        reset_robot_to_reference=reset_robot_to_reference,
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
    source_key: str | None = None,
    source_mode: TextOpOnlineSourceMode,
    future_steps: int = TEXTOP_FUTURE_STEPS,
    num_envs: int = 1,
    anchor_alignment: Literal["align_to_robot_start", "direct_world"] = (
        "align_to_robot_start"
    ),
    reset_robot_to_reference: bool = True,
) -> str:
    mode_name = source_mode.capitalize()
    task_name = f"{ONLINE_TEXTOP_ONNX_TASK_NAME}-{mode_name}-{uuid4().hex}"
    env_cfg = make_online_textop_onnx_g1_flat_tracking_env_cfg(
        play=True,
        future_steps=future_steps,
        source=source,
        source_key=source_key,
        source_mode=source_mode,
        anchor_alignment=anchor_alignment,
        reset_robot_to_reference=reset_robot_to_reference,
    )
    env_cfg.scene.num_envs = num_envs

    register_mjlab_task(
        task_id=task_name,
        env_cfg=env_cfg,
        play_env_cfg=env_cfg,
        rl_cfg=unitree_g1_tracking_ppo_runner_cfg(),
        runner_cls=TextOpOnnxPolicyRunner,
    )
    return task_name


def _configure_textop_actor_observations(cfg) -> None:
    old_actor = cfg.observations["actor"]
    terms = {
        "future_joint_window": ObservationTermCfg(
            func=future_joint_window,
            params={"command_name": "motion"},
        ),
        "future_anchor_pos_b": ObservationTermCfg(
            func=future_anchor_pos_b,
            params={"command_name": "motion"},
            noise=Unoise(n_min=-0.25, n_max=0.25),
        ),
        "future_anchor_ori_b": ObservationTermCfg(
            func=future_anchor_ori_b,
            params={"command_name": "motion"},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        ),
        "projected_gravity": ObservationTermCfg(func=projected_gravity),
        "base_lin_vel": old_actor.terms["base_lin_vel"],
        "base_ang_vel": old_actor.terms["base_ang_vel"],
        "joint_pos": old_actor.terms["joint_pos"],
        "joint_vel": old_actor.terms["joint_vel"],
        "actions": old_actor.terms["actions"],
    }

    cfg.observations["actor"] = ObservationGroupCfg(
        terms=terms,
        concatenate_terms=True,
        enable_corruption=old_actor.enable_corruption,
    )


def _configure_textop_onnx_actor_observations(cfg) -> None:
    old_actor = cfg.observations["actor"]
    terms = {
        "future_joint_window": ObservationTermCfg(
            func=future_joint_window_textop_order,
            params={"command_name": "motion"},
        ),
        "future_anchor_pos_b": ObservationTermCfg(
            func=future_anchor_pos_b,
            params={"command_name": "motion"},
        ),
        "future_anchor_ori_b": ObservationTermCfg(
            func=future_anchor_ori_b,
            params={"command_name": "motion"},
        ),
        "projected_gravity": ObservationTermCfg(func=projected_gravity),
        "base_lin_vel": old_actor.terms["base_lin_vel"],
        "base_ang_vel": old_actor.terms["base_ang_vel"],
        "joint_pos": ObservationTermCfg(func=joint_pos_rel_textop_order),
        "joint_vel": ObservationTermCfg(func=joint_vel_rel_textop_order),
        "actions": ObservationTermCfg(func=last_action_textop_order),
    }

    cfg.observations["actor"] = ObservationGroupCfg(
        terms=terms,
        concatenate_terms=True,
        enable_corruption=False,
    )


def _configure_textop_critic_observations(cfg) -> None:
    old_critic = cfg.observations["critic"]
    terms = {
        "future_joint_window": ObservationTermCfg(
            func=future_joint_window,
            params={"command_name": "motion"},
        ),
        "future_anchor_pos_b": ObservationTermCfg(
            func=future_anchor_pos_b,
            params={"command_name": "motion"},
        ),
        "future_anchor_ori_b": ObservationTermCfg(
            func=future_anchor_ori_b,
            params={"command_name": "motion"},
        ),
        "body_pos": old_critic.terms["body_pos"],
        "body_ori": old_critic.terms["body_ori"],
        "base_lin_vel": old_critic.terms["base_lin_vel"],
        "base_ang_vel": old_critic.terms["base_ang_vel"],
        "joint_pos": old_critic.terms["joint_pos"],
        "joint_vel": old_critic.terms["joint_vel"],
        "actions": old_critic.terms["actions"],
    }

    cfg.observations["critic"] = ObservationGroupCfg(
        terms=terms,
        concatenate_terms=True,
        enable_corruption=False,
    )


def _configure_online_textop_tracking_terms(cfg) -> None:
    critic_terms = cfg.observations["critic"].terms
    critic_terms.pop("body_pos", None)
    critic_terms.pop("body_ori", None)

    for reward_name in (
        "motion_body_pos",
        "motion_body_ori",
        "motion_body_lin_vel",
        "motion_body_ang_vel",
    ):
        cfg.rewards.pop(reward_name, None)

    cfg.terminations.pop("ee_body_pos", None)


def ensure_textop_task_registered() -> None:
    task_names = list_tasks()

    if TEXTOP_TASK_NAME not in task_names:
        register_mjlab_task(
            task_id=TEXTOP_TASK_NAME,
            env_cfg=make_textop_g1_flat_tracking_env_cfg(play=False),
            play_env_cfg=make_textop_g1_flat_tracking_env_cfg(play=True),
            rl_cfg=unitree_g1_tracking_ppo_runner_cfg(),
            runner_cls=MotionTrackingOnPolicyRunner,
        )

    if ONLINE_TEXTOP_TASK_NAME not in task_names:
        register_mjlab_task(
            task_id=ONLINE_TEXTOP_TASK_NAME,
            env_cfg=make_online_textop_g1_flat_tracking_env_cfg(play=True),
            play_env_cfg=make_online_textop_g1_flat_tracking_env_cfg(play=True),
            rl_cfg=unitree_g1_tracking_ppo_runner_cfg(),
            runner_cls=MotionTrackingOnPolicyRunner,
        )

    if ONLINE_TEXTOP_ONNX_TASK_NAME not in task_names:
        register_mjlab_task(
            task_id=ONLINE_TEXTOP_ONNX_TASK_NAME,
            env_cfg=make_online_textop_onnx_g1_flat_tracking_env_cfg(play=True),
            play_env_cfg=make_online_textop_onnx_g1_flat_tracking_env_cfg(play=True),
            rl_cfg=unitree_g1_tracking_ppo_runner_cfg(),
            runner_cls=TextOpOnnxPolicyRunner,
        )


ensure_textop_task_registered()
