from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import torch
import tyro
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp.commands import MotionCommand
from mjlab.tasks.tracking.mdp.metrics import (
    compute_ee_orientation_error,
    compute_ee_position_error,
    compute_joint_velocity_error,
    compute_mpkpe,
    compute_root_relative_mpkpe,
)
from mjlab.utils.torch import configure_torch_backends

from mjlab_vla.scripts.config import NormalizedMotionConfig
from mjlab_vla.tracking import TASK_NAME, get_motion_command_cfg, set_motion_file


@dataclass(kw_only=True)
class EvalCommand(NormalizedMotionConfig):
    checkpoint_file: str = field(default=tyro.MISSING)
    device: str = "cuda:0"
    num_envs: int = 1024
    output_file: str | None = None
    enable_corruption: bool = True


def evaluate_textop_motion(
    cfg: EvalCommand,
    *,
    motion_file: Path,
    checkpoint_file: Path,
) -> dict[str, float]:
    configure_torch_backends()
    env_cfg = load_env_cfg(TASK_NAME, play=False)
    agent_cfg = load_rl_cfg(TASK_NAME)

    set_motion_file(env_cfg, motion_file)
    motion_cmd = get_motion_command_cfg(env_cfg.commands)
    motion_cmd.sampling_mode = "start"
    env_cfg.observations["actor"].enable_corruption = cfg.enable_corruption
    env_cfg.events.pop("push_robot", None)
    env_cfg.scene.num_envs = cfg.num_envs

    env = ManagerBasedRlEnv(cfg=env_cfg, device=cfg.device)
    wrapped_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    try:
        runner_cls = load_runner_cls(TASK_NAME) or MjlabOnPolicyRunner
        runner = runner_cls(wrapped_env, asdict(agent_cfg), device=cfg.device)
        runner.load(
            str(checkpoint_file),
            load_cfg={"actor": True},
            strict=True,
            map_location=cfg.device,
        )
        policy = runner.get_inference_policy(device=cfg.device)

        command = cast(MotionCommand, env.command_manager.get_term("motion"))
        ee_body_names = env_cfg.terminations["ee_body_pos"].params["body_names"]
        metrics = _run_eval_rollout(
            env=wrapped_env,
            command=command,
            policy=policy,
            ee_body_names=ee_body_names,
            num_envs=cfg.num_envs,
            device=cfg.device,
            max_steps=2000,
        )
    finally:
        wrapped_env.close()

    _print_eval_metrics(metrics)
    if cfg.output_file is not None:
        output_path = Path(cfg.output_file).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"[INFO] Metrics saved to {output_path}")

    return metrics


def _run_eval_rollout(
    env: RslRlVecEnvWrapper,
    command: MotionCommand,
    policy,
    ee_body_names: tuple[str, ...],
    num_envs: int,
    device: str,
    max_steps: int,
) -> dict[str, float]:
    all_mpkpe: list[torch.Tensor] = []
    all_r_mpkpe: list[torch.Tensor] = []
    all_joint_vel_error: list[torch.Tensor] = []
    all_ee_pos_error: list[torch.Tensor] = []
    all_ee_ori_error: list[torch.Tensor] = []
    all_active: list[torch.Tensor] = []

    done_envs = torch.zeros(num_envs, dtype=torch.bool, device=device)
    success = torch.zeros(num_envs, dtype=torch.bool, device=device)
    obs = env.get_observations()

    step = 0
    print(f"[INFO] Running {num_envs} evaluation episodes...")
    while not done_envs.all():
        if step >= max_steps:
            print(
                f"[WARNING] Evaluation reached max_steps={max_steps} before all "
                "episodes completed."
            )
            break

        ref = SimpleNamespace(
            num_envs=command.num_envs,
            device=command.device,
            cfg=command.cfg,
            body_pos_w=command.body_pos_w.clone(),
            body_pos_relative_w=command.body_pos_relative_w.clone(),
            body_quat_relative_w=command.body_quat_relative_w.clone(),
            joint_vel=command.joint_vel.clone(),
        )

        with torch.no_grad():
            actions = policy(obs)
        obs, _, dones, _ = env.step(actions)

        ref.robot_body_pos_w = command.robot_body_pos_w
        ref.robot_body_quat_w = command.robot_body_quat_w
        ref.robot_joint_vel = command.robot_joint_vel
        ref_command = cast(MotionCommand, ref)

        active = ~done_envs
        all_active.append(active.float())
        all_mpkpe.append(torch.where(active, compute_mpkpe(ref_command), 0.0))
        all_r_mpkpe.append(
            torch.where(active, compute_root_relative_mpkpe(ref_command), 0.0)
        )
        all_joint_vel_error.append(
            torch.where(active, compute_joint_velocity_error(ref_command), 0.0)
        )
        all_ee_pos_error.append(
            torch.where(
                active, compute_ee_position_error(ref_command, ee_body_names), 0.0
            )
        )
        all_ee_ori_error.append(
            torch.where(
                active, compute_ee_orientation_error(ref_command, ee_body_names), 0.0
            )
        )

        terminated = env.unwrapped.termination_manager.terminated
        truncated = env.unwrapped.termination_manager.time_outs
        newly_done = dones.bool() & ~done_envs

        if newly_done.any():
            success = success | (newly_done & truncated & ~terminated)
            done_envs = done_envs | newly_done
            print(
                f"[INFO] {done_envs.sum().item()}/{num_envs} episodes completed "
                f"(step {step}, truncated={(newly_done & truncated).sum().item()}, "
                f"terminated={(newly_done & terminated).sum().item()})"
            )
        step += 1

    stacks = [
        all_mpkpe,
        all_r_mpkpe,
        all_joint_vel_error,
        all_ee_pos_error,
        all_ee_ori_error,
    ]
    stacked = [torch.stack(values, dim=0) for values in stacks]
    active_steps = torch.stack(all_active, dim=0).sum(dim=0).clamp(min=1)
    means = [values.sum(dim=0) / active_steps for values in stacked]

    return {
        "success_rate": success.float().mean().item(),
        "mpkpe": means[0].mean().item(),
        "r_mpkpe": means[1].mean().item(),
        "joint_vel_error": means[2].mean().item(),
        "ee_pos_error": means[3].mean().item(),
        "ee_ori_error": means[4].mean().item(),
    }


def _print_eval_metrics(metrics: dict[str, float]) -> None:
    print("\n" + "-" * 50)
    print("Evaluation Results")
    print("-" * 50)
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")
    print("-" * 50)
