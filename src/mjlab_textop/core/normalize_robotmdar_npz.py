from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from mjlab.entity import Entity
from mjlab.scene import Scene
from mjlab.sim.sim import Simulation, SimulationCfg
from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg

from mjlab_textop.core.contract import MJLAB_G1_JOINT_NAMES
from mjlab_textop.core.motion import reindex_textop_g1_joints_to_mjlab
from mjlab_textop.core.normalize_tracker_npz import (
    _append_frame,
    _validate_normalized_output,
)
from mjlab_textop.core.robotmdar_record import load_robotmdar_raw_record


def normalize_robotmdar_npz(
    input_file: Path,
    output_file: Path,
    *,
    device: str = "cuda:0",
    max_frames: int | None = None,
) -> Path:
    """Replay a raw RobotMDAR record through MJLab FK and save train-ready NPZ."""

    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[WARNING]: CUDA is not available. Falling back to CPU.")
        device = "cpu"

    record = load_robotmdar_raw_record(input_file)
    frame_count = (
        record.num_frames if max_frames is None else min(max_frames, record.num_frames)
    )
    if frame_count <= 0:
        raise ValueError("No frames selected for normalization")

    joint_pos_mjlab = reindex_textop_g1_joints_to_mjlab(record.joint_pos)
    joint_vel_mjlab = reindex_textop_g1_joints_to_mjlab(record.joint_vel)
    root_lin_vel_w = _finite_difference_linear_velocity(
        record.anchor_pos_w, record.fps
    )
    root_ang_vel_w = np.zeros_like(root_lin_vel_w, dtype=np.float32)

    sim_cfg = SimulationCfg()
    sim_cfg.mujoco.timestep = 1.0 / record.fps

    scene = Scene(unitree_g1_flat_tracking_env_cfg().scene, device=device)
    model = scene.compile()
    sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)
    scene.initialize(sim.mj_model, sim.model, sim.data)
    scene.reset()

    robot: Entity = scene["robot"]
    robot_joint_idxs = robot.find_joints(MJLAB_G1_JOINT_NAMES, preserve_order=True)[0]

    log: dict[str, list[np.ndarray] | list[float] | np.ndarray] = {
        "fps": [record.fps],
        "joint_pos": [],
        "joint_vel": [],
        "body_pos_w": [],
        "body_quat_w": [],
        "body_lin_vel_w": [],
        "body_ang_vel_w": [],
    }

    for frame in range(frame_count):
        root_states = robot.data.default_root_state.clone()
        root_states[:, 0:3] = torch.as_tensor(
            record.anchor_pos_w[frame], dtype=torch.float32, device=sim.device
        ).unsqueeze(0)
        root_states[:, 3:7] = torch.as_tensor(
            record.anchor_quat_w[frame], dtype=torch.float32, device=sim.device
        ).unsqueeze(0)
        root_states[:, 7:10] = torch.as_tensor(
            root_lin_vel_w[frame], dtype=torch.float32, device=sim.device
        ).unsqueeze(0)
        root_states[:, 10:13] = torch.as_tensor(
            root_ang_vel_w[frame], dtype=torch.float32, device=sim.device
        ).unsqueeze(0)
        robot.write_root_state_to_sim(root_states)

        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = robot.data.default_joint_vel.clone()
        joint_pos[:, robot_joint_idxs] = torch.as_tensor(
            joint_pos_mjlab[frame], dtype=torch.float32, device=sim.device
        ).unsqueeze(0)
        joint_vel[:, robot_joint_idxs] = torch.as_tensor(
            joint_vel_mjlab[frame], dtype=torch.float32, device=sim.device
        ).unsqueeze(0)
        robot.write_joint_state_to_sim(joint_pos, joint_vel)

        sim.forward()
        scene.update(sim.mj_model.opt.timestep)

        _append_frame(log, robot)

    for key in (
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
    ):
        log[key] = np.stack(log[key], axis=0)  # ty:ignore[no-matching-overload]

    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_file, **log)  # ty:ignore[invalid-argument-type]
    _validate_normalized_output(output_file)
    print(f"Saved MJLab-native RobotMDAR motion to {output_file}")
    print(f"Frames: {frame_count}, fps: {record.fps:g}")
    return output_file


normalize_robotmdar_record_npz = normalize_robotmdar_npz


def _finite_difference_linear_velocity(pos: np.ndarray, fps: float) -> np.ndarray:
    vel = np.zeros_like(pos, dtype=np.float32)
    if pos.shape[0] > 1:
        vel[:-1] = (pos[1:] - pos[:-1]) * fps
        vel[-1] = vel[-2]
    return vel
