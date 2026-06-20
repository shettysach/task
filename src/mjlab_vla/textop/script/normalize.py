from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from mjlab.entity import Entity
from mjlab.scene import Scene
from mjlab.sim.sim import Simulation, SimulationCfg
from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg

from mjlab_vla.textop.script.motion import (
    MJLAB_G1_JOINT_NAMES,
    load_textop_motion,
)

DEFAULT_MOTION_REL = (
    "TextOpTracker/artifacts/Data10k-open/"
    "homejrhangmr_dataset_pbhc_contact_maskACCADFemale1Walking_c3dB3-walk1_posespkl/"
    "motion.npz"
)


@dataclass(kw_only=True)
class NormalizeCommand:
    normalized_motion_file: str = "/tmp/textop_walk_mjlab.npz"
    motion_rel: str = DEFAULT_MOTION_REL
    data_dir: str = "/tmp/textop-data"
    device: str = "cuda:0"


def normalize_textop_npz(
    input_file: Path,
    output_file: Path,
    *,
    fps: float | None = None,
    device: str = "cuda:0",
    max_frames: int | None = None,
) -> Path:
    """Replay a TextOp NPZ through MJLab and save an MJLab-native tracking NPZ."""

    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[WARNING]: CUDA is not available. Falling back to CPU.")
        device = "cpu"

    motion = load_textop_motion(input_file, fps=fps)
    output_fps = motion.fps
    frame_count = (
        motion.num_frames if max_frames is None else min(max_frames, motion.num_frames)
    )
    if frame_count <= 0:
        raise ValueError("No frames selected for normalization")

    sim_cfg = SimulationCfg()
    sim_cfg.mujoco.timestep = 1.0 / output_fps

    scene = Scene(unitree_g1_flat_tracking_env_cfg().scene, device=device)
    model = scene.compile()
    sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)
    scene.initialize(sim.mj_model, sim.model, sim.data)
    scene.reset()

    robot: Entity = scene["robot"]
    robot_joint_idxs = robot.find_joints(MJLAB_G1_JOINT_NAMES, preserve_order=True)[0]

    log: dict[str, list[np.ndarray] | list[float] | np.ndarray] = {
        "fps": [output_fps],
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
            motion.root_pos_w[frame], dtype=torch.float32, device=sim.device
        ).unsqueeze(0)
        root_states[:, 3:7] = torch.as_tensor(
            motion.root_quat_w[frame], dtype=torch.float32, device=sim.device
        ).unsqueeze(0)
        root_states[:, 7:10] = torch.as_tensor(
            motion.root_lin_vel_w[frame], dtype=torch.float32, device=sim.device
        ).unsqueeze(0)
        root_states[:, 10:13] = torch.as_tensor(
            motion.root_ang_vel_w[frame], dtype=torch.float32, device=sim.device
        ).unsqueeze(0)
        robot.write_root_state_to_sim(root_states)

        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = robot.data.default_joint_vel.clone()
        joint_pos[:, robot_joint_idxs] = torch.as_tensor(
            motion.joint_pos[frame], dtype=torch.float32, device=sim.device
        ).unsqueeze(0)
        joint_vel[:, robot_joint_idxs] = torch.as_tensor(
            motion.joint_vel[frame], dtype=torch.float32, device=sim.device
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
    print(f"Saved MJLab-native motion to {output_file}")
    print(f"Frames: {frame_count}, fps: {output_fps:g}")
    return output_file


def _append_frame(
    log: dict[str, list[np.ndarray] | list[float] | np.ndarray], robot: Entity
) -> None:
    assert isinstance(log["joint_pos"], list)
    assert isinstance(log["joint_vel"], list)
    assert isinstance(log["body_pos_w"], list)
    assert isinstance(log["body_quat_w"], list)
    assert isinstance(log["body_lin_vel_w"], list)
    assert isinstance(log["body_ang_vel_w"], list)

    log["joint_pos"].append(robot.data.joint_pos[0, :].cpu().numpy().copy())
    log["joint_vel"].append(robot.data.joint_vel[0, :].cpu().numpy().copy())
    log["body_pos_w"].append(robot.data.body_link_pos_w[0, :].cpu().numpy().copy())
    log["body_quat_w"].append(robot.data.body_link_quat_w[0, :].cpu().numpy().copy())
    log["body_lin_vel_w"].append(
        robot.data.body_link_lin_vel_w[0, :].cpu().numpy().copy()
    )
    log["body_ang_vel_w"].append(
        robot.data.body_link_ang_vel_w[0, :].cpu().numpy().copy()
    )


def _validate_normalized_output(path: Path) -> None:
    data = np.load(path)
    required = (
        "fps",
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
    )
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"Normalized MJLab motion is missing keys: {missing}")

    num_frames = data["joint_pos"].shape[0]
    for key in required:
        if key == "fps":
            continue
        if data[key].shape[0] != num_frames:
            raise ValueError(
                f"Normalized output key {key} has inconsistent frame count: "
                f"{data[key].shape[0]} vs {num_frames}"
            )
