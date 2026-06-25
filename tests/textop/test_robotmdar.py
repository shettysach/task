from __future__ import annotations

import numpy as np
import pytest

from mjlab_textop.core.contract import MJLAB_G1_JOINT_NAMES
from mjlab_textop.core.motion import (
    load_mjlab_motion,
    reindex_mjlab_g1_joints_to_textop,
    reindex_textop_g1_joints_to_mjlab,
)
from mjlab_textop.core.normalize_robotmdar_record import normalize_robotmdar_record_npz
from mjlab_textop.core.online.source import TextOpMotionBlock
from mjlab_textop.core.robotmdar import (
    ROBOTMDAR_G1_DOF_INDEX,
    ROBOTMDAR_G1_DOF_LINK_NAMES,
    ROBOTMDAR_G1_DOF_NAMES,
    expand_robotmdar_dof_to_mjlab_g1,
    robotmdar_motion_dict_to_block,
    save_textop_motion_blocks_as_mjlab_npz,
    slice_motion_dict_tail,
)
from mjlab_textop.core.robotmdar_record import (
    load_robotmdar_raw_record,
    save_robotmdar_raw_record,
)


def test_robotmdar_dof_indices_are_derived_from_joint_names() -> None:
    assert ROBOTMDAR_G1_DOF_LINK_NAMES == (
        "left_hip_pitch_link",
        "left_hip_roll_link",
        "left_hip_yaw_link",
        "left_knee_link",
        "left_ankle_pitch_link",
        "left_ankle_roll_link",
        "right_hip_pitch_link",
        "right_hip_roll_link",
        "right_hip_yaw_link",
        "right_knee_link",
        "right_ankle_pitch_link",
        "right_ankle_roll_link",
        "waist_yaw_link",
        "waist_roll_link",
        "torso_link",
        "left_shoulder_pitch_link",
        "left_shoulder_roll_link",
        "left_shoulder_yaw_link",
        "left_elbow_link",
        "right_shoulder_pitch_link",
        "right_shoulder_roll_link",
        "right_shoulder_yaw_link",
        "right_elbow_link",
    )
    assert ROBOTMDAR_G1_DOF_INDEX == tuple(
        MJLAB_G1_JOINT_NAMES.index(name) for name in ROBOTMDAR_G1_DOF_NAMES
    )
    assert tuple(
        name
        for name in MJLAB_G1_JOINT_NAMES
        if name not in ROBOTMDAR_G1_DOF_NAMES
    ) == (
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    )


def test_expand_robotmdar_dof_to_mjlab_g1_places_known_dofs() -> None:
    robotmdar_dof = np.arange(2 * 23, dtype=np.float32).reshape(2, 23)

    mjlab_dof = expand_robotmdar_dof_to_mjlab_g1(robotmdar_dof)

    assert mjlab_dof.shape == (2, 29)
    np.testing.assert_allclose(mjlab_dof[:, ROBOTMDAR_G1_DOF_INDEX], robotmdar_dof)
    missing_joint_indices = [
        MJLAB_G1_JOINT_NAMES.index(name)
        for name in MJLAB_G1_JOINT_NAMES
        if name not in ROBOTMDAR_G1_DOF_NAMES
    ]
    np.testing.assert_allclose(mjlab_dof[:, missing_joint_indices], 0.0)


def test_expand_robotmdar_dof_to_mjlab_g1_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match=r"Expected \[T, 23\] RobotMDAR DoF array"):
        expand_robotmdar_dof_to_mjlab_g1(np.zeros((2, 22), dtype=np.float32))


def test_robotmdar_motion_dict_to_block_converts_to_textop_block() -> None:
    dof_pos = np.arange(3 * 23, dtype=np.float32).reshape(1, 3, 23)
    dof_vel = dof_pos + 1000.0
    root_rot_xyzw = np.tile(
        np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        (1, 3, 1),
    )
    root_trans_offset = np.arange(9, dtype=np.float32).reshape(1, 3, 3)

    block = robotmdar_motion_dict_to_block(
        {
            "dof_pos": dof_pos,
            "dof_vel": dof_vel,
            "root_rot": root_rot_xyzw,
            "root_trans_offset": root_trans_offset,
        },
        index=11,
    )

    expected_mjlab_pos = expand_robotmdar_dof_to_mjlab_g1(dof_pos[0])
    expected_mjlab_vel = expand_robotmdar_dof_to_mjlab_g1(dof_vel[0])
    assert block.index == 11
    np.testing.assert_allclose(
        block.joint_pos, reindex_mjlab_g1_joints_to_textop(expected_mjlab_pos)
    )
    np.testing.assert_allclose(
        block.joint_vel, reindex_mjlab_g1_joints_to_textop(expected_mjlab_vel)
    )
    np.testing.assert_allclose(block.anchor_pos_w, root_trans_offset[0])
    np.testing.assert_allclose(
        block.anchor_quat_w,
        np.tile(np.array([4.0, 1.0, 2.0, 3.0], dtype=np.float32), (3, 1)),
    )


def test_slice_motion_dict_tail_slices_batched_time_arrays() -> None:
    batched = np.arange(1 * 5 * 2, dtype=np.float32).reshape(1, 5, 2)
    scalar = object()

    result = slice_motion_dict_tail({"batched": batched, "scalar": scalar}, 2)

    np.testing.assert_allclose(result["batched"], batched[:, -2:])
    assert result["scalar"] is scalar


def test_save_textop_motion_blocks_as_mjlab_npz_records_replay_motion(tmp_path) -> None:
    textop_joint_pos = np.arange(4 * 29, dtype=np.float32).reshape(4, 29)
    textop_joint_vel = textop_joint_pos + 1000.0
    anchor_pos_w = np.arange(4 * 3, dtype=np.float32).reshape(4, 3)
    anchor_quat_w = np.tile(
        np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        (4, 1),
    )
    blocks = [
        TextOpMotionBlock(
            index=2,
            joint_pos=textop_joint_pos[2:4],
            joint_vel=textop_joint_vel[2:4],
            anchor_pos_w=anchor_pos_w[2:4],
            anchor_quat_w=anchor_quat_w[2:4],
        ),
        TextOpMotionBlock(
            index=0,
            joint_pos=textop_joint_pos[0:2],
            joint_vel=textop_joint_vel[0:2],
            anchor_pos_w=anchor_pos_w[0:2],
            anchor_quat_w=anchor_quat_w[0:2],
        ),
    ]
    output_file = tmp_path / "robotmdar_recorded.npz"

    save_textop_motion_blocks_as_mjlab_npz(output_file, blocks, fps=50.0)

    motion = load_mjlab_motion(output_file)
    assert motion.fps == 50.0
    assert motion.joint_pos.shape == (4, 29)
    assert motion.joint_vel.shape == (4, 29)
    assert motion.body_pos_w.shape == (4, 1, 3)
    assert motion.body_quat_w.shape == (4, 1, 4)
    np.testing.assert_allclose(
        motion.joint_pos, reindex_textop_g1_joints_to_mjlab(textop_joint_pos)
    )
    np.testing.assert_allclose(
        motion.joint_vel, reindex_textop_g1_joints_to_mjlab(textop_joint_vel)
    )
    np.testing.assert_allclose(motion.body_pos_w[:, 0, :], anchor_pos_w)
    np.testing.assert_allclose(motion.body_quat_w[:, 0, :], anchor_quat_w)


def test_robotmdar_record_saves_raw_reference_terms(tmp_path) -> None:
    textop_joint_pos = np.arange(3 * 29, dtype=np.float32).reshape(3, 29)
    textop_joint_vel = textop_joint_pos + 1000.0
    anchor_pos_w = np.arange(3 * 3, dtype=np.float32).reshape(3, 3)
    anchor_quat_w = np.tile(
        np.array([2.0, 0.0, 0.0, 0.0], dtype=np.float32),
        (3, 1),
    )
    output_file = tmp_path / "robotmdar_raw.npz"

    save_robotmdar_raw_record(
        output_file,
        [
            TextOpMotionBlock(
                index=0,
                joint_pos=textop_joint_pos,
                joint_vel=textop_joint_vel,
                anchor_pos_w=anchor_pos_w,
                anchor_quat_w=anchor_quat_w,
            )
        ],
        fps=50.0,
        prompt="walk forward",
        guidance_scale=5.0,
    )

    data = np.load(output_file)
    assert set(data.files) >= {
        "fps",
        "joint_pos",
        "joint_vel",
        "anchor_pos_w",
        "anchor_quat_w",
        "frame_index",
        "prompt",
        "guidance_scale",
        "num_blocks",
        "source",
    }
    np.testing.assert_allclose(data["joint_pos"], textop_joint_pos)
    np.testing.assert_allclose(data["joint_vel"], textop_joint_vel)
    np.testing.assert_allclose(data["anchor_pos_w"], anchor_pos_w)
    np.testing.assert_allclose(
        data["anchor_quat_w"],
        np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (3, 1)),
    )
    np.testing.assert_array_equal(data["frame_index"], np.arange(3))
    assert str(data["prompt"]) == "walk forward"
    assert str(data["source"]) == "robotmdar"
    assert int(data["num_blocks"][0]) == 1


def test_robotmdar_record_preserves_textop_joint_order(tmp_path) -> None:
    textop_joint_pos = np.arange(29, dtype=np.float32).reshape(1, 29)
    output_file = tmp_path / "robotmdar_raw.npz"

    save_robotmdar_raw_record(
        output_file,
        [
            TextOpMotionBlock(
                index=0,
                joint_pos=textop_joint_pos,
                joint_vel=textop_joint_pos + 100.0,
                anchor_pos_w=np.zeros((1, 3), dtype=np.float32),
                anchor_quat_w=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
            )
        ],
        fps=50.0,
        prompt="stand",
        guidance_scale=5.0,
    )

    record = load_robotmdar_raw_record(output_file)

    np.testing.assert_allclose(record.joint_pos, textop_joint_pos)
    np.testing.assert_allclose(record.joint_vel, textop_joint_pos + 100.0)


def test_normalize_robotmdar_record_does_not_double_reindex_joints(
    tmp_path, monkeypatch
) -> None:
    textop_joint_pos = np.arange(29, dtype=np.float32).reshape(1, 29)
    textop_joint_vel = textop_joint_pos + 100.0
    raw_file = tmp_path / "robotmdar_raw.npz"
    normalized_file = tmp_path / "robotmdar_train_ready.npz"
    save_robotmdar_raw_record(
        raw_file,
        [
            TextOpMotionBlock(
                index=0,
                joint_pos=textop_joint_pos,
                joint_vel=textop_joint_vel,
                anchor_pos_w=np.array([[0.0, 0.0, 1.0]], dtype=np.float32),
                anchor_quat_w=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
            )
        ],
        fps=50.0,
        prompt="stand",
        guidance_scale=5.0,
    )
    _patch_fake_mjlab_normalizer(monkeypatch)

    normalize_robotmdar_record_npz(raw_file, normalized_file, device="cpu")

    data = np.load(normalized_file)
    np.testing.assert_allclose(
        data["joint_pos"], reindex_textop_g1_joints_to_mjlab(textop_joint_pos)
    )
    np.testing.assert_allclose(
        data["joint_vel"], reindex_textop_g1_joints_to_mjlab(textop_joint_vel)
    )


def test_train_ready_robotmdar_npz_has_required_keys(tmp_path, monkeypatch) -> None:
    raw_file = tmp_path / "robotmdar_raw.npz"
    normalized_file = tmp_path / "robotmdar_train_ready.npz"
    save_robotmdar_raw_record(
        raw_file,
        [
            TextOpMotionBlock(
                index=0,
                joint_pos=np.zeros((2, 29), dtype=np.float32),
                joint_vel=np.zeros((2, 29), dtype=np.float32),
                anchor_pos_w=np.zeros((2, 3), dtype=np.float32),
                anchor_quat_w=np.tile(
                    np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (2, 1)
                ),
            )
        ],
        fps=50.0,
        prompt="stand",
        guidance_scale=5.0,
    )
    _patch_fake_mjlab_normalizer(monkeypatch)

    normalize_robotmdar_record_npz(raw_file, normalized_file, device="cpu")

    data = np.load(normalized_file)
    assert set(data.files) >= {
        "fps",
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
    }
    assert data["joint_pos"].shape == (2, 29)
    assert data["body_pos_w"].shape == (2, 2, 3)
    assert data["body_quat_w"].shape == (2, 2, 4)


def _patch_fake_mjlab_normalizer(monkeypatch) -> None:
    import torch

    import mjlab_textop.core.normalize_robotmdar_record as normalizer

    class FakeRobot:
        def __init__(self) -> None:
            self.data = type("Data", (), {})()
            self.data.default_root_state = torch.zeros((1, 13), dtype=torch.float32)
            self.data.default_joint_pos = torch.zeros((1, 29), dtype=torch.float32)
            self.data.default_joint_vel = torch.zeros((1, 29), dtype=torch.float32)
            self.data.joint_pos = torch.zeros((1, 29), dtype=torch.float32)
            self.data.joint_vel = torch.zeros((1, 29), dtype=torch.float32)
            self.data.body_link_pos_w = torch.zeros((1, 2, 3), dtype=torch.float32)
            self.data.body_link_quat_w = torch.tensor(
                [[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]],
                dtype=torch.float32,
            )
            self.data.body_link_lin_vel_w = torch.zeros((1, 2, 3), dtype=torch.float32)
            self.data.body_link_ang_vel_w = torch.zeros((1, 2, 3), dtype=torch.float32)

        def find_joints(self, names, preserve_order):
            assert names == MJLAB_G1_JOINT_NAMES
            assert preserve_order
            return [list(range(29))]

        def write_root_state_to_sim(self, root_states):
            self.data.body_link_pos_w[:, 0, :] = root_states[:, 0:3]
            self.data.body_link_quat_w[:, 0, :] = root_states[:, 3:7]
            self.data.body_link_lin_vel_w[:, 0, :] = root_states[:, 7:10]
            self.data.body_link_ang_vel_w[:, 0, :] = root_states[:, 10:13]

        def write_joint_state_to_sim(self, joint_pos, joint_vel):
            self.data.joint_pos = joint_pos.clone()
            self.data.joint_vel = joint_vel.clone()

    class FakeScene:
        robot = FakeRobot()

        def __init__(self, scene_cfg, device):
            self.scene_cfg = scene_cfg
            self.device = device

        def compile(self):
            return object()

        def initialize(self, mj_model, model, data):
            return None

        def reset(self):
            return None

        def update(self, dt):
            return None

        def __getitem__(self, key):
            assert key == "robot"
            return self.robot

    class FakeSimulation:
        def __init__(self, num_envs, cfg, model, device):
            self.device = device
            self.mj_model = type("MjModel", (), {"opt": type("Opt", (), {})()})()
            self.mj_model.opt.timestep = cfg.mujoco.timestep
            self.model = object()
            self.data = object()

        def forward(self):
            return None

    monkeypatch.setattr(
        normalizer,
        "unitree_g1_flat_tracking_env_cfg",
        lambda: type("Cfg", (), {"scene": object()})(),
    )
    monkeypatch.setattr(normalizer, "Scene", FakeScene)
    monkeypatch.setattr(normalizer, "Simulation", FakeSimulation)
