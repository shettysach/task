from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from builders import fake_env, motion_block, write_mjlab_motion_npz

from mjlab_vla.textop.contract import TEXTOP_ISAACLAB_TO_MJLAB_G1_JOINT_INDEX
from mjlab_vla.textop.mdp.online_commands import (
    OnlineTextOpMotionCommand,
    OnlineTextOpMotionCommandCfg,
    use_online_textop_motion_command,
)
from mjlab_vla.textop.online.buffer import (
    TextOpMotionBlock,
    TextOpRollingMotionBuffer,
)
from mjlab_vla.textop.online.replay import (
    QueueTextOpOnlineSource,
    make_mjlab_npz_replay_source,
)


def test_rolling_buffer_reindexes_and_slices_first_five_frames() -> None:
    block = motion_block(frames=8)
    buffer = TextOpRollingMotionBuffer()

    buffer.append_block(block)
    joint_pos, joint_vel, anchor_pos_w, anchor_quat_w, stale_steps = buffer.get_future(
        0, 5
    )

    assert stale_steps == 0
    assert joint_pos.shape == (5, 29)
    expected = block.joint_pos[:5, list(TEXTOP_ISAACLAB_TO_MJLAB_G1_JOINT_INDEX)]
    np.testing.assert_allclose(joint_pos.cpu().numpy(), expected)
    np.testing.assert_allclose(
        joint_vel.cpu().numpy(),
        block.joint_vel[:5, list(TEXTOP_ISAACLAB_TO_MJLAB_G1_JOINT_INDEX)],
    )
    np.testing.assert_allclose(anchor_pos_w.cpu().numpy(), block.anchor_pos_w[:5])
    np.testing.assert_allclose(
        anchor_quat_w.cpu().numpy(),
        np.tile([1.0, 0.0, 0.0, 0.0], (5, 1)),
    )


def test_rolling_buffer_overwrites_overlapping_block_frames() -> None:
    buffer = TextOpRollingMotionBuffer()
    buffer.append_block(motion_block(index=0, frames=8, offset=0.0))
    buffer.append_block(motion_block(index=4, frames=3, offset=5000.0))

    joint_pos, _, _, _, stale_steps = buffer.get_future(3, 4)

    assert stale_steps == 0
    expected_source = np.concatenate(
        [
            motion_block(index=0, frames=8, offset=0.0).joint_pos[3:4],
            motion_block(index=4, frames=3, offset=5000.0).joint_pos[:3],
        ],
        axis=0,
    )
    expected = expected_source[:, list(TEXTOP_ISAACLAB_TO_MJLAB_G1_JOINT_INDEX)]
    np.testing.assert_allclose(joint_pos.cpu().numpy(), expected)


def test_rolling_buffer_requires_contiguous_start_window() -> None:
    buffer = TextOpRollingMotionBuffer()
    buffer.append_block(motion_block(index=1, frames=4))

    assert buffer.can_start(0, 5) is False
    assert buffer.can_start(1, 4) is True


def test_rolling_buffer_repeats_latest_available_frame_on_underrun() -> None:
    buffer = TextOpRollingMotionBuffer()
    block = motion_block(index=0, frames=5)
    buffer.append_block(block)

    joint_pos, _, _, _, stale_steps = buffer.get_future(3, 5)

    assert stale_steps == 3
    expected_source = np.stack(
        [
            block.joint_pos[3],
            block.joint_pos[4],
            block.joint_pos[4],
            block.joint_pos[4],
            block.joint_pos[4],
        ],
        axis=0,
    )
    expected = expected_source[:, list(TEXTOP_ISAACLAB_TO_MJLAB_G1_JOINT_INDEX)]
    np.testing.assert_allclose(joint_pos.cpu().numpy(), expected)


def test_rolling_buffer_rejects_request_before_earliest_frame() -> None:
    buffer = TextOpRollingMotionBuffer()
    buffer.append_block(motion_block(index=10, frames=5))

    with pytest.raises(RuntimeError, match="at or before 0"):
        buffer.get_future(0, 5)


def test_rolling_buffer_evicts_old_frames() -> None:
    buffer = TextOpRollingMotionBuffer(max_frames=5)
    buffer.append_block(motion_block(index=0, frames=8))

    assert buffer.frame_count == 5
    assert buffer.can_start(0, 5) is False
    assert buffer.can_start(3, 5) is True


def test_rolling_buffer_rejects_wrong_joint_count() -> None:
    block = motion_block(frames=1)
    bad = TextOpMotionBlock(
        index=0,
        joint_pos=np.zeros((1, 28), dtype=np.float32),
        joint_vel=block.joint_vel,
        anchor_pos_w=block.anchor_pos_w,
        anchor_quat_w=block.anchor_quat_w,
    )

    with pytest.raises(ValueError, match="29 joints"):
        TextOpRollingMotionBuffer().append_block(bad)


def test_mjlab_npz_replay_source_chunks_and_round_trips_joint_order(tmp_path) -> None:
    path = tmp_path / "motion.npz"
    joint_pos, _, _, _ = write_mjlab_motion_npz(path, frames=10)

    source = make_mjlab_npz_replay_source(path, block_size=8)
    buffer = TextOpRollingMotionBuffer()
    while (block := source.poll()) is not None:
        buffer.append_block(block)

    round_trip_joint_pos, _, _, _, stale_steps = buffer.get_future(0, 10)

    assert stale_steps == 0
    np.testing.assert_allclose(round_trip_joint_pos.cpu().numpy(), joint_pos)


def test_online_command_polls_source_and_exposes_five_step_window() -> None:
    source = QueueTextOpOnlineSource([motion_block(frames=8)])
    command = OnlineTextOpMotionCommand(
        OnlineTextOpMotionCommandCfg(source=source, future_steps=5),
        fake_env(),
    )

    command._update_command()

    assert command.future_joint_pos.shape == (1, 5, 29)
    assert command.future_joint_vel.shape == (1, 5, 29)
    assert command.future_anchor_pos_w.shape == (1, 5, 3)
    assert command.future_anchor_quat_w.shape == (1, 5, 4)
    assert command.joint_pos.shape == (1, 29)
    assert command.joint_vel.shape == (1, 29)
    assert command.anchor_pos_w.shape == (1, 3)
    assert command.anchor_quat_w.shape == (1, 4)
    assert command.current_frame == 0

    command._update_command()
    assert command.current_frame == 1


def test_online_command_exposes_startup_window_before_source_poll() -> None:
    source = QueueTextOpOnlineSource([motion_block(frames=8)])
    command = OnlineTextOpMotionCommand(
        OnlineTextOpMotionCommandCfg(source=source, future_steps=5),
        fake_env(robot_anchor_pos=(10.0, 20.0, 30.0)),
    )

    assert command.future_joint_pos.shape == (1, 5, 29)
    assert command.future_joint_vel.shape == (1, 5, 29)
    assert command.future_anchor_pos_w.shape == (1, 5, 3)
    assert command.future_anchor_quat_w.shape == (1, 5, 4)
    assert command.buffer.frame_count == 0
    torch.testing.assert_close(command.future_joint_pos, torch.zeros(1, 5, 29))
    torch.testing.assert_close(
        command.future_anchor_pos_w,
        torch.tensor([[[10.0, 20.0, 30.0]]]).expand(1, 5, 3),
    )


def test_online_command_aligns_anchor_position_to_robot_start() -> None:
    source = QueueTextOpOnlineSource([motion_block(frames=8, offset=100.0)])
    command = OnlineTextOpMotionCommand(
        OnlineTextOpMotionCommandCfg(source=source, future_steps=5),
        fake_env(robot_anchor_pos=(10.0, 20.0, 30.0)),
    )

    command._update_command()

    future_anchor_pos = command.future_anchor_pos_w[0]
    torch.testing.assert_close(future_anchor_pos[0], torch.tensor([10.0, 20.0, 30.0]))
    torch.testing.assert_close(future_anchor_pos[1], torch.tensor([11.0, 20.0, 30.0]))


def test_online_command_rejects_vectorized_envs() -> None:
    source = QueueTextOpOnlineSource([motion_block(frames=8)])

    with pytest.raises(ValueError, match="one environment"):
        OnlineTextOpMotionCommand(
            OnlineTextOpMotionCommandCfg(source=source),
            fake_env(num_envs=2),
        )


def test_online_command_rejects_too_many_consecutive_stale_windows() -> None:
    source = QueueTextOpOnlineSource([motion_block(frames=5)])
    command = OnlineTextOpMotionCommand(
        OnlineTextOpMotionCommandCfg(
            source=source,
            future_steps=5,
            max_stale_steps=1,
        ),
        fake_env(),
    )
    command._update_command()

    command._update_command()
    command._update_command()
    command._update_command()
    _ = command.future_joint_pos

    command._update_command()
    with pytest.raises(RuntimeError, match="max consecutive stale"):
        _ = command.future_joint_pos


def test_online_command_reset_clears_buffer_by_default() -> None:
    source = QueueTextOpOnlineSource([motion_block(frames=8)])
    command = OnlineTextOpMotionCommand(
        OnlineTextOpMotionCommandCfg(source=source, future_steps=5),
        fake_env(),
    )
    command._update_command()

    assert command.buffer.frame_count == 8

    command._resample_command(torch.tensor([0]))

    assert command.buffer.frame_count == 0
    assert command.current_frame == 0


def test_use_online_textop_motion_command_preserves_injected_source() -> None:
    source = QueueTextOpOnlineSource([motion_block(frames=8)])
    env_cfg = SimpleNamespace(
        commands={
            "motion": SimpleNamespace(entity_name="robot", anchor_body_name="pelvis")
        }
    )

    use_online_textop_motion_command(env_cfg, source=source)

    assert env_cfg.commands["motion"].source is source
