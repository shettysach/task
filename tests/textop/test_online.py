from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from builders import fake_env, motion_block, write_mjlab_motion_npz

from mjlab_textop.core.contract import TEXTOP_ISAACLAB_TO_MJLAB_G1_JOINT_INDEX
from mjlab_textop.core.mdp.online_commands import (
    OnlineTextOpMotionCommand,
    OnlineTextOpMotionCommandCfg,
    use_online_textop_motion_command,
)
from mjlab_textop.core.online.buffer import (
    TextOpMotionBlock,
    TextOpRollingMotionBuffer,
)
from mjlab_textop.core.online.replay import (
    QueueTextOpOnlineSource,
    make_mjlab_npz_replay_source,
)


class _LiveTextOpOnlineSource:
    def __init__(self, blocks: list[TextOpMotionBlock]) -> None:
        self.blocks = list(blocks)

    def poll(self) -> TextOpMotionBlock | None:
        if not self.blocks:
            return None
        return self.blocks.pop(0)


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


def test_rolling_buffer_finds_earliest_contiguous_start_window() -> None:
    buffer = TextOpRollingMotionBuffer()
    buffer.append_block(motion_block(index=100, frames=3))

    assert buffer.earliest_start_frame(5) is None

    buffer.append_block(motion_block(index=103, frames=5))

    assert buffer.earliest_start_frame(5) == 100


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
    assert source.fps == 50.0
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


def test_online_command_clamps_too_many_consecutive_stale_windows() -> None:
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
    future = command.future_joint_pos

    assert command._consecutive_stale_steps > command.cfg.max_stale_steps
    assert future.shape == (1, 5, 29)


def test_online_command_replay_allows_stale_windows_at_clip_end() -> None:
    source = QueueTextOpOnlineSource([motion_block(frames=5)])
    command = OnlineTextOpMotionCommand(
        OnlineTextOpMotionCommandCfg(
            source=source,
            source_mode="replay",
            future_steps=5,
            max_stale_steps=1,
        ),
        fake_env(),
    )
    command._update_command()

    for _ in range(4):
        command._update_command()
        _ = command.future_joint_pos

    assert command._consecutive_stale_steps > command.cfg.max_stale_steps
    assert command.future_joint_pos.shape == (1, 5, 29)


def test_online_command_replay_does_not_evict_preloaded_frames() -> None:
    blocks = [
        motion_block(index=block_index * 64, frames=64, offset=float(block_index * 1000))
        for block_index in range(16)
    ]
    source = QueueTextOpOnlineSource(blocks)
    command = OnlineTextOpMotionCommand(
        OnlineTextOpMotionCommandCfg(
            source=source,
            source_mode="replay",
            future_steps=5,
            max_poll_blocks=16,
            max_buffer_frames=32,
        ),
        fake_env(),
    )

    command._update_command()

    assert command._started is True
    assert command.current_frame == 0
    assert command.buffer.max_frames is None
    assert command.buffer.earliest_index == 0
    assert command.buffer.frame_count == 1024
    assert command.future_joint_pos.shape == (1, 5, 29)


def test_online_command_rejects_replay_source_without_reset() -> None:
    with pytest.raises(TypeError, match="implement reset"):
        OnlineTextOpMotionCommandCfg(
            source=_LiveTextOpOnlineSource([motion_block(frames=8)]),
            source_mode="replay",
        )


def test_online_command_rejects_replay_source_fps_mismatch() -> None:
    source = QueueTextOpOnlineSource([motion_block(frames=8)], fps=25.0)

    with pytest.raises(ValueError, match="FPS must match env control rate"):
        OnlineTextOpMotionCommand(
            OnlineTextOpMotionCommandCfg(
                source=source,
                source_mode="replay",
                future_steps=5,
            ),
            fake_env(step_dt=0.02),
        )


def test_online_command_rejects_live_source_fps_mismatch() -> None:
    source = QueueTextOpOnlineSource([motion_block(frames=8)], fps=25.0)

    with pytest.raises(ValueError, match="FPS must match env control rate"):
        OnlineTextOpMotionCommand(
            OnlineTextOpMotionCommandCfg(
                source=source,
                source_mode="live",
                future_steps=5,
            ),
            fake_env(step_dt=0.02),
        )


def test_online_command_updates_live_diagnostics_metrics() -> None:
    source = QueueTextOpOnlineSource([motion_block(frames=8)])
    source.diagnostics = SimpleNamespace(
        queue_depth=3,
        blocks_received=4,
        blocks_dropped=1,
        bad_messages=2,
    )
    command = OnlineTextOpMotionCommand(
        OnlineTextOpMotionCommandCfg(source=source, future_steps=5),
        fake_env(),
    )

    command._update_command()
    command._update_metrics()

    assert command.metrics["online_started"].item() == 1.0
    assert command.metrics["online_current_frame"].item() == 0.0
    assert command.metrics["online_latest_frame"].item() == 7.0
    assert command.metrics["online_lag_frames"].item() == 7.0
    assert command.metrics["online_queue_depth"].item() == 3.0
    assert command.metrics["online_blocks_received"].item() == 4.0
    assert command.metrics["online_blocks_dropped"].item() == 1.0
    assert command.metrics["online_bad_messages"].item() == 2.0


def test_online_command_replay_reset_rewinds_source() -> None:
    source = QueueTextOpOnlineSource([motion_block(frames=8)])
    env = fake_env()
    command = OnlineTextOpMotionCommand(
        OnlineTextOpMotionCommandCfg(
            source=source,
            source_mode="replay",
            future_steps=5,
        ),
        env,
    )
    command._update_command()

    assert command.buffer.frame_count == 8

    command._resample_command(torch.tensor([0]))

    assert command.buffer.frame_count == 8
    assert command.current_frame == 0
    assert command._started is True
    assert command.future_joint_pos.shape == (1, 5, 29)
    robot = env.scene["robot"]
    torch.testing.assert_close(
        robot.written_joint_pos,
        command.future_joint_pos[:, 0],
    )
    torch.testing.assert_close(
        robot.written_joint_vel,
        command.future_joint_vel[:, 0],
    )
    torch.testing.assert_close(
        robot.written_root_state[:, :7],
        torch.cat([command.future_anchor_pos_w[:, 0], command.future_anchor_quat_w[:, 0]], dim=-1),
    )
    torch.testing.assert_close(robot.written_root_state[:, 7:], torch.zeros(1, 6))
    torch.testing.assert_close(robot.reset_env_ids, torch.tensor([0]))


def test_online_command_live_reset_does_not_rewind_source() -> None:
    source = _LiveTextOpOnlineSource([motion_block(frames=8)])
    command = OnlineTextOpMotionCommand(
        OnlineTextOpMotionCommandCfg(
            source=source,
            source_mode="live",
            future_steps=5,
            startup_timeout_steps=1,
        ),
        fake_env(),
    )
    command._update_command()

    assert command.buffer.frame_count == 8

    command._resample_command(torch.tensor([0]))

    assert command.buffer.frame_count == 0
    command._update_command()
    with pytest.raises(RuntimeError, match="did not receive enough contiguous"):
        command._update_command()


def test_online_command_live_attaches_to_earliest_full_future_window() -> None:
    source = _LiveTextOpOnlineSource(
        [
            motion_block(index=100, frames=3),
            motion_block(index=103, frames=5),
        ]
    )
    env = fake_env(robot_anchor_pos=(10.0, 20.0, 30.0))
    command = OnlineTextOpMotionCommand(
        OnlineTextOpMotionCommandCfg(
            source=source,
            source_mode="live",
            future_steps=5,
        ),
        env,
    )

    command._update_command()

    assert command._started is True
    assert command.current_frame == 100
    assert command._last_stale_steps == 0
    robot = env.scene["robot"]
    torch.testing.assert_close(
        robot.written_joint_pos,
        command.future_joint_pos[:, 0],
    )
    torch.testing.assert_close(
        robot.written_joint_vel,
        command.future_joint_vel[:, 0],
    )
    torch.testing.assert_close(
        robot.written_root_state[:, :7],
        torch.cat(
            [command.future_anchor_pos_w[:, 0], command.future_anchor_quat_w[:, 0]],
            dim=-1,
        ),
    )
    torch.testing.assert_close(robot.written_root_state[:, 7:], torch.zeros(1, 6))
    torch.testing.assert_close(robot.reset_env_ids, torch.tensor([0]))


def test_online_command_live_reset_attaches_to_next_full_future_window() -> None:
    source = _LiveTextOpOnlineSource([motion_block(index=0, frames=8)])
    command = OnlineTextOpMotionCommand(
        OnlineTextOpMotionCommandCfg(
            source=source,
            source_mode="live",
            future_steps=5,
        ),
        fake_env(),
    )
    command._update_command()

    command._resample_command(torch.tensor([0]))
    source.blocks.extend(
        [
            motion_block(index=100, frames=3),
            motion_block(index=103, frames=5),
        ]
    )
    command._update_command()

    assert command._started is True
    assert command.current_frame == 100
    assert command._last_stale_steps == 0


def test_use_online_textop_motion_command_preserves_injected_source() -> None:
    source = QueueTextOpOnlineSource([motion_block(frames=8)])
    env_cfg = SimpleNamespace(
        commands={
            "motion": SimpleNamespace(entity_name="robot", anchor_body_name="pelvis")
        }
    )

    use_online_textop_motion_command(env_cfg, source=source)

    assert env_cfg.commands["motion"].source is source
    assert env_cfg.commands["motion"].source_mode == "live"
