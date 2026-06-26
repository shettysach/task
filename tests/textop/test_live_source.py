from __future__ import annotations

import numpy as np
import pytest
from builders import motion_block

from mjlab_textop.core.online.live import (
    SocketTextOpOnlineSource,
    SocketTextOpSourceCfg,
    parse_textop_block_message,
    textop_block_to_ndjson_message,
)


def test_textop_block_ndjson_round_trip() -> None:
    block = motion_block(index=100, frames=8)

    parsed, fps = parse_textop_block_message(
        textop_block_to_ndjson_message(block, fps=50.0)
    )

    assert fps == 50.0
    assert parsed.index == 100
    np.testing.assert_allclose(parsed.joint_pos, block.joint_pos)
    np.testing.assert_allclose(parsed.joint_vel, block.joint_vel)
    np.testing.assert_allclose(parsed.anchor_pos_w, block.anchor_pos_w)
    np.testing.assert_allclose(
        parsed.anchor_quat_w,
        np.tile([1.0, 0.0, 0.0, 0.0], (8, 1)),
    )


def test_textop_block_parser_rejects_missing_field() -> None:
    with pytest.raises(ValueError, match="missing required fields"):
        parse_textop_block_message({"index": 0})


def test_textop_block_parser_rejects_bad_shape() -> None:
    block = motion_block(index=0, frames=8)
    message = {
        "index": 0,
        "joint_pos": np.zeros((8, 28), dtype=np.float32).tolist(),
        "joint_vel": block.joint_vel.tolist(),
        "anchor_pos_w": block.anchor_pos_w.tolist(),
        "anchor_quat_w": block.anchor_quat_w.tolist(),
    }

    with pytest.raises(ValueError, match="29 joints"):
        parse_textop_block_message(message)


def test_socket_source_queues_and_drops_oldest_blocks() -> None:
    source = SocketTextOpOnlineSource(SocketTextOpSourceCfg(max_queue_blocks=1))
    source.append_message(textop_block_to_ndjson_message(motion_block(index=0)))
    source.append_message(textop_block_to_ndjson_message(motion_block(index=8)))

    assert source.diagnostics.blocks_received == 2
    assert source.diagnostics.blocks_dropped == 1
    assert source.diagnostics.queue_depth == 1

    block = source.poll()

    assert block is not None
    assert block.index == 8
    assert source.diagnostics.blocks_polled == 1
    assert source.diagnostics.queue_depth == 0
    assert [block.index for block in source.recorded_blocks()] == [0, 8]


def test_socket_source_records_bad_messages() -> None:
    source = SocketTextOpOnlineSource()

    source._handle_line(b"{not json}\n")

    assert source.diagnostics.bad_messages == 1
    assert source.diagnostics.last_error is not None
