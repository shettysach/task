from __future__ import annotations

import json
from copy import deepcopy

import pytest
import torch

from mjlab_textop.core.feedback.observation import (
    UdpObservationPublisher,
    UdpObservationPublisherCfg,
    make_online_textop_observation,
)


class _FakeSocket:
    def __init__(self, *args, **kwargs) -> None:
        self.sent = []
        self.closed = False

    def sendto(self, data, address) -> None:
        self.sent.append((data, address))

    def close(self) -> None:
        self.closed = True


def test_make_online_textop_observation_payload() -> None:
    payload = make_online_textop_observation(
        frame=10,
        started=True,
        current_frame=10,
        latest_frame=18,
        lag_frames=8,
        buffer_frames=32,
        stale_steps=0,
        consecutive_stale_steps=0,
        robot_anchor_pos_w=torch.tensor([1, 2, 3]),
        robot_anchor_quat_w=torch.tensor([1, 0, 0, 0]),
    )

    assert payload == {
        "schema": "mjlab_textop.online_observation.v1",
        "frame": 10,
        "started": True,
        "current_frame": 10,
        "latest_frame": 18,
        "lag_frames": 8,
        "buffer_frames": 32,
        "stale_steps": 0,
        "consecutive_stale_steps": 0,
        "fallen": False,
        "fall_reason": None,
        "robot_anchor_pos_w": [1.0, 2.0, 3.0],
        "robot_anchor_quat_w": [1.0, 0.0, 0.0, 0.0],
    }


def test_udp_observation_publisher_sends_json(monkeypatch) -> None:
    fake_socket = _FakeSocket()
    monkeypatch.setattr(
        "mjlab_textop.core.feedback.observation.socket.socket",
        lambda *args, **kwargs: fake_socket,
    )

    publisher = UdpObservationPublisher(
        UdpObservationPublisherCfg(host="127.0.0.1", port=9999)
    )
    publisher.publish({"frame": 1})
    publisher.close()

    assert len(fake_socket.sent) == 1
    data, address = fake_socket.sent[0]
    assert json.loads(data.decode("utf-8")) == {"frame": 1}
    assert address == ("127.0.0.1", 9999)
    assert fake_socket.closed is True


def test_udp_observation_publisher_rejects_invalid_port() -> None:
    with pytest.raises(ValueError, match="port must be positive"):
        UdpObservationPublisher(UdpObservationPublisherCfg(port=0))


def test_udp_observation_publisher_cfg_is_deepcopyable() -> None:
    cfg = UdpObservationPublisherCfg(host="127.0.0.1", port=8766)

    copied = deepcopy(cfg)

    assert copied == cfg
