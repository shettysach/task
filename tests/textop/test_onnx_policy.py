from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mjlab_textop.core.contract import TEXTOP_ISAACLAB_TO_MJLAB_G1_JOINT_INDEX
from mjlab_textop.core.onnx_policy import TextOpOnnxPolicy


class _FakeSession:
    def __init__(self, _policy_file: str):
        self.output = np.arange(29, dtype=np.float32).reshape(1, 29)
        self.inputs = [SimpleNamespace(name="actor_obs")]
        self.received: np.ndarray | None = None

    def get_inputs(self):
        return self.inputs

    def run(self, _output_names, inputs):
        self.received = inputs["actor_obs"]
        return [np.repeat(self.output, self.received.shape[0], axis=0)]


def _install_fake_onnxruntime(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_ort = SimpleNamespace(InferenceSession=_FakeSession)
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)


def test_textop_onnx_policy_reindexes_action_to_mjlab_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_onnxruntime(monkeypatch)
    policy = TextOpOnnxPolicy(Path("latest.onnx"))
    obs = torch.zeros(2, 431)

    action = policy(obs)

    expected_one = torch.arange(29, dtype=torch.float32)[
        list(TEXTOP_ISAACLAB_TO_MJLAB_G1_JOINT_INDEX)
    ]
    assert action.shape == (2, 29)
    torch.testing.assert_close(action, expected_one.repeat(2, 1))
    assert policy.session.received is not None
    assert policy.session.received.dtype == np.float32
    assert policy.session.received.shape == (2, 431)


def test_textop_onnx_policy_accepts_actor_observation_dict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_onnxruntime(monkeypatch)
    policy = TextOpOnnxPolicy(Path("latest.onnx"))

    action = policy({"actor": torch.zeros(1, 431)})

    assert action.shape == (1, 29)


def test_textop_onnx_policy_rejects_unbatched_obs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_onnxruntime(monkeypatch)
    policy = TextOpOnnxPolicy(Path("latest.onnx"))

    with pytest.raises(RuntimeError, match=r"\[N, 431\]"):
        policy(torch.zeros(431))


def test_textop_onnx_policy_rejects_wrong_obs_dim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_onnxruntime(monkeypatch)
    policy = TextOpOnnxPolicy(Path("latest.onnx"))

    with pytest.raises(RuntimeError, match="Expected ONNX obs dim 431"):
        policy(torch.zeros(1, 430))
