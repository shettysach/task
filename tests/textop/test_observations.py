from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from mjlab.envs.mdp.observations import projected_gravity
from mjlab.managers.scene_entity_config import SceneEntityCfg

import mjlab_textop.core.mdp.observations as textop_observations
from mjlab_textop.core.mdp.observations import (
    future_anchor_ori_b,
    future_anchor_pos_b,
    future_joint_window,
    joint_pos_rel_textop_order,
    joint_vel_rel_textop_order,
    last_action_textop_order,
)
from mjlab_textop.core.motion import MJLAB_TO_TEXTOP_G1_JOINT_INDEX


class _FakeCommandManager:
    def __init__(self, command) -> None:
        self.command = command

    def get_term(self, _name: str):
        return self.command


class _FakeFutureReferenceCommand:
    @property
    def future_joint_pos(self) -> torch.Tensor:
        return self._values["future_joint_pos"]

    @property
    def future_joint_vel(self) -> torch.Tensor:
        return self._values["future_joint_vel"]

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        return self._values["robot_anchor_pos_w"]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        return self._values["robot_anchor_quat_w"]

    @property
    def future_anchor_pos_w(self) -> torch.Tensor:
        return self._values["future_anchor_pos_w"]

    @property
    def future_anchor_quat_w(self) -> torch.Tensor:
        return self._values["future_anchor_quat_w"]


def _fake_textop_command(**kwargs) -> _FakeFutureReferenceCommand:
    if "future_joint_pos" in kwargs:
        num_envs = int(kwargs["future_joint_pos"].shape[0])
        future_steps = int(kwargs["future_joint_pos"].shape[1])
    else:
        num_envs = 1
        future_steps = 5
    kwargs.setdefault("future_joint_vel", torch.zeros(num_envs, future_steps, 29))
    kwargs.setdefault("future_anchor_pos_w", torch.zeros(num_envs, future_steps, 3))
    kwargs.setdefault("future_anchor_quat_w", torch.zeros(num_envs, future_steps, 4))
    kwargs.setdefault("robot_anchor_pos_w", torch.zeros(num_envs, 3))
    kwargs.setdefault("robot_anchor_quat_w", torch.zeros(num_envs, 4))
    command = object.__new__(_FakeFutureReferenceCommand)
    command._values = kwargs
    return command


def _fake_env_with_command(command=None, *, num_envs: int | None = None, **kwargs):
    if command is None:
        command = _fake_textop_command(**kwargs)
    if num_envs is None:
        num_envs = int(command.future_joint_pos.shape[0])
    return SimpleNamespace(
        num_envs=num_envs,
        command_manager=_FakeCommandManager(command),
    )


def test_future_joint_window_shape_and_textop_order() -> None:
    n = 2
    f = 5
    j = 29
    pos = torch.arange(n * f * j, dtype=torch.float32).reshape(n, f, j)
    vel = 1000.0 + torch.arange(n * f * j, dtype=torch.float32).reshape(n, f, j)
    env = _fake_env_with_command(
        future_joint_pos=pos,
        future_joint_vel=vel,
    )

    out = future_joint_window(env)

    assert out.shape == (n, f * j * 2)
    torch.testing.assert_close(out[:, : f * j], pos.reshape(n, -1))
    torch.testing.assert_close(out[:, f * j :], vel.reshape(n, -1))


def test_future_anchor_pos_b_uses_robot_anchor_frame() -> None:
    robot_anchor_pos_w = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
    robot_anchor_quat_w = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    future_anchor_pos_w = torch.tensor(
        [
            [
                [2.0, 2.0, 3.0],
                [1.0, 4.0, 3.0],
                [1.0, 2.0, 6.0],
                [0.0, 2.0, 3.0],
                [1.0, 1.0, 3.0],
            ]
        ],
        dtype=torch.float32,
    )
    future_anchor_quat_w = robot_anchor_quat_w[:, None, :].repeat(1, 5, 1)
    env = _fake_env_with_command(
        robot_anchor_pos_w=robot_anchor_pos_w,
        robot_anchor_quat_w=robot_anchor_quat_w,
        future_anchor_pos_w=future_anchor_pos_w,
        future_anchor_quat_w=future_anchor_quat_w,
        future_joint_pos=torch.zeros(1, 5, 29),
    )

    out = future_anchor_pos_b(env)

    assert out.shape == (1, 15)
    torch.testing.assert_close(
        out.reshape(1, 5, 3),
        future_anchor_pos_w - robot_anchor_pos_w[:, None, :],
    )


def test_future_anchor_ori_b_identity_orientation() -> None:
    robot_anchor_quat_w = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    future_anchor_quat_w = robot_anchor_quat_w[:, None, :].repeat(1, 5, 1)
    env = _fake_env_with_command(
        robot_anchor_pos_w=torch.zeros(1, 3),
        robot_anchor_quat_w=robot_anchor_quat_w,
        future_anchor_pos_w=torch.zeros(1, 5, 3),
        future_anchor_quat_w=future_anchor_quat_w,
        future_joint_pos=torch.zeros(1, 5, 29),
    )

    out = future_anchor_ori_b(env)

    assert out.shape == (1, 30)
    expected_one = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    torch.testing.assert_close(out.reshape(1, 5, 6)[0, 0], expected_one)


def test_projected_gravity_reuses_mjlab_observation() -> None:
    expected = torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32)
    env = SimpleNamespace(
        scene={
            "robot": SimpleNamespace(data=SimpleNamespace(projected_gravity_b=expected))
        }
    )

    out = projected_gravity(env, asset_cfg=SceneEntityCfg("robot"))

    torch.testing.assert_close(out, expected)


def test_observation_rejects_non_textop_command() -> None:
    env = _fake_env_with_command(command=object(), num_envs=1)

    with pytest.raises(TypeError, match="FutureReferenceCommand"):
        future_joint_window(env)


def test_joint_pos_rel_textop_order_reindexes_unbiased_mjlab_joint_pos(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = torch.arange(29, dtype=torch.float32).reshape(1, 29)
    calls = []

    def fake_joint_pos_rel(env, *, biased: bool):
        calls.append((env, biased))
        return value

    monkeypatch.setattr(textop_observations, "joint_pos_rel", fake_joint_pos_rel)
    env = object()

    out = joint_pos_rel_textop_order(env)

    assert calls == [(env, False)]
    torch.testing.assert_close(
        out,
        value[:, list(MJLAB_TO_TEXTOP_G1_JOINT_INDEX)],
    )


def test_joint_vel_rel_textop_order_reindexes_mjlab_joint_vel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = torch.arange(29, dtype=torch.float32).reshape(1, 29)
    monkeypatch.setattr(textop_observations, "joint_vel_rel", lambda _env: value)

    out = joint_vel_rel_textop_order(object())

    torch.testing.assert_close(
        out,
        value[:, list(MJLAB_TO_TEXTOP_G1_JOINT_INDEX)],
    )


def test_last_action_textop_order_reindexes_mjlab_action(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = torch.arange(29, dtype=torch.float32).reshape(1, 29)
    monkeypatch.setattr(textop_observations, "last_action", lambda _env: value)

    out = last_action_textop_order(object())

    torch.testing.assert_close(
        out,
        value[:, list(MJLAB_TO_TEXTOP_G1_JOINT_INDEX)],
    )
