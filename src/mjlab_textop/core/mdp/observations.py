from __future__ import annotations

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.observations import joint_pos_rel, joint_vel_rel, last_action
from mjlab.utils.lab_api.math import matrix_from_quat, subtract_frame_transforms

from mjlab_textop.core.mdp.future_reference import FutureReferenceCommand
from mjlab_textop.core.motion import MJLAB_TO_TEXTOP_G1_JOINT_INDEX

_MJLAB_TO_TEXTOP_INDEX_CACHE: dict[torch.device, torch.Tensor] = {}


def _get_future_reference_command(
    env: ManagerBasedRlEnv,
    command_name: str,
) -> FutureReferenceCommand:
    command = env.command_manager.get_term(command_name)

    if not isinstance(command, FutureReferenceCommand):
        raise TypeError(
            f"Expected command {command_name!r} to satisfy "
            f"FutureReferenceCommand, "
            f"got {type(command).__name__}"
        )

    return command


def future_joint_window(
    env: ManagerBasedRlEnv,
    command_name: str = "motion",
) -> torch.Tensor:
    command = _get_future_reference_command(env, command_name)

    return torch.cat(
        [
            command.future_joint_pos.reshape(env.num_envs, -1),
            command.future_joint_vel.reshape(env.num_envs, -1),
        ],
        dim=-1,
    )


def future_joint_window_textop_order(
    env: ManagerBasedRlEnv,
    command_name: str = "motion",
) -> torch.Tensor:
    command = _get_future_reference_command(env, command_name)
    index = _mjlab_to_textop_index(command.future_joint_pos.device)

    joint_pos = command.future_joint_pos.index_select(-1, index)
    joint_vel = command.future_joint_vel.index_select(-1, index)

    return torch.cat(
        [
            joint_pos.reshape(env.num_envs, -1),
            joint_vel.reshape(env.num_envs, -1),
        ],
        dim=-1,
    )


def future_anchor_pos_b(
    env: ManagerBasedRlEnv,
    command_name: str = "motion",
) -> torch.Tensor:
    command = _get_future_reference_command(env, command_name)
    pos_b, _ = _future_anchor_pose_b(command)

    return pos_b.reshape(env.num_envs, -1)


def future_anchor_ori_b(
    env: ManagerBasedRlEnv,
    command_name: str = "motion",
) -> torch.Tensor:
    command = _get_future_reference_command(env, command_name)
    _, ori_b = _future_anchor_pose_b(command)

    mat = matrix_from_quat(ori_b)
    return mat[..., :2].reshape(env.num_envs, -1)


def _future_anchor_pose_b(
    command: FutureReferenceCommand,
) -> tuple[torch.Tensor, torch.Tensor]:
    robot_anchor_pos_w = command.robot_anchor_pos_w[:, None, :].expand_as(
        command.future_anchor_pos_w
    )
    robot_anchor_quat_w = command.robot_anchor_quat_w[:, None, :].expand_as(
        command.future_anchor_quat_w
    )
    return subtract_frame_transforms(
        robot_anchor_pos_w,
        robot_anchor_quat_w,
        command.future_anchor_pos_w,
        command.future_anchor_quat_w,
    )


def _mjlab_to_textop_index(device: torch.device | str) -> torch.Tensor:
    torch_device = torch.device(device)
    index = _MJLAB_TO_TEXTOP_INDEX_CACHE.get(torch_device)
    if index is None:
        index = torch.tensor(
            MJLAB_TO_TEXTOP_G1_JOINT_INDEX,
            device=torch_device,
            dtype=torch.long,
        )
        _MJLAB_TO_TEXTOP_INDEX_CACHE[torch_device] = index
    return index


def joint_pos_rel_textop_order(env: ManagerBasedRlEnv) -> torch.Tensor:
    value = joint_pos_rel(env, biased=False)
    return value.index_select(-1, _mjlab_to_textop_index(value.device))


def joint_vel_rel_textop_order(env: ManagerBasedRlEnv) -> torch.Tensor:
    value = joint_vel_rel(env)
    return value.index_select(-1, _mjlab_to_textop_index(value.device))


def last_action_textop_order(env: ManagerBasedRlEnv) -> torch.Tensor:
    value = last_action(env)
    return value.index_select(-1, _mjlab_to_textop_index(value.device))
