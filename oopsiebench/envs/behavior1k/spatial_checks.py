"""
Geometric predicates shared across OmniGibson Behavior-1K tasks.

RoboSuite / Robocasa often use ``robocasa.utils.object_utils.gripper_obj_far``.
OmniGibson manipulation robots expose end-effector world positions via
``ManipulationRobot.get_eef_position``, which uses each model's configured
``eef_link_names`` (Franka ``eef_link``, other grippers' palm links, multi-arm setups).
"""

from __future__ import annotations

from typing import Any, Sequence, Union

import torch as th

__all__ = [
    "DEFAULT_GRIPPER_OBJECT_FAR_THRESHOLD_M",
    "eef_world_position_or_raise",
    "gripper_far_from_object",
]

# Match robocasa.utils.object_utils.gripper_obj_far default (~25 cm)
DEFAULT_GRIPPER_OBJECT_FAR_THRESHOLD_M = 0.25


def _tensor_vec3(vec: Union[th.Tensor, Sequence[float], Any]) -> th.Tensor:
    """First three coords as float32 1-D tensor."""
    if isinstance(vec, th.Tensor):
        v = vec.detach().float().reshape(-1)[:3]
    else:
        v = th.as_tensor(vec, dtype=th.float32).reshape(-1)[:3]
    return v


def eef_world_position_or_raise(robot: Any, arm: str = "default") -> th.Tensor:
    """
    EE world xyz. Prefers ``get_eef_position``; falls back to ``eef_links`` pose.
    """
    if hasattr(robot, "get_eef_position"):
        return _tensor_vec3(robot.get_eef_position(arm=arm))

    arm_key = arm
    if arm == "default" and getattr(robot, "default_arm", None) is not None:
        arm_key = robot.default_arm
    eef_links = getattr(robot, "eef_links", None)
    if eef_links is not None and arm_key in eef_links:
        link = eef_links[arm_key]
        pos, _ = link.get_position_orientation()
        return _tensor_vec3(pos)

    raise TypeError(
        f"Robot {type(robot).__name__!r} has no get_eef_position/eef_links; "
        "cannot compute gripper–object distance."
    )


def gripper_far_from_object(
    robot,
    obj,
    *,
    threshold: float = DEFAULT_GRIPPER_OBJECT_FAR_THRESHOLD_M,
    xy_only: bool = False,
    arm: str = "default",
) -> bool:
    """True iff distance(EE world pos, ``obj`` root pos) exceeds *threshold* (meters)."""
    eef_pos = eef_world_position_or_raise(robot, arm=arm)
    obj_pos, _ = obj.get_position_orientation()
    obj_xyz = _tensor_vec3(obj_pos)
    delta = eef_pos - obj_xyz.to(device=eef_pos.device, dtype=eef_pos.dtype)
    if xy_only:
        dist = float(th.norm(delta[:2]).item())
    else:
        dist = float(th.norm(delta).item())
    return dist > float(threshold)
