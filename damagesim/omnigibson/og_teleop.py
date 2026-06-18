"""
OmniGibson teleop fixes for mobile manipulators (Tiago, Fetch, …).

HolonomicBaseRobot.teleop_data_to_action only fills base + arm commands. Trunk and camera
action slots stay at zero, which with ``use_delta_commands=False`` commands absolute joint
position 0 — the torso_lift_joint drives down and the trunk collapses.
"""

from __future__ import annotations

import types

import torch as th

# Per-unit telemoma ``torso`` command (SpaceMouse Z in base mode) when trunk uses deltas.
TRUNK_TELEOP_DELTA_SCALE = 0.02

# Per-unit telemoma ``torso`` command when trunk uses absolute position goals.
TRUNK_TELEOP_ABS_SCALE = 0.02


def _controller_action_idx(robot, controller_name: str) -> th.Tensor:
    controller_idx = robot.controller_order.index(controller_name)
    start = sum(
        robot.controllers[robot.controller_order[i]].command_dim
        for i in range(controller_idx)
    )
    dim = robot.controllers[controller_name].command_dim
    return th.arange(start, start + dim)


def patch_holonomic_teleop_trunk_and_camera(robot) -> None:
    """
    Hold trunk/camera at current joint positions during teleop unless ``teleop_action.torso``
    is non-zero (SpaceMouse base mode).
    """
    if getattr(robot, "_oopsie_teleop_trunk_patch_applied", False):
        return

    # HolonomicBaseRobot on the MRO (Tiago, Fetch, …).
    try:
        from omnigibson.robots.holonomic_base_robot import HolonomicBaseRobot
    except ImportError:
        return

    if not isinstance(robot, HolonomicBaseRobot):
        return

    parent_teleop = HolonomicBaseRobot.teleop_data_to_action

    def teleop_data_to_action(self, teleop_action):
        action = parent_teleop(self, teleop_action)
        joint_pos = self.get_joint_positions()
        torso = float(getattr(teleop_action, "torso", 0.0) or 0.0)

        if "trunk" in self.controllers:
            trunk_slice = _controller_action_idx(self, "trunk")
            trunk_q = joint_pos[self.trunk_control_idx].clone()
            trunk_ctrl = self._controllers["trunk"]
            if getattr(trunk_ctrl, "use_delta_commands", False):
                if abs(torso) > 1e-6:
                    action[trunk_slice] = th.tensor(
                        [torso * TRUNK_TELEOP_DELTA_SCALE], dtype=th.float32
                    )
                else:
                    action[trunk_slice] = th.zeros(trunk_ctrl.command_dim, dtype=th.float32)
            else:
                if abs(torso) > 1e-6:
                    goal = trunk_q + th.tensor(
                        [torso * TRUNK_TELEOP_ABS_SCALE], device=trunk_q.device, dtype=trunk_q.dtype
                    )
                else:
                    goal = trunk_q
                action[trunk_slice] = goal.float()

        if "camera" in self.controllers and hasattr(self, "camera_control_idx"):
            cam_slice = _controller_action_idx(self, "camera")
            action[cam_slice] = joint_pos[self.camera_control_idx].clone().float()

        return action

    robot.teleop_data_to_action = types.MethodType(teleop_data_to_action, robot)
    robot._oopsie_teleop_trunk_patch_applied = True
