"""
SpaceMouse teleop with base/torso speed scaling.

telemoma's SpaceMouseInterface only honors ``arm_speed_scaledown``; base commands
use raw pyspacemouse deflection (~[-1, 1]). This subclass applies scaling and maps
roll/pitch to Tiago head joints (via teleop_action.extra["head"]) in base mode.
"""

from __future__ import annotations

import numpy as np
from telemoma.human_interface.spacemouse import SpaceMouseInterface
from telemoma.human_interface.teleop_core import TeleopAction, TeleopObservation


class ScaledSpaceMouseInterface(SpaceMouseInterface):
    def __init__(self, *args, **kwargs) -> None:
        self.base_speed_scaledown = float(kwargs.get("base_speed_scaledown", 1.0))
        self.torso_speed_scaledown = float(
            kwargs.get("torso_speed_scaledown", kwargs.get("base_speed_scaledown", 1.0))
        )
        self.head_speed_scaledown = float(kwargs.get("head_speed_scaledown", 1.0))
        super().__init__(*args, **kwargs)

    def get_action(self, obs: TeleopObservation) -> TeleopAction:
        self.actions.base = np.zeros(3)
        self.actions.torso = 0.0
        self.actions.extra = {}
        if self.raw_data:
            controlling_robot_part = self.controllable_robot_parts[self.cur_control_idx]
            if controlling_robot_part == "base":
                bscale = self.base_speed_scaledown
                tscale = self.torso_speed_scaledown
                hscale = self.head_speed_scaledown
                self.actions.base[0] = self.raw_data.y * bscale
                self.actions.base[1] = -self.raw_data.x * bscale
                self.actions.base[2] = -self.raw_data.yaw * bscale
                self.actions.torso = self.raw_data.z * tscale
                self.actions.extra["head"] = np.array(
                    [self.raw_data.roll * hscale, -self.raw_data.pitch * hscale],
                    dtype=np.float64,
                )
            else:
                self.actions[controlling_robot_part][:3] = np.array(
                    [self.raw_data.y, -self.raw_data.x, self.raw_data.z]
                ) * self.arm_speed_scaledown
                self.actions[controlling_robot_part][3:6] = np.array(
                    [self.raw_data.roll, self.raw_data.pitch, -self.raw_data.yaw]
                ) * self.arm_speed_scaledown
        return self.actions


def register_scaled_spacemouse() -> None:
    """Use ScaledSpaceMouseInterface for telemoma 'spacemouse' controller."""
    import telemoma.human_interface.teleop_policy as teleop_policy

    teleop_policy.INTERFACE_MAP["spacemouse"] = ScaledSpaceMouseInterface
