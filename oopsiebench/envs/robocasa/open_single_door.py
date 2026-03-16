"""
Open Single Door environment for oopsieverse.

Task: open the microwave door.
"""

import numpy as np
import robocasa.utils.env_utils as EnvUtils
from robocasa.environments.kitchen.kitchen import FixtureType, Kitchen

from damagesim.robosuite.damageable_env import RSDamageableEnvironment


class OpenSingleDoor(Kitchen):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = "open the microwave door"
        return ep_meta

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.microwave = self.register_fixture_ref(
            "microwave", dict(id=FixtureType.MICROWAVE)
        )
        self.init_robot_base_ref = self.microwave

    def _setup_scene(self):
        self.microwave.close_door(env=self)
        super()._setup_scene()

    def _load_model(self, *args, **kwargs):
        super()._load_model(*args, **kwargs)
        robot_offset = (0.0, -0.1)
        pos, ori = EnvUtils.compute_robot_base_placement_pose(
            self, ref_fixture=self.microwave, offset=robot_offset
        )
        self.init_robot_base_pos_anchor = pos
        self.init_robot_base_ori_anchor = ori

    def _get_obj_cfgs(self):
        return []

    def reward(self, action=None):
        try:
            door_state = self.microwave.get_door_state(env=self)
            avg_state = np.mean(list(door_state.values()))
            return avg_state * 10.0
        except Exception:
            return 0.0

    def _check_success(self):
        door_state = self.microwave.get_door_state(env=self)
        for joint_p in door_state.values():
            if joint_p < 0.90:
                return False
        return True


class DamageableOpenSingleDoor(RSDamageableEnvironment, OpenSingleDoor):
    """OpenSingleDoor with damage tracking enabled."""

    def __init__(self, *args, **kwargs):
        super().__init__(task_name="open_single_door", *args, **kwargs)
