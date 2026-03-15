"""
Open Single Door environment for oopsieverse.

Task: open the microwave door.
"""

import numpy as np
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
