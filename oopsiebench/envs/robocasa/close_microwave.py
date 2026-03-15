"""
Close Microwave environment for oopsieverse.

Task: close the microwave door.
"""

import numpy as np
from robocasa.environments.kitchen.kitchen import FixtureType, Kitchen

from damagesim.robosuite.damageable_env import RSDamageableEnvironment


class CloseMicrowave(Kitchen):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = "close the microwave door"
        return ep_meta

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.microwave = self.register_fixture_ref(
            "microwave", dict(id=FixtureType.MICROWAVE)
        )
        self.init_robot_base_ref = self.microwave

    def _setup_scene(self):
        self.microwave.open_door(env=self)
        super()._setup_scene()

    def _get_obj_cfgs(self):
        return []

    def reward(self, action=None):
        try:
            door_state = self.microwave.get_door_state(env=self)
            avg_state = np.mean(list(door_state.values()))
            return (1.0 - avg_state) * 10.0
        except Exception:
            return 0.0

    def _check_success(self):
        door_state = self.microwave.get_door_state(env=self)
        for joint_p in door_state.values():
            if joint_p > 0.05:
                return False
        return True


class DamageableCloseMicrowave(RSDamageableEnvironment, CloseMicrowave):
    """CloseMicrowave with damage tracking enabled."""

    def __init__(self, *args, **kwargs):
        super().__init__(task_name="close_microwave", *args, **kwargs)
