"""
Counter to Microwave environment for oopsieverse.

Task: pick the coffee cup from the counter and place it in the microwave.
"""

import numpy as np
import robocasa.utils.env_utils as EnvUtils
import robocasa.utils.object_utils as OU
from robocasa.environments.kitchen.kitchen import FixtureType, Kitchen

from damagesim.robosuite.damageable_env import RSDamageableEnvironment


# ═══════════════════════════════════════════════════════════════════════
# CounterToMicrowave environment
# ═══════════════════════════════════════════════════════════════════════


class CounterToMicrowave(Kitchen):

    EXCLUDE_LAYOUTS = [8]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = "pick the coffee cup from the counter and place it in the microwave"
        return ep_meta

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.microwave = self.register_fixture_ref(
            "microwave",
            dict(id=FixtureType.MICROWAVE),
        )
        self.counter = self.register_fixture_ref(
            "counter",
            dict(id=FixtureType.COUNTER, ref=self.microwave),
        )
        self.init_robot_base_ref = self.microwave

    def _load_model(self, **kwargs):
        super()._load_model(**kwargs)
        robot_offset = (0.0, -0.1)
        pos, ori = EnvUtils.compute_robot_base_placement_pose(
            self, ref_fixture=self.microwave, offset=robot_offset
        )
        self.init_robot_base_pos_anchor = pos
        self.init_robot_base_ori_anchor = ori

    def _setup_scene(self):
        super()._setup_scene()
        self.microwave.open_door(env=self)

    def _get_obj_cfgs(self):
        cup_pos = ("ref", -1.0)
        cup_size = (
            0.30,
            0.30,
        )

        return [
            dict(
                name="coffee_cup",
                obj_groups="coffee_cup",
                graspable=True,
                microwavable=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.microwave,
                    ),
                    size=cup_size,
                    pos=cup_pos,
                    rotation=(-0.1, 0.1),
                ),
            )
        ]

    # ── Task checks ────────────────────────────────────────────────────

    def reward(self, action=None):
        try:
            obj_inside = OU.obj_inside_of(self, "coffee_cup", self.microwave)
            return 10.0 if obj_inside else 0.0
        except Exception:
            return 0.0

    def _check_success(self):
        try:
            obj_inside_microwave = OU.obj_inside_of(self, "coffee_cup", self.microwave)
            gripper_obj_far = OU.gripper_obj_far(self)
            return obj_inside_microwave and gripper_obj_far
        except Exception:
            return False



# ═══════════════════════════════════════════════════════════════════════
# Damageable variant
# ═══════════════════════════════════════════════════════════════════════


class DamageableCounterToMicrowave(RSDamageableEnvironment, CounterToMicrowave):
    """CounterToMicrowave with damage tracking enabled."""

    def __init__(self, *args, **kwargs):
        super().__init__(task_name="counter_to_microwave", *args, **kwargs)
