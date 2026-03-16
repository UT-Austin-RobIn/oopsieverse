"""
Prepare Coffee environment for oopsieverse.

Task: pick the mug from the cabinet and place it under the coffee machine dispenser.
"""

import numpy as np
import robocasa.utils.env_utils as EnvUtils
import robocasa.utils.object_utils as OU
from robocasa.environments.kitchen.kitchen import FixtureType, Kitchen
from robocasa.models.objects.kitchen_object_utils import OBJ_CATEGORIES

from damagesim.robosuite.damageable_env import RSDamageableEnvironment


class PrepareCoffee(Kitchen):

    def __init__(self, cab_id=FixtureType.CABINET, *args, **kwargs):
        self.cab_id = cab_id
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()

        self.coffee_machine = self.register_fixture_ref(
            "coffee_machine", dict(id="coffee_machine")
        )
        self.cab = self.register_fixture_ref(
            "cab", dict(id=self.cab_id, ref=self.coffee_machine)
        )
        self.init_robot_base_ref = self.cab

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        obj_name = self.get_obj_lang()
        ep_meta[
            "lang"
        ] = f"Pick the {obj_name} from the cabinet and place it under the coffee machine dispenser."
        return ep_meta

    def _load_model(self, *args, **kwargs):
        super()._load_model(*args, **kwargs)
        robot_offset = (0.0, -0.1)
        pos, ori = EnvUtils.compute_robot_base_placement_pose(
            self, ref_fixture=self.cab, offset=robot_offset
        )
        self.init_robot_base_pos_anchor = pos
        self.init_robot_base_ori_anchor = ori

    def _get_obj_cfgs(self):
        mug_1_path = next(
            p for p in OBJ_CATEGORIES["mug"]["objaverse"].mjcf_paths
            if p.split("/")[-2] == "mug_1"
        )

        cfgs = []

        cfgs.append(
            dict(
                name="mug",
                obj_groups=mug_1_path,
                graspable=True,
                placement=dict(
                    fixture=self.cab,
                    size=(
                        0.30,
                        0.20,
                    ),
                    pos=(0, -1.0),
                    rotation=(-0.1, 0.1),
                ),
            )
        )
        cfgs.append(
            dict(
                name="distr_cab",
                obj_groups="all",
                placement=dict(
                    fixture=self.cab,
                    size=(1.0, 0.20),
                    pos=(0.0, 1.0),
                    offset=(0.0, 0.0),
                ),
            )
        )

        return cfgs

    def _reset_internal(self):
        super()._reset_internal()
        self.cab.set_door_state(min=0.90, max=1.0, env=self)

    def _post_action(self, action):
        reward, done, info = super()._post_action(action)

        mug_in_machine = self.coffee_machine.check_receptacle_placement_for_pouring(self, "mug")
        gripper_away = OU.gripper_obj_far(self, "mug")

        info['mug_in_coffee_machine'] = mug_in_machine
        info['gripper_away'] = gripper_away
        info['task_success'] = self._check_success()

        return reward, done, info

    def reward(self, action=None):
        reward = 0.0

        if self.coffee_machine.check_receptacle_placement_for_pouring(self, "mug"):
            reward += 10.0

        if OU.gripper_obj_far(self, "mug"):
            reward += 1.0

        return reward

    def _check_success(self):
        gripper_obj_far = OU.gripper_obj_far(self, "mug")
        contact_check = self.coffee_machine.check_receptacle_placement_for_pouring(self, "mug")

        return contact_check and gripper_obj_far


class DamageablePrepareCoffee(RSDamageableEnvironment, PrepareCoffee):
    """PrepareCoffee with damage tracking enabled."""

    def __init__(self, *args, **kwargs):
        super().__init__(task_name="prepare_coffee", *args, **kwargs)
