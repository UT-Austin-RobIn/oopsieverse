"""
Prepare Coffee environment for oopsieverse.

Task: pick the mug from the cabinet, place it under the coffee machine dispenser,
turn the machine on, and release the mug.
"""

import os
import numpy as np
import robocasa.utils.env_utils as EnvUtils
import robocasa.utils.object_utils as OU
from robocasa.environments.kitchen.kitchen import FixtureType, Kitchen
from robocasa.models.objects.kitchen_object_utils import OBJ_CATEGORIES

from damagesim.robosuite.damageable_env import RSDamageableEnvironment


# ═══════════════════════════════════════════════════════════════════════
# PrepareCoffee environment
# ═══════════════════════════════════════════════════════════════════════


class PrepareCoffee(Kitchen):

    def __init__(self, cab_id=FixtureType.CABINET, *args, **kwargs):
        self.cab_id = cab_id
        self.randomize_scene = True
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()

        self.coffee_machine = self.get_fixture(FixtureType.COFFEE_MACHINE)
        self.cab = self.get_fixture(self.cab_id, ref=self.coffee_machine)
        self.init_robot_base_ref = self.cab

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = (
            "Pick the mug from the cabinet, place it under the coffee machine dispenser, "
            "press start to turn the machine on, then release the mug."
        )
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
            if os.path.basename(os.path.dirname(p)) == "mug_1"
        )

        cfgs = []

        cfgs.append(
            dict(
                name="mug",
                obj_groups=mug_1_path,
                graspable=True,
                placement=dict(
                    fixture=self.cab,
                    size=(0.30, 0.20),
                    pos=(0, -1.0),
                    rotation=(-0.1, 0.1),
                ),
            )
        )
        cfgs.append(
            dict(
                name="distr_object",
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

    def _setup_scene(self):
        # Keep the mug cabinet fully open: set before parent setup so the
        # cabinet is open during scene assembly, then again in case the base
        # class resets fixture doors.
        self.cab.open_door(env=self, min=1.0, max=1.0)
        super()._setup_scene()
        self.cab.open_door(env=self, min=1.0, max=1.0)
        # Episode starts with the coffee machine off (must press start during the task).
        if hasattr(self.coffee_machine, "_turned_on"):
            self.coffee_machine._turned_on = False
        try:
            self.coffee_machine.update_state(self)
        except Exception:
            pass

    def _ensure_mug_cabinet_stays_open(self):
        """Re-open the cabinet if contacts or dynamics let the door drift closed."""
        try:
            if not self.cab.is_open(env=self, th=0.95):
                self.cab.open_door(env=self, min=1.0, max=1.0)
        except Exception:
            pass

    # ── Task checks ────────────────────────────────────────────────────

    def _post_action(self, action):
        reward, done, info = super()._post_action(action)

        mug_in_machine = self.coffee_machine.check_receptacle_placement_for_pouring(self, "mug")
        gripper_away = OU.gripper_obj_far(self, "mug")
        coffee_on = self._coffee_machine_turned_on()

        info['mug_in_coffee_machine'] = mug_in_machine
        info['gripper_away'] = gripper_away
        info['coffee_machine_on'] = coffee_on
        info['task_success'] = self._check_success()

        return reward, done, info

    def reward(self, action=None):
        reward = 0.0

        if self.coffee_machine.check_receptacle_placement_for_pouring(self, "mug"):
            reward += 10.0

        if OU.gripper_obj_far(self, "mug"):
            reward += 1.0

        if self._coffee_machine_turned_on():
            reward += 5.0

        return reward

    def _coffee_machine_turned_on(self):
        try:
            return bool(self.coffee_machine.get_state().get("turned_on", False))
        except Exception:
            return False

    def _check_success(self):
        gripper_obj_far = OU.gripper_obj_far(self, "mug")
        contact_check = self.coffee_machine.check_receptacle_placement_for_pouring(self, "mug")
        coffee_on = self._coffee_machine_turned_on()

        return contact_check and gripper_obj_far and coffee_on



# ═══════════════════════════════════════════════════════════════════════
# Damageable variant
# ═══════════════════════════════════════════════════════════════════════


class DamageablePrepareCoffee(RSDamageableEnvironment, PrepareCoffee):
    """PrepareCoffee with damage tracking enabled."""

    def __init__(self, *args, **kwargs):
        super().__init__(task_name="prepare_coffee", *args, **kwargs)