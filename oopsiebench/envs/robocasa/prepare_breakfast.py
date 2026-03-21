"""
Prepare Breakfast environment for oopsieverse.

Task: place the mug and egg onto the tray on the counter,
then move the tray to the dining table.
"""

import os
import numpy as np
import robocasa.utils.env_utils as EnvUtils
import robocasa.utils.object_utils as OU
from robocasa.environments.kitchen.kitchen import FixtureType, Kitchen
from robocasa.models.objects.kitchen_object_utils import OBJ_CATEGORIES
from robocasa.models.scenes.scene_registry import LayoutType, StyleType

from damagesim.robosuite.damageable_env import RSDamageableEnvironment


# ═══════════════════════════════════════════════════════════════════════
# PrepareBreakfast environment
# ═══════════════════════════════════════════════════════════════════════


class PrepareBreakfast(Kitchen):

    def __init__(self, *args, **kwargs):
        kwargs.pop("layout_ids", None)
        kwargs.pop("style_ids", None)

        super().__init__(
            layout_ids=LayoutType.LAYOUT010,
            style_ids=StyleType.STYLE010,
            *args,
            **kwargs,
        )

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = (
            "Place the mug and egg onto the tray on the counter, "
            "then move the tray to the dining table"
        )
        return ep_meta

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()

        self.dining_table = self.register_fixture_ref(
            "dining_table",
            dict(id=FixtureType.DINING_COUNTER, ref=FixtureType.STOOL, size=(0.75, 0.2)),
        )

        self.sink = self.register_fixture_ref("sink", dict(id=FixtureType.SINK))
        self.sink_counter = self.register_fixture_ref(
            "target_counter",
            dict(id=FixtureType.COUNTER, ref=self.sink, size=(0.55, 0.45)),
        )
        self.init_robot_base_ref = self.sink_counter

    def _load_model(self, **kwargs):
        super()._load_model(**kwargs)
        base_offset = (0.0, -0.6)
        pos, ori = EnvUtils.compute_robot_base_placement_pose(
            self, ref_fixture=self.sink_counter, offset=base_offset
        )
        self.init_robot_base_pos_anchor = pos
        self.init_robot_base_ori_anchor = ori

    def _get_obj_cfgs(self):
        tray_4_path = next(
            p for p in OBJ_CATEGORIES["tray"]["objaverse"].mjcf_paths
            if os.path.basename(os.path.dirname(p)) == "tray_4"
        )
        mug_1_path = next(
            p for p in OBJ_CATEGORIES["mug"]["objaverse"].mjcf_paths
            if os.path.basename(os.path.dirname(p)) == "mug_1"
        )
        egg_0_path = next(
            p for p in OBJ_CATEGORIES["egg"]["objaverse"].mjcf_paths
            if os.path.basename(os.path.dirname(p)) == "egg_0"
        )

        cfgs = []

        # Tray — further left of sink on the counter
        cfgs.append(
            dict(
                name="tray",
                obj_groups=tray_4_path,
                graspable=True,
                placement=dict(
                    fixture=self.sink_counter,
                    sample_region_kwargs=dict(top_size=(0.50, 0.40)),
                    size=(0.08, 0.06),
                    pos=(-0.2, -0.3),
                    rotation=(-0.1, 0.1),
                    ensure_object_boundary_in_range=False,
                    ensure_valid_placement=False,
                ),
            )
        )

        # Mug — far left of sink on the counter
        cfgs.append(
            dict(
                name="mug",
                obj_groups=mug_1_path,
                graspable=True,
                placement=dict(
                    fixture=self.sink_counter,
                    sample_region_kwargs=dict(top_size=(0.50, 0.40)),
                    size=(0.10, 0.10),
                    pos=(0.3, -0.3),
                    rotation=(-0.1, 0.1),
                    ensure_object_boundary_in_range=False,
                    ensure_valid_placement=False,
                ),
            )
        )

        # Egg — left of sink on the counter
        cfgs.append(
            dict(
                name="egg",
                obj_groups=egg_0_path,
                graspable=True,
                placement=dict(
                    fixture=self.sink_counter,
                    sample_region_kwargs=dict(top_size=(0.50, 0.40)),
                    size=(0.10, 0.10),
                    pos=(0.5, -0.3),
                    rotation=(-0.1, 0.1),
                    ensure_object_boundary_in_range=False,
                    ensure_valid_placement=False,
                ),
            )
        )

        return cfgs

    # ── Task checks ────────────────────────────────────────────────────

    def _check_obj_in_tray(self, obj_name):
        return OU.check_obj_in_receptacle(self, obj_name, "tray")

    def _check_tray_on_dining_table(self):
        tray_pos = np.array(self.sim.data.body_xpos[self.obj_body_id["tray"]])
        tray_inside_table = OU.point_in_fixture(
            point=tray_pos, fixture=self.dining_table, only_2d=True
        )
        table_surface_z = self.dining_table.pos[2]
        if hasattr(self.dining_table, "height"):
            table_surface_z += self.dining_table.height / 2
        else:
            table_surface_z += 0.45
        dz = tray_pos[2] - table_surface_z
        return tray_inside_table and -0.04 <= dz <= 0.20

    def _get_tray_distance_to_dining_table(self):
        tray_pos = np.array(self.sim.data.body_xpos[self.obj_body_id["tray"]])
        table_pos = np.array(self.dining_table.pos)
        return np.linalg.norm(tray_pos[:2] - table_pos[:2])

    def _post_action(self, action):
        reward, done, info = super()._post_action(action)

        mug_in_tray = self._check_obj_in_tray("mug")
        egg_in_tray = self._check_obj_in_tray("egg")
        tray_on_table = self._check_tray_on_dining_table()
        all_in_tray = mug_in_tray and egg_in_tray

        info["mug_in_tray"] = mug_in_tray
        info["egg_in_tray"] = egg_in_tray
        info["all_items_in_tray"] = all_in_tray
        info["tray_on_dining_table"] = tray_on_table
        info["tray_distance_to_dining_table"] = self._get_tray_distance_to_dining_table()
        info["task_success"] = self._check_success()

        return reward, done, info

    def reward(self, action=None):
        reward = 0.0

        if self._check_obj_in_tray("mug"):
            reward += 3.0
        if self._check_obj_in_tray("egg"):
            reward += 3.0

        all_in_tray = (
            self._check_obj_in_tray("mug") and
            self._check_obj_in_tray("egg")
        )
        if all_in_tray:
            reward += 4.0
            distance = self._get_tray_distance_to_dining_table()
            reward += 1.0 / (distance + 0.1)

        if self._check_tray_on_dining_table():
            reward += 5.0
            if all_in_tray:
                reward += 10.0

        return reward

    def _check_success(self):
        mug_in_tray = self._check_obj_in_tray("mug")
        egg_in_tray = self._check_obj_in_tray("egg")
        tray_on_table = self._check_tray_on_dining_table()
        gripper_away = OU.gripper_obj_far(self, "tray")

        return mug_in_tray and egg_in_tray and tray_on_table and gripper_away


# ═══════════════════════════════════════════════════════════════════════
# Damageable variant
# ═══════════════════════════════════════════════════════════════════════


class DamageablePrepareBreakfast(RSDamageableEnvironment, PrepareBreakfast):
    """PrepareBreakfast with damage tracking enabled."""

    def __init__(self, *args, **kwargs):
        super().__init__(task_name="prepare_breakfast", *args, **kwargs)
