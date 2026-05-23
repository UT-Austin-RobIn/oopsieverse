"""
Dishes to sink environment for oopsieverse.

Task: place the bowl, cup, and plate into the sink, then turn on the faucet.
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
# DishesToSink environment
# ═══════════════════════════════════════════════════════════════════════


class DishesToSink(Kitchen):

    def __init__(self, *args, **kwargs):
        self.layout_id = LayoutType.LAYOUT002
        self.style_id = StyleType.STYLE004
        self.randomize_scene = False
        super().__init__(
            layout_ids=self.layout_id,
            style_ids=self.style_id,
            *args,
            **kwargs,
        )

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = "Place the bowl, cup, and plate into the sink"
        return ep_meta

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()

        self.sink = self.register_fixture_ref(
            "sink", dict(id=FixtureType.SINK)
        )

        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.sink, size=(0.5, 0.4))
        )

        self.init_robot_base_ref = self.sink

    def _load_model(self, **kwargs):
        super()._load_model(**kwargs)
        robot_offset = (0.0, 0.0)
        pos, ori = EnvUtils.compute_robot_base_placement_pose(
            self, ref_fixture=self.sink, offset=robot_offset
        )
        self.init_robot_base_pos_anchor = pos
        self.init_robot_base_ori_anchor = ori

    def _setup_scene(self):
        super()._setup_scene()
        self.sink.set_handle_state(mode="off", env=self, rng=self.rng)

    def _get_obj_cfgs(self):
        bowl_10_path = next(
            p for p in OBJ_CATEGORIES["bowl"]["objaverse"].mjcf_paths
            if os.path.basename(os.path.dirname(p)) == "bowl_5"
        )
        mug_1_path = next(
            p for p in OBJ_CATEGORIES["mug"]["objaverse"].mjcf_paths
            if os.path.basename(os.path.dirname(p)) == "mug_1"
        )
        plate_4_path = next(
            p for p in OBJ_CATEGORIES["plate"]["objaverse"].mjcf_paths
            if os.path.basename(os.path.dirname(p)) == "plate_4"
        )

        cfgs = []

        cfgs.append(
            dict(
                name="bowl",
                obj_groups=bowl_10_path,
                graspable=True,
                washable=True,
                object_scale=[0.7, 0.7, 0.7],
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.sink,
                        loc="left_right",
                    ),
                    size=(0.30, 0.30),
                    pos=("ref", -1.0),
                    rotation=(-0.1, 0.1),
                ),
            )
        )

        cfgs.append(
            dict(
                name="cup",
                obj_groups=mug_1_path,
                graspable=True,
                washable=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.sink,
                        loc="left_right",
                    ),
                    size=(0.30, 0.30),
                    pos=("ref", -0.5),
                    rotation=(-0.1, 0.1),
                ),
            )
        )

        cfgs.append(
            dict(
                name="plate",
                obj_groups=plate_4_path,
                graspable=True,
                washable=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.sink,
                        loc="left_right",
                    ),
                    size=(0.30, 0.30),
                    pos=("ref", -0.7),
                    rotation=(-0.1, 0.1),
                ),
            )
        )

        return cfgs

    # ── Task checks ────────────────────────────────────────────────────

    def _check_dish_in_sink(self, dish_name):
        try:
            return OU.obj_inside_of(self, dish_name, self.sink)
        except Exception:
            return False

    def _post_action(self, action):
        reward, done, info = super()._post_action(action)

        bowl_in_sink = self._check_dish_in_sink("bowl")
        cup_in_sink = self._check_dish_in_sink("cup")
        plate_in_sink = self._check_dish_in_sink("plate")

        info["bowl_in_sink"] = bowl_in_sink
        info["cup_in_sink"] = cup_in_sink
        info["plate_in_sink"] = plate_in_sink
        info["all_dishes_in_sink"] = bowl_in_sink and cup_in_sink and plate_in_sink
        info["task_success"] = self._check_success()

        return reward, done, info

    def reward(self, action=None):
        """
        Reward based on task progress.
        - 2.0 per dish placed in sink (bowl, cup, plate)
        - 4.0 bonus when all three dishes are in sink
        - 10.0 bonus when all dishes in sink and faucet is on
        """
        try:
            reward = 0.0

            if self._check_dish_in_sink("bowl"):
                reward += 2.0
            if self._check_dish_in_sink("cup"):
                reward += 2.0
            if self._check_dish_in_sink("plate"):
                reward += 2.0

            all_in_sink = (
                self._check_dish_in_sink("bowl") and
                self._check_dish_in_sink("cup") and
                self._check_dish_in_sink("plate")
            )
            if all_in_sink:
                reward += 10.0

            return reward
        except Exception:
            return 0.0

    def _check_success(self):
        """
        Check if the task is successful.
        Success requires all dishes in the sink and the gripper far from each dish.
        """
        try:
            bowl_in_sink = self._check_dish_in_sink("bowl")
            cup_in_sink = self._check_dish_in_sink("cup")
            plate_in_sink = self._check_dish_in_sink("plate")

            gripper_bowl_far = OU.gripper_obj_far(self, obj_name="bowl")
            gripper_cup_far = OU.gripper_obj_far(self, obj_name="cup")
            gripper_plate_far = OU.gripper_obj_far(self, obj_name="plate")

            return (
                bowl_in_sink and
                cup_in_sink and
                plate_in_sink and
                gripper_bowl_far and
                gripper_cup_far and
                gripper_plate_far
            )
        except Exception:
            return False


# ═══════════════════════════════════════════════════════════════════════
# Damageable variant
# ═══════════════════════════════════════════════════════════════════════


class DamageableDishesToSink(RSDamageableEnvironment, DishesToSink):
    """DishesToSink with damage tracking enabled."""

    def __init__(self, *args, **kwargs):
        super().__init__(task_name="dishes_to_sink", *args, **kwargs)
