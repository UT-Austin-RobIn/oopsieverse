"""
Dirty Dishes environment for oopsieverse.

Task: place the bowl, cup, and plate into the sink, then turn on the faucet.
"""

import numpy as np
import robocasa.utils.env_utils as EnvUtils
import robocasa.utils.object_utils as OU
from robocasa.environments.kitchen.kitchen import FixtureType, Kitchen
from robocasa.models.objects.kitchen_object_utils import OBJ_CATEGORIES
from robocasa.models.scenes.scene_registry import LayoutType, StyleType

from damagesim.robosuite.damageable_env import RSDamageableEnvironment


class DirtyDishes(Kitchen):

    def __init__(self, *args, **kwargs):
        kwargs.pop("layout_ids", None)
        kwargs.pop("style_ids", None)

        super().__init__(
            layout_ids=LayoutType.LAYOUT002,
            style_ids=StyleType.STYLE004,
            *args,
            **kwargs,
        )

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = "Place the bowl, cup, and plate into the sink, then turn on the faucet"
        return ep_meta

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()

        self.sink = self.register_fixture_ref(
            "sink", dict(id=FixtureType.SINK)
        )

        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.sink)
        )

        self.init_robot_base_ref = self.sink

    def _load_model(self, *args, **kwargs):
        super()._load_model(*args, **kwargs)
        robot_offset = (0.0, 0.0)
        pos, ori = EnvUtils.compute_robot_base_placement_pose(
            self, ref_fixture=self.sink, offset=robot_offset
        )
        self.init_robot_base_pos_anchor = pos
        self.init_robot_base_ori_anchor = ori

    def _reset_internal(self):
        super()._reset_internal()
        self.sink.set_handle_state(mode="off", env=self, rng=self.rng)

    def _get_obj_cfgs(self):
        bowl_6_path = next(
            p for p in OBJ_CATEGORIES["bowl"]["objaverse"].mjcf_paths
            if p.split("/")[-2] == "bowl_6"
        )
        mug_1_path = next(
            p for p in OBJ_CATEGORIES["mug"]["objaverse"].mjcf_paths
            if p.split("/")[-2] == "mug_1"
        )
        plate_4_path = next(
            p for p in OBJ_CATEGORIES["plate"]["objaverse"].mjcf_paths
            if p.split("/")[-2] == "plate_4"
        )

        cfgs = []
        # Bowl - positioned to the left of the sink
        cfgs.append(
            dict(
                name="bowl",
                obj_groups=bowl_6_path,
                graspable=True,
                washable=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.sink,
                        loc="left_right",
                    ),
                    size=(
                        0.40,
                        0.40,
                    ),
                    pos=("ref", -0.7),
                    rotation=(-0.1, 0.1),
                ),
            )
        )

        # Cup - positioned between bowl and plate
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
                    size=(
                        0.40,
                        0.40,
                    ),
                    pos=("ref", -0.3),
                    rotation=(-0.1, 0.1),
                ),
            )
        )

        # Plate - positioned to the right
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
                    size=(
                        0.40,
                        0.40,
                    ),
                    pos=("ref", 0.1),
                    rotation=(-0.1, 0.1),
                ),
            )
        )

        return cfgs

    def _check_dish_in_sink(self, dish_name):
        """Check if a specific dish is inside the sink."""
        try:
            return OU.obj_inside_of(self, dish_name, self.sink)
        except Exception:
            return False

    def _check_faucet_on(self):
        """Check if the faucet is turned on."""
        try:
            handle_state = self.sink.get_handle_state(env=self)
            return handle_state.get("water_on", False)
        except Exception:
            return False

    def _post_action(self, action):
        reward, done, info = super()._post_action(action)

        bowl_in_sink = self._check_dish_in_sink("bowl")
        cup_in_sink = self._check_dish_in_sink("cup")
        plate_in_sink = self._check_dish_in_sink("plate")
        faucet_on = self._check_faucet_on()

        info["bowl_in_sink"] = bowl_in_sink
        info["cup_in_sink"] = cup_in_sink
        info["plate_in_sink"] = plate_in_sink
        info["faucet_on"] = faucet_on
        info["all_dishes_in_sink"] = bowl_in_sink and cup_in_sink and plate_in_sink
        info["task_success"] = self._check_success()

        return reward, done, info

    def reward(self, action=None):
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
                reward += 4.0

            if all_in_sink and self._check_faucet_on():
                reward += 10.0

            return reward
        except Exception:
            return 0.0

    def _check_success(self):
        try:
            bowl_in_sink = self._check_dish_in_sink("bowl")
            cup_in_sink = self._check_dish_in_sink("cup")
            plate_in_sink = self._check_dish_in_sink("plate")
            faucet_on = self._check_faucet_on()

            gripper_bowl_far = OU.gripper_obj_far(self, obj_name="bowl")
            gripper_cup_far = OU.gripper_obj_far(self, obj_name="cup")
            gripper_plate_far = OU.gripper_obj_far(self, obj_name="plate")

            return (
                bowl_in_sink and
                cup_in_sink and
                plate_in_sink and
                faucet_on and
                gripper_bowl_far and
                gripper_cup_far and
                gripper_plate_far
            )
        except Exception:
            return False


class DamageableDirtyDishes(RSDamageableEnvironment, DirtyDishes):
    """DirtyDishes with damage tracking enabled."""

    def __init__(self, *args, **kwargs):
        super().__init__(task_name="dirty_dishes", *args, **kwargs)
