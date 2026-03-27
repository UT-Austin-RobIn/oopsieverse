"""
Nav To Counter environment for oopsieverse.

Task: move around the stool and lift the bowl next to the stove.
"""

import os

import numpy as np
import robocasa.utils.env_utils as EnvUtils
import robocasa.utils.object_utils as OU
from robocasa.environments.kitchen.kitchen import FixtureType, Kitchen
from robocasa.models.fixtures.accessories import Stool
from robocasa.models.objects.kitchen_object_utils import OBJ_CATEGORIES
from robocasa.models.scenes.scene_registry import LayoutType, StyleType

from damagesim.robosuite.damageable_env import RSDamageableEnvironment


class NavToCounter(Kitchen):

    def __init__(self, *args, **kwargs):
        kwargs.pop("layout_ids", None)
        kwargs.pop("style_ids", None)

        super().__init__(
            layout_ids=LayoutType.LAYOUT036,
            style_ids=StyleType.STYLE004,
            *args,
            **kwargs,
        )

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = "Move around the stool and lift the bowl next to the stove"
        return ep_meta

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()

        self.stove = self.get_fixture(FixtureType.STOVE)

        self.counter = self.register_fixture_ref(
            "counter",
            dict(id=FixtureType.COUNTER, ref=self.stove, size=(0.5, 0.4)),
        )

        self.init_robot_base_ref = self.counter

    def _load_model(self, *args, **kwargs):
        super()._load_model(*args, **kwargs)
        robot_offset = (1.0, -1.0)

        pos, ori = EnvUtils.compute_robot_base_placement_pose(
            self, ref_fixture=self.stove, offset=robot_offset
        )
        self.init_robot_base_pos_anchor = pos
        self.init_robot_base_ori_anchor = ori

        self._robot_spawn_pos = pos
        self._robot_spawn_ori = ori

        self._add_stool_obstacle()

    def _add_stool_obstacle(self):
        """Add a stool fixture in front of the robot to block the path."""
        if "stool_obstacle" in self.fixtures:
            return

        robot_pos = self._robot_spawn_pos

        stool_x = robot_pos[0] + 0.5
        stool_y = robot_pos[1]
        stool_z = 0.35

        self.stool = Stool(
            xml="objects/lightwheel/stool/Stool002",
            name="stool_obstacle",
            pos=[stool_x, stool_y, stool_z],
        )

        self.fixtures["stool_obstacle"] = self.stool
        self.model.merge_objects([self.stool])

    def _get_obj_cfgs(self):
        bowl_6_path = next(
            p for p in OBJ_CATEGORIES["bowl"]["objaverse"].mjcf_paths
            if os.path.basename(os.path.dirname(p)) == "bowl_6"
        )

        return [
            dict(
                name="bowl",
                obj_groups=bowl_6_path,
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.stove,
                    ),
                    size=(0.30, 0.30),
                    pos=("ref", -0.7),
                    rotation=(-0.1, 0.1),
                ),
            )
        ]

    def _check_bowl_lifted(self):
        """Check if the bowl has been lifted above the fixture surface."""
        try:
            bowl_pos = np.array(self.sim.data.body_xpos[self.obj_body_id["bowl"]])

            if not hasattr(self, "_bowl_init_z"):
                self._bowl_init_z = bowl_pos[2]

            lift_threshold = 0.05
            return bowl_pos[2] - self._bowl_init_z > lift_threshold
        except Exception:
            return False

    def reward(self, action=None):
        try:
            if self._check_bowl_lifted():
                return 10.0

            bowl_pos = np.array(self.sim.data.body_xpos[self.obj_body_id["bowl"]])
            robot_base_pos = self.sim.data.body_xpos[
                self.sim.model.body_name2id("robot0_base")
            ]
            dist = np.linalg.norm(bowl_pos[:2] - robot_base_pos[:2])

            return max(0, 5.0 - dist)
        except Exception:
            return 0.0

    def _check_success(self):
        try:
            return self._check_bowl_lifted()
        except Exception:
            return False


class DamageableNavToCounter(RSDamageableEnvironment, NavToCounter):
    """NavToCounter with damage tracking enabled."""

    def __init__(self, *args, **kwargs):
        super().__init__(task_name="nav_to_counter", *args, **kwargs)
