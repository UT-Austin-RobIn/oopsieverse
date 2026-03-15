"""
Open Drawer environment for oopsieverse.

Task: open the drawer.
"""

import numpy as np
import robocasa.utils.env_utils as EnvUtils
import robocasa.utils.object_utils as OU
from robocasa.environments.kitchen.kitchen import FixtureType, Kitchen

from damagesim.robosuite.damageable_env import RSDamageableEnvironment


class OpenDrawer(Kitchen):

    def __init__(self, drawer_id=FixtureType.TOP_DRAWER, *args, **kwargs):
        self.robot_side = ""
        self.drawer_id = drawer_id
        self.drawer_side = ""
        super().__init__(*args, **kwargs)

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = f"open the {self.drawer_side} drawer"
        return ep_meta

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.drawer = self.register_fixture_ref("drawer", dict(id=self.drawer_id))
        self.init_robot_base_ref = self.drawer

    def _load_model(self, *args, **kwargs):
        super()._load_model(*args, **kwargs)
        x_ofs = (self.drawer.width / 2) + 0.3
        inits = []

        robot_base_pos_left, robot_base_ori_left = EnvUtils.compute_robot_base_placement_pose(
            self, ref_fixture=self.drawer, offset=(-x_ofs, -0.23)
        )
        test_pos_left, _ = EnvUtils.compute_robot_base_placement_pose(
            self, ref_fixture=self.drawer, offset=(-x_ofs - 0.3, -0.23)
        )

        if not self._check_fxtr_contact(test_pos_left) and not self._check_sidewall_contact(test_pos_left):
            inits.append((robot_base_pos_left, robot_base_ori_left, "right"))

        robot_base_pos_right, robot_base_ori_right = EnvUtils.compute_robot_base_placement_pose(
            self, ref_fixture=self.drawer, offset=(x_ofs, -0.23)
        )
        test_pos_right, _ = EnvUtils.compute_robot_base_placement_pose(
            self, ref_fixture=self.drawer, offset=(x_ofs + 0.3, -0.23)
        )

        if not self._check_fxtr_contact(test_pos_right) and not self._check_sidewall_contact(test_pos_right):
            inits.append((robot_base_pos_right, robot_base_ori_right, "left"))

        if len(inits) == 0:
            robot_base_pos, robot_base_ori = EnvUtils.compute_robot_base_placement_pose(
                self, ref_fixture=self.drawer, offset=(0.0, -0.23)
            )
            side = "left"
        else:
            random_index = self.rng.integers(len(inits))
            robot_base_pos, robot_base_ori, side = inits[random_index]
        self.drawer_side = side
        self.init_robot_base_pos_anchor = robot_base_pos
        self.init_robot_base_ori_anchor = robot_base_ori

    def _setup_scene(self):
        self.drawer.set_door_state(min=0.0, max=0.0, env=self)
        super()._setup_scene()

    def _check_fxtr_contact(self, pos):
        """Check if the point is in contact with any fixture (excluding walls)."""
        for fxtr in self.fixtures.values():
            if hasattr(fxtr, 'wall_side'):
                continue
            try:
                if OU.point_in_fixture(point=pos, fixture=fxtr, only_2d=True):
                    return True
            except Exception:
                continue
        return False

    def _check_sidewall_contact(self, pos):
        """Check if the point is in contact with any wall."""
        for name, fxtr in self.fixtures.items():
            if not hasattr(fxtr, 'wall_side'):
                continue
            if fxtr.wall_side == "right" and pos[0] > fxtr.pos[0]:
                return True
            if (
                fxtr.wall_side == "left"
                and "2" not in name
                and pos[0] < fxtr.pos[0]
            ):
                return True
            if fxtr.wall_side == "back" and pos[1] > fxtr.pos[1]:
                return True
        return False

    def _get_obj_cfgs(self):
        return []

    def reward(self, action=None):
        try:
            door_state = self.drawer.get_door_state(env=self)
            avg_state = np.mean(list(door_state.values()))
            return avg_state * 10.0
        except Exception:
            return 0.0

    def _check_success(self):
        door_state = self.drawer.get_door_state(env=self)
        for joint_p in door_state.values():
            if joint_p < 0.95:
                return False
        return True


class DamageableOpenDrawer(RSDamageableEnvironment, OpenDrawer):
    """OpenDrawer with damage tracking enabled."""

    def __init__(self, *args, **kwargs):
        super().__init__(task_name="open_drawer", *args, **kwargs)
