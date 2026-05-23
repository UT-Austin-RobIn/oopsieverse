"""
Close Drawer environment for oopsieverse.

Task: close the drawer (episode starts open).
"""

import numpy as np
import robocasa.utils.env_utils as EnvUtils
import robocasa.utils.object_utils as OU
from robocasa.environments.kitchen.kitchen import FixtureType, Kitchen
from robocasa.models.scenes.scene_registry import StyleType

from damagesim.robosuite.damageable_env import RSDamageableEnvironment

# Lateral clearance (m) from drawer half-width anchor: larger than OpenDrawer's
# 0.3 so the robot base starts farther from the cabinet / open drawer volume.
_CLOSE_DRAWER_SIDE_CLEARANCE = 0.3

# Min distance (m) from right-arm eef site to drawer fixture body for success
# (matches robocasa ``object_utils.gripper_fxtr_far`` default threshold).
_GRIPPER_FXTR_FAR_TH = 0.75


# ═══════════════════════════════════════════════════════════════════════
# CloseDrawer environment
# ═══════════════════════════════════════════════════════════════════════


class CloseDrawer(Kitchen):

    def __init__(self, drawer_id=FixtureType.TOP_DRAWER, *args, **kwargs):
        self.drawer_id = drawer_id
        self.drawer_side = ""
        self.randomize_scene = True
        super().__init__(*args, **kwargs)

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = f"close the {self.drawer_side} drawer"
        return ep_meta

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.drawer = self.register_fixture_ref("drawer", dict(id=self.drawer_id))
        self.init_robot_base_ref = self.drawer

    def _load_model(self, *args, **kwargs):
        super()._load_model(*args, **kwargs)
        side_clear = _CLOSE_DRAWER_SIDE_CLEARANCE
        x_ofs = (self.drawer.width / 2) + side_clear
        y_ofs = -0.50
        inits = []

        robot_base_pos_left, robot_base_ori_left = EnvUtils.compute_robot_base_placement_pose(
            self, ref_fixture=self.drawer, offset=(-x_ofs, y_ofs)
        )
        test_pos_left, _ = EnvUtils.compute_robot_base_placement_pose(
            self, ref_fixture=self.drawer, offset=(-x_ofs - side_clear, y_ofs)
        )

        if not self._check_fxtr_contact(test_pos_left) and not self._check_sidewall_contact(test_pos_left):
            inits.append((robot_base_pos_left, robot_base_ori_left, "right"))

        robot_base_pos_right, robot_base_ori_right = EnvUtils.compute_robot_base_placement_pose(
            self, ref_fixture=self.drawer, offset=(x_ofs, y_ofs)
        )
        test_pos_right, _ = EnvUtils.compute_robot_base_placement_pose(
            self, ref_fixture=self.drawer, offset=(x_ofs + side_clear, y_ofs)
        )

        if not self._check_fxtr_contact(test_pos_right) and not self._check_sidewall_contact(test_pos_right):
            inits.append((robot_base_pos_right, robot_base_ori_right, "left"))

        if len(inits) == 0:
            robot_base_pos, robot_base_ori = EnvUtils.compute_robot_base_placement_pose(
                self,
                ref_fixture=self.drawer,
                offset=(0.0, y_ofs),
            )
            side = "left"
        else:
            robot_base_pos, robot_base_ori, side = inits[0]
        self.drawer_side = side
        self.init_robot_base_pos_anchor = robot_base_pos
        self.init_robot_base_ori_anchor = robot_base_ori

    def _setup_scene(self):
        self.drawer.set_door_state(min=0.95, max=1.0, env=self)
        super()._setup_scene()

    def _check_fxtr_contact(self, pos):
        for fxtr in self.fixtures.values():
            if hasattr(fxtr, "wall_side"):
                continue
            try:
                if OU.point_in_fixture(point=pos, fixture=fxtr, only_2d=True):
                    return True
            except Exception:
                continue
        return False

    def _check_sidewall_contact(self, pos):
        for name, fxtr in self.fixtures.items():
            if not hasattr(fxtr, "wall_side"):
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

    # ── Task checks ────────────────────────────────────────────────────

    def reward(self, action=None):
        try:
            door_state = self.drawer.get_door_state(env=self)
            avg_state = np.mean(list(door_state.values()))
            return 10.0 * (1.0 - avg_state)
        except Exception:
            return 0.0

    def _check_success(self):
        try:
            door_state = self.drawer.get_door_state(env=self)
            for joint_p in door_state.values():
                if joint_p > 0.05:
                    return False
            # Fixture: use gripper_fxtr_far (gripper_obj_far is for env.objects names).
            if not OU.gripper_fxtr_far(
                self, self.drawer.root_body, th=_GRIPPER_FXTR_FAR_TH
            ):
                return False
            return True
        except Exception:
            return False


# ═══════════════════════════════════════════════════════════════════════
# Damageable variant
# ═══════════════════════════════════════════════════════════════════════


class DamageableCloseDrawer(RSDamageableEnvironment, CloseDrawer):
    """CloseDrawer with damage tracking enabled."""

    def __init__(self, *args, **kwargs):
        super().__init__(task_name="close_drawer", *args, **kwargs)
