"""
Turn On Microwave environment for oopsieverse.

Task: press the start button on the microwave.
"""

import numpy as np
import robocasa.utils.env_utils as EnvUtils
from robocasa.environments.kitchen.kitchen import FixtureType, Kitchen

from damagesim.robosuite.damageable_env import RSDamageableEnvironment


# ═══════════════════════════════════════════════════════════════════════
# TurnOnMicrowave environment
# ═══════════════════════════════════════════════════════════════════════


class TurnOnMicrowave(Kitchen):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = "press the start button on the microwave"
        return ep_meta

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.microwave = self.register_fixture_ref(
            "microwave", dict(id=FixtureType.MICROWAVE)
        )
        self.init_robot_base_ref = self.microwave

    def _load_model(self, *args, **kwargs):
        super()._load_model(*args, **kwargs)
        robot_offset = (0.0, -0.1)
        pos, ori = EnvUtils.compute_robot_base_placement_pose(
            self, ref_fixture=self.microwave, offset=robot_offset
        )
        self.init_robot_base_pos_anchor = pos
        self.init_robot_base_ori_anchor = ori

    def _setup_scene(self):
        self.microwave.close_door(env=self)
        if hasattr(self.microwave, '_turned_on'):
            self.microwave._turned_on = False
        super()._setup_scene()

    def _get_obj_cfgs(self):
        return []

    # ── Task checks ────────────────────────────────────────────────────

    def reward(self, action=None):
        try:
            state = self.microwave.get_state()
            turned_on = state.get("turned_on", False)
            return 10.0 if turned_on else 0.0
        except Exception:
            return 0.0

    def _check_success(self):
        try:
            turned_on = self.microwave.get_state().get("turned_on", False)
            gripper_button_far = self.microwave.gripper_button_far(
                self, button="start_button"
            )
            return turned_on and gripper_button_far
        except Exception:
            return False


# ═══════════════════════════════════════════════════════════════════════
# Damageable variant
# ═══════════════════════════════════════════════════════════════════════


class DamageableTurnOnMicrowave(RSDamageableEnvironment, TurnOnMicrowave):
    """TurnOnMicrowave with damage tracking enabled."""

    def __init__(self, *args, **kwargs):
        super().__init__(task_name="turn_on_microwave", *args, **kwargs)
