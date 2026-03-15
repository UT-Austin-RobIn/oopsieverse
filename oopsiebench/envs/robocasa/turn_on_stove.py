"""
Turn On Stove environment for oopsieverse.

Task: turn on a stove burner knob.
"""

import numpy as np
import robocasa.utils.env_utils as EnvUtils
from robocasa.environments.kitchen.kitchen import FixtureType, Kitchen

from damagesim.robosuite.damageable_env import RSDamageableEnvironment


class TurnOnStove(Kitchen):

    def __init__(self, knob_id="random", *args, **kwargs):
        self.knob_id = knob_id
        self.knob = None
        self.cookware_burner = None
        super().__init__(*args, **kwargs)

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        knob_name = self.knob.replace('_', ' ') if self.knob else "stove"
        ep_meta["lang"] = f"turn on the {knob_name} burner of the stove"
        ep_meta["task_refs"] = dict(
            knob=self.knob,
            cookware_burner=self.cookware_burner,
        )
        return ep_meta

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.stove = self.register_fixture_ref(
            "stove", dict(id=FixtureType.STOVE)
        )

        if "task_refs" in self._ep_meta:
            self.knob = self._ep_meta["task_refs"]["knob"]
            self.cookware_burner = self._ep_meta["task_refs"]["cookware_burner"]
        else:
            valid_knobs = [
                k for (k, v) in self.stove.knob_joints.items() if v is not None
            ]
            if self.knob_id == "random":
                self.knob = self.rng.choice(list(valid_knobs))
            else:
                assert self.knob_id in valid_knobs
                self.knob = self.knob_id
            self.cookware_burner = (
                self.knob
                if self.rng.uniform() <= 0.50
                else self.rng.choice(valid_knobs)
            )

        self.init_robot_base_ref = self.stove

    def _load_model(self, *args, **kwargs):
        super()._load_model(*args, **kwargs)
        pos, ori = EnvUtils.compute_robot_base_placement_pose(
            self, ref_fixture=self.stove, offset=(0.0, -0.1)
        )
        self.init_robot_base_pos_anchor = pos
        self.init_robot_base_ori_anchor = ori

    def _reset_internal(self):
        super()._reset_internal()
        self.stove.set_knob_state(
            mode="off", knob=self.knob, env=self, rng=self.rng
        )

    def _get_obj_cfgs(self):
        return []

    def reward(self, action=None):
        try:
            knobs_state = self.stove.get_knobs_state(env=self)
            knob_value = knobs_state[self.knob]
            knob_on = 0.35 <= np.abs(knob_value) <= 2 * np.pi - 0.35
            return 10.0 if knob_on else 0.0
        except Exception:
            return 0.0

    def _check_success(self):
        try:
            knobs_state = self.stove.get_knobs_state(env=self)
            knob_value = knobs_state[self.knob]
            knob_on = 0.35 <= np.abs(knob_value) <= 2 * np.pi - 0.35
            return knob_on
        except Exception:
            return False


class DamageableTurnOnStove(RSDamageableEnvironment, TurnOnStove):
    """TurnOnStove with damage tracking enabled."""

    def __init__(self, *args, **kwargs):
        super().__init__(task_name="turn_on_stove", *args, **kwargs)
