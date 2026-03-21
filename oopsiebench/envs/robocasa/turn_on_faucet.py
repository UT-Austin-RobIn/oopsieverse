"""
Turn On Faucet environment for oopsieverse.

Task: turn on the sink faucet.
"""

import numpy as np
import robocasa.utils.env_utils as EnvUtils
from robocasa.environments.kitchen.kitchen import FixtureType, Kitchen

from damagesim.robosuite.damageable_env import RSDamageableEnvironment


# ═══════════════════════════════════════════════════════════════════════
# TurnOnFaucet environment
# ═══════════════════════════════════════════════════════════════════════


class TurnOnFaucet(Kitchen):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = "turn on the sink faucet"
        return ep_meta

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.sink = self.register_fixture_ref(
            "sink", dict(id=FixtureType.SINK)
        )
        self.init_robot_base_ref = self.sink

    def _load_model(self, *args, **kwargs):
        super()._load_model(*args, **kwargs)
        robot_offset = (0.0, -0.1)
        pos, ori = EnvUtils.compute_robot_base_placement_pose(
            self, ref_fixture=self.sink, offset=robot_offset
        )
        self.init_robot_base_pos_anchor = pos
        self.init_robot_base_ori_anchor = ori

    def _reset_internal(self):
        super()._reset_internal()
        self.sink.set_handle_state(mode="off", env=self, rng=self.rng)

    def _get_obj_cfgs(self):
        return []

    # ── Task checks ────────────────────────────────────────────────────

    def reward(self, action=None):
        try:
            handle_state = self.sink.get_handle_state(env=self)
            water_on = handle_state.get("water_on", False)
            return 10.0 if water_on else 0.0
        except Exception:
            return 0.0

    def _check_success(self):
        try:
            handle_state = self.sink.get_handle_state(env=self)
            water_on = handle_state.get("water_on", False)
            return water_on
        except Exception:
            return False


# ═══════════════════════════════════════════════════════════════════════
# Damageable variant
# ═══════════════════════════════════════════════════════════════════════


class DamageableTurnOnFaucet(RSDamageableEnvironment, TurnOnFaucet):
    """TurnOnFaucet with damage tracking enabled."""

    def __init__(self, *args, **kwargs):
        super().__init__(task_name="turn_on_faucet", *args, **kwargs)
