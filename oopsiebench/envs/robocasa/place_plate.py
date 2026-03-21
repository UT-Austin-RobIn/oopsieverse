"""
Place Plate environment for oopsieverse.

Task: pick up the plate and place it into the sink.
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
# PlacePlate environment
# ═══════════════════════════════════════════════════════════════════════


class PlacePlate(Kitchen):

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
        ep_meta["lang"] = "Pick up the plate and place it into the sink"
        return ep_meta

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()

        self.sink = self.register_fixture_ref("sink", dict(id=FixtureType.SINK))

        self.counter = self.register_fixture_ref(
            "counter",
            dict(id=FixtureType.COUNTER, ref=self.sink, size=(0.5, 0.4))
        )

        self.init_robot_base_ref = self.sink

    def _load_model(self, **kwargs):
        super()._load_model(**kwargs)
        robot_offset = [0.0, 0.0]
        pos, ori = EnvUtils.compute_robot_base_placement_pose(
            self, ref_fixture=self.sink, offset=robot_offset
        )
        self.init_robot_base_pos_anchor = pos
        self.init_robot_base_ori_anchor = ori

    def _get_obj_cfgs(self):
        plate_4_path = next(
            p for p in OBJ_CATEGORIES["plate"]["objaverse"].mjcf_paths
            if os.path.basename(os.path.dirname(p)) == "plate_4"
        )
        
        return [
            dict(
                name="plate",
                obj_groups=plate_4_path,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.sink,
                        loc="left_right",
                    ),
                    size=(0.40, 0.40),
                    pos=("ref", -0.7),
                    rotation=(-0.1, 0.1),
                ),
            )
        ]

    # ── Task checks ────────────────────────────────────────────────────

    def _get_sink_bounds(self):
        try:
            sink_pos = np.array(self.sink.pos)

            if hasattr(self.sink, 'size'):
                sink_size = self.sink.size
            else:
                sink_size = [0.4, 0.3, 0.2]

            return {
                'center': sink_pos,
                'half_size': np.array(sink_size) / 2,
            }
        except Exception:
            return None

    def _check_plate_in_sink(self):
        sink_bounds = self._get_sink_bounds()
        if sink_bounds is None:
            return False

        try:
            plate_pos = np.array(self.sim.data.body_xpos[self.obj_body_id["plate"]])
            sink_center = sink_bounds['center']
            half_size = sink_bounds['half_size']

            dx = abs(plate_pos[0] - sink_center[0])
            dy = abs(plate_pos[1] - sink_center[1])

            within_x = dx <= half_size[0]
            within_y = dy <= half_size[1]

            sink_rim_z = sink_center[2] + 0.1
            plate_below_rim = plate_pos[2] <= sink_rim_z

            return within_x and within_y and plate_below_rim
        except Exception:
            return False

    def _check_plate_settled(self):
        try:
            plate_body_id = self.obj_body_id["plate"]
            lin_vel = self.sim.data.cvel[plate_body_id][3:6]
            speed = np.linalg.norm(lin_vel)
            return speed < 0.05
        except Exception:
            return False

    def _check_gripper_away_from_sink(self):
        try:
            sink_pos = np.array(self.sink.pos)
            gripper_pos = self.sim.data.site_xpos[self.robots[0].eef_site_id["right"]]
            distance = np.linalg.norm(gripper_pos[:2] - sink_pos[:2])
            return distance > 0.10
        except Exception:
            return False

    def _post_action(self, action):
        reward, done, info = super()._post_action(action)

        info['plate_in_sink'] = self._check_plate_in_sink()
        info['plate_settled'] = self._check_plate_settled()
        info['task_success'] = self._check_success()

        return reward, done, info

    def reward(self, action=None):
        try:
            plate_pos = np.array(self.sim.data.body_xpos[self.obj_body_id["plate"]])
            sink_pos = np.array(self.sink.pos)

            distance = np.linalg.norm(plate_pos[:2] - sink_pos[:2])

            reward = 1.0 / (distance + 0.1)

            if self._check_plate_in_sink():
                reward += 5.0
                if self._check_plate_settled():
                    reward += 5.0

            return reward
        except Exception:
            return 0.0

    def _check_success(self):
        plate_in_sink = self._check_plate_in_sink()
        plate_settled = self._check_plate_settled()
        gripper_obj_far = OU.gripper_obj_far(self, "plate")
        gripper_away_from_sink = self._check_gripper_away_from_sink()

        return plate_in_sink and plate_settled and gripper_obj_far and gripper_away_from_sink


# ═══════════════════════════════════════════════════════════════════════
# Damageable variant
# ═══════════════════════════════════════════════════════════════════════


class DamageablePlacePlate(RSDamageableEnvironment, PlacePlate):
    """PlacePlate with damage tracking enabled."""

    def __init__(self, *args, **kwargs):
        super().__init__(task_name="place_plate", *args, **kwargs)
