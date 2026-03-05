import os

import numpy as np
import robocasa
import robocasa.utils.env_utils as EnvUtils
from robocasa.environments.kitchen.kitchen import FixtureType, Kitchen
from robocasa.models.scenes.scene_registry import LayoutType, StyleType

from damagesim.robosuite.damageable_env import RSDamageableEnvironment
from damagesim.robosuite.damageable_mixin import DamageableMJCFObject
from damagesim.robosuite.params import get_params_for_object


class PickEgg(Kitchen):

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
        ep_meta["lang"] = "Pick up the egg gently without crushing it"
        return ep_meta

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.sink = self.get_fixture(FixtureType.SINK)
        self.counter = self.get_fixture(FixtureType.COUNTER, ref=self.sink)
        self.init_robot_base_ref = self.counter

    def _load_model(self):
        super()._load_model()
        pos, ori = EnvUtils.compute_robot_base_placement_pose(
            self, ref_fixture=self.counter, offset=[1.0, 0.0]
        )
        self.init_robot_base_pos_anchor = pos
        self.init_robot_base_ori_anchor = ori

    def _get_obj_cfgs(self):
        return [
            dict(
                name="egg",
                obj_groups="egg",
                info=dict(
                    mjcf_path=os.path.join(
                        robocasa.models.assets_root,
                        "objects/objaverse/egg/egg_0/model.xml",
                    )
                ),
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(ref=self.sink, loc="right"),
                    size=(0.1, 0.1),
                    offset=(0, -0.50),
                    rotation=0.0,
                ),
            )
        ]

    def _create_obj(self, cfg):
        if cfg.get("name") == "egg" and isinstance(self, RSDamageableEnvironment):
            mjcf_path = cfg["info"]["mjcf_path"]
            egg = DamageableMJCFObject(
                name="egg",
                mjcf_path=mjcf_path,
                params=get_params_for_object("egg"),
            )
            info = {
                "groups_containing_sampled_obj": ["egg", "food"],
                "obj_path": mjcf_path,
            }
            return egg, info
        else:
            return super()._create_obj(cfg)

    def reward(self, action=None):
        return 0.0

    def _check_success(self):
        egg_pos = np.array(self.sim.data.body_xpos[self.obj_body_id["egg"]])
        egg_z = egg_pos[2]

        counter_surface_z = self.counter.pos[2] + self.counter.height / 2

        height_above_table = egg_z - counter_surface_z

        return height_above_table >= 0.1


class DamageablePickEgg(RSDamageableEnvironment, PickEgg):
    """PickEgg with damage tracking enabled."""

    def __init__(self, *args, **kwargs):
        super().__init__(task_name="pick_egg", *args, **kwargs)
