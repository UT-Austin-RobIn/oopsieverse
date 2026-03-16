"""
Shelve Item environment for oopsieverse.

Task: pick the cereal box and place it on the table mat.
"""

import numpy as np
import robocasa.utils.env_utils as EnvUtils
import robocasa.utils.object_utils as OU
from robocasa.environments.kitchen.kitchen import FixtureType, Kitchen
from robocasa.models.objects.kitchen_object_utils import OBJ_CATEGORIES
from robocasa.models.scenes.scene_registry import LayoutType, StyleType

from robosuite.models.objects import BoxObject
from robosuite.utils.mjcf_utils import CustomMaterial

from damagesim.robosuite.damageable_env import RSDamageableEnvironment
from damagesim.robosuite.params.damage_params import get_params_for_object


TABLE_MAT_SIZE = [0.09, 0.18, 0.002]
TABLE_MAT_COLOR = [0.06, 0.10, 0.30, 1.0]


class ShelveItem(Kitchen):

    def __init__(self, *args, **kwargs):
        kwargs.pop("layout_ids", None)
        kwargs.pop("style_ids", None)

        self.table_mat = None
        self._table_mat_pos = None

        super().__init__(
            layout_ids=LayoutType.LAYOUT010,
            style_ids=StyleType.STYLE010,
            *args,
            **kwargs,
        )

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = "Pick the cereal box and place it on the table mat"
        return ep_meta

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()

        self.dining_table = self.register_fixture_ref(
            "dining_table",
            dict(id=FixtureType.DINING_COUNTER, ref=FixtureType.STOOL, size=(0.75, 0.2)),
        )

        self.init_robot_base_ref = self.dining_table

    def _load_model(self, **kwargs):
        super()._load_model(**kwargs)
        self._add_table_mat()
        robot_offset = [1.5, 0.6]
        pos, ori = EnvUtils.compute_robot_base_placement_pose(
            self, ref_fixture=self.dining_table, offset=robot_offset
        )
        ori = np.array(ori)
        ori[2] += np.pi / 2
        self.init_robot_base_pos_anchor = pos
        self.init_robot_base_ori_anchor = ori

    def _add_table_mat(self):
        """Add a table mat to the far right edge of the dining table."""
        existing_bodies = [child.get("name") for child in self.model.worldbody]
        if "table_mat" in existing_bodies or "table_mat_main" in existing_bodies:
            return

        table_mat_params = get_params_for_object("table_mat")
        is_damageable = isinstance(self, RSDamageableEnvironment)

        tex_attrib = {"type": "cube"}
        mat_attrib = {"texrepeat": "1 1", "specular": "0.1", "shininess": "0.3"}

        mat = CustomMaterial(
            texture=TABLE_MAT_COLOR,
            tex_name="table_mat_tex",
            mat_name="table_mat_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
            shared=True,
        )

        dining_pos = self.dining_table.pos
        dining_surface_z = dining_pos[2]
        if hasattr(self.dining_table, 'height'):
            dining_surface_z += self.dining_table.height / 2
        else:
            dining_surface_z += 0.45

        dining_x = self.dining_table.pos[0]
        dining_y = self.dining_table.pos[1]

        mat_x = dining_x - 0.83
        mat_y = dining_y + 0.28
        mat_z = dining_surface_z + TABLE_MAT_SIZE[2] + 0.001

        self._table_mat_pos = np.array([mat_x, mat_y, mat_z])

        self.table_mat = BoxObject(
            name="table_mat",
            size=TABLE_MAT_SIZE,
            rgba=TABLE_MAT_COLOR,
            material=mat,
            obj_type="all" if is_damageable else "visual",
            joints=None,
            density=100,
            friction=table_mat_params.get("friction", (0.9, 0.4, 0.2)),
        )

        self.model.merge_assets(self.table_mat)
        table_mat_obj = self.table_mat.get_obj()

        if is_damageable:
            solimp = table_mat_params.get("solimp", (0.90, 0.95, 0.003))
            solref = table_mat_params.get("solref", (0.01, 1))
            for geom in table_mat_obj.findall(".//geom"):
                geom.set("solimp", " ".join(map(str, solimp)))
                geom.set("solref", " ".join(map(str, solref)))
            self._table_mat_params = table_mat_params

        sites_to_remove = table_mat_obj.findall(".//site")
        for site in sites_to_remove:
            for parent in table_mat_obj.iter():
                if site in parent:
                    parent.remove(site)
                    break

        table_mat_obj.set("pos", f"{mat_x} {mat_y} {mat_z}")

        self.model.worldbody.append(table_mat_obj)

    def _get_obj_cfgs(self):
        cereal_2_path = next(
            p for p in OBJ_CATEGORIES["cereal"]["objaverse"].mjcf_paths
            if p.split("/")[-2] == "cereal_2"
        )
        wine_5_path = next(
            p for p in OBJ_CATEGORIES["wine"]["objaverse"].mjcf_paths
            if p.split("/")[-2] == "wine_5"
        )
        wine_2_path = next(
            p for p in OBJ_CATEGORIES["wine"]["objaverse"].mjcf_paths
            if p.split("/")[-2] == "wine_2"
        )
        wine_4_path = next(
            p for p in OBJ_CATEGORIES["wine"]["objaverse"].mjcf_paths
            if p.split("/")[-2] == "wine_4"
        )
        mug_1_path = next(
            p for p in OBJ_CATEGORIES["mug"]["objaverse"].mjcf_paths
            if p.split("/")[-2] == "mug_1"
        )

        cfgs = []
        # Cereal box
        cfgs.append(
            dict(
                name="cereal",
                obj_groups=cereal_2_path,
                graspable=True,
                placement=dict(
                    fixture=self.dining_table,
                    sample_region_kwargs=dict(
                        top_size=(0.50, 0.40)
                    ),
                    size=(0.0, 0.0),
                    pos=(0.94, -0.68),
                    rotation=(1.57 - 0.1, 1.57 + 0.1),
                    ensure_object_boundary_in_range=False,
                    ensure_valid_placement=False,
                ),
            )
        )

        # Wine bottle 1
        cfgs.append(
            dict(
                name="wine_1",
                obj_groups=wine_5_path,
                graspable=True,
                placement=dict(
                    fixture=self.dining_table,
                    sample_region_kwargs=dict(
                        top_size=(0.50, 0.40)
                    ),
                    size=(0.0, 0.0),
                    pos=(0.78, -0.95),
                    rotation=(-0.1, 0.1),
                    ensure_object_boundary_in_range=False,
                    ensure_valid_placement=False,
                ),
            )
        )

        # Wine bottle 2
        cfgs.append(
            dict(
                name="wine_2",
                obj_groups=wine_2_path,
                graspable=True,
                placement=dict(
                    fixture=self.dining_table,
                    sample_region_kwargs=dict(
                        top_size=(0.50, 0.40)
                    ),
                    size=(0.0, 0.0),
                    pos=(0.78, -0.68),
                    rotation=(-0.1, 0.1),
                    ensure_object_boundary_in_range=False,
                    ensure_valid_placement=False,
                ),
            )
        )

        # Wine bottle 3
        cfgs.append(
            dict(
                name="wine_3",
                obj_groups=wine_4_path,
                graspable=True,
                placement=dict(
                    fixture=self.dining_table,
                    sample_region_kwargs=dict(
                        top_size=(0.50, 0.40)
                    ),
                    size=(0.0, 0.0),
                    pos=(0.78, -0.5),
                    rotation=(-0.1, 0.1),
                    ensure_object_boundary_in_range=False,
                    ensure_valid_placement=False,
                ),
            )
        )

        # Mug
        cfgs.append(
            dict(
                name="mug",
                obj_groups=mug_1_path,
                graspable=True,
                placement=dict(
                    fixture=self.dining_table,
                    sample_region_kwargs=dict(
                        top_size=(0.50, 0.40)
                    ),
                    size=(0.0, 0.0),
                    pos=(0.78, -0.35),
                    rotation=(1.57 - 0.1, 1.57 + 0.1),
                    ensure_object_boundary_in_range=False,
                    ensure_valid_placement=False,
                ),
            )
        )

        return cfgs

    def _check_cereal_on_table_mat(self):
        """Check if the cereal box is positioned on the table mat."""
        if self._table_mat_pos is None:
            return False

        try:
            cereal_pos = np.array(self.sim.data.body_xpos[self.obj_body_id["cereal"]])

            try:
                mat_body_id = self.sim.model.body_name2id("table_mat")
                mat_pos = np.array(self.sim.data.body_xpos[mat_body_id])
            except Exception:
                mat_pos = self._table_mat_pos

            dx = abs(cereal_pos[0] - mat_pos[0])
            dy = abs(cereal_pos[1] - mat_pos[1])
            dz = cereal_pos[2] - mat_pos[2]

            within_x = dx <= TABLE_MAT_SIZE[0] * 1.5
            within_y = dy <= TABLE_MAT_SIZE[1] * 1.5
            above_mat = -0.02 <= dz <= 0.15

            return within_x and within_y and above_mat
        except Exception:
            return False

    def _get_cereal_distance_to_mat(self):
        """Get the distance from cereal box to table mat."""
        if self._table_mat_pos is None:
            return float('inf')

        try:
            cereal_pos = np.array(self.sim.data.body_xpos[self.obj_body_id["cereal"]])

            try:
                mat_body_id = self.sim.model.body_name2id("table_mat")
                mat_pos = np.array(self.sim.data.body_xpos[mat_body_id])
            except Exception:
                mat_pos = self._table_mat_pos

            return np.linalg.norm(cereal_pos[:2] - mat_pos[:2])
        except Exception:
            return float('inf')

    def _post_action(self, action):
        reward, done, info = super()._post_action(action)

        cereal_on_mat = self._check_cereal_on_table_mat()
        info["cereal_on_table_mat"] = cereal_on_mat
        info["cereal_distance_to_mat"] = self._get_cereal_distance_to_mat()
        info["task_success"] = self._check_success()

        return reward, done, info

    def reward(self, action=None):
        try:
            reward = 0.0

            distance = self._get_cereal_distance_to_mat()
            if distance < float('inf'):
                reward += 1.0 / (distance + 0.1)

            if self._check_cereal_on_table_mat():
                reward += 10.0

            return reward
        except Exception:
            return 0.0

    def _check_success(self):
        try:
            cereal_on_mat = self._check_cereal_on_table_mat()
            gripper_away = OU.gripper_obj_far(self, "cereal")
            return cereal_on_mat and gripper_away
        except Exception:
            return False


class DamageableShelveItem(RSDamageableEnvironment, ShelveItem):
    """ShelveItem with damage tracking enabled."""

    def __init__(self, *args, **kwargs):
        super().__init__(task_name="shelve_item", *args, **kwargs)
