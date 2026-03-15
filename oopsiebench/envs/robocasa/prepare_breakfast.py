"""
Prepare Breakfast environment for oopsieverse.

Task: place the mug and wine bottle onto the tray, then move the tray to the table mat.
"""

import numpy as np
import robocasa.utils.object_utils as OU
from robocasa.environments.kitchen.kitchen import FixtureType, Kitchen
from robocasa.models.objects.kitchen_object_utils import OBJ_CATEGORIES
from robocasa.models.scenes.scene_registry import LayoutType, StyleType

from robosuite.models.objects import BoxObject
from robosuite.utils.mjcf_utils import CustomMaterial

from damagesim.robosuite.damageable_env import RSDamageableEnvironment
from damagesim.robosuite.params.damage_params import get_params_for_object


TABLE_MAT_SIZE = [0.20, 0.15, 0.002]
TABLE_MAT_COLOR = [0.06, 0.10, 0.30, 1.0]


class PrepareBreakfast(Kitchen):

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
        ep_meta["lang"] = "Place the mug and wine bottle onto the tray, then move the tray to the table mat"
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

    def _add_table_mat(self):
        """Add a table mat to the dining table."""
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

        mat_x = dining_x - 0.5
        mat_y = dining_y + 0.23
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
        tray_4_path = next(
            p for p in OBJ_CATEGORIES["tray"]["objaverse"].mjcf_paths
            if p.split("/")[-2] == "tray_4"
        )
        mug_1_path = next(
            p for p in OBJ_CATEGORIES["mug"]["objaverse"].mjcf_paths
            if p.split("/")[-2] == "mug_1"
        )
        wine_5_path = next(
            p for p in OBJ_CATEGORIES["wine"]["objaverse"].mjcf_paths
            if p.split("/")[-2] == "wine_5"
        )

        cfgs = []

        # Tray centered in front of robot on dining table
        cfgs.append(
            dict(
                name="tray",
                obj_groups=tray_4_path,
                graspable=True,
                placement=dict(
                    fixture=self.dining_table,
                    sample_region_kwargs=dict(
                        top_size=(0.50, 0.40)
                    ),
                    size=(0.0, 0.0),
                    pos=(0.0, -0.5),
                    rotation=0.0,
                    ensure_object_boundary_in_range=False,
                    ensure_valid_placement=False,
                ),
            )
        )

        # Mug positioned to the left of the tray
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
                    size=(0.20, 0.20),
                    pos=(-0.23, -0.4),
                    rotation=5.0,
                ),
            )
        )

        # Wine bottle positioned to the right of the tray
        cfgs.append(
            dict(
                name="wine",
                obj_groups=wine_5_path,
                graspable=True,
                placement=dict(
                    fixture=self.dining_table,
                    sample_region_kwargs=dict(
                        top_size=(0.50, 0.40)
                    ),
                    size=(0.20, 0.20),
                    pos=(-0.2, -0.8),
                    rotation=0.0,
                ),
            )
        )

        return cfgs

    def _check_obj_in_tray(self, obj_name):
        """Check if an object is inside the tray."""
        try:
            return OU.check_obj_in_receptacle(self, obj_name, "tray")
        except Exception:
            return False

    def _check_tray_on_table_mat(self):
        """Check if the tray is positioned on the table mat."""
        if self._table_mat_pos is None:
            return False

        try:
            tray_pos = np.array(self.sim.data.body_xpos[self.obj_body_id["tray"]])

            try:
                mat_body_id = self.sim.model.body_name2id("table_mat")
                mat_pos = np.array(self.sim.data.body_xpos[mat_body_id])
            except Exception:
                mat_pos = self._table_mat_pos

            dx = abs(tray_pos[0] - mat_pos[0])
            dy = abs(tray_pos[1] - mat_pos[1])
            dz = tray_pos[2] - mat_pos[2]

            within_x = dx <= TABLE_MAT_SIZE[0] * 1.5
            within_y = dy <= TABLE_MAT_SIZE[1] * 1.5
            above_mat = -0.02 <= dz <= 0.15

            return within_x and within_y and above_mat
        except Exception:
            return False

    def _get_tray_distance_to_mat(self):
        """Get the distance from tray to table mat."""
        if self._table_mat_pos is None:
            return float('inf')

        try:
            tray_pos = np.array(self.sim.data.body_xpos[self.obj_body_id["tray"]])

            try:
                mat_body_id = self.sim.model.body_name2id("table_mat")
                mat_pos = np.array(self.sim.data.body_xpos[mat_body_id])
            except Exception:
                mat_pos = self._table_mat_pos

            return np.linalg.norm(tray_pos[:2] - mat_pos[:2])
        except Exception:
            return float('inf')

    def _post_action(self, action):
        reward, done, info = super()._post_action(action)

        mug_in_tray = self._check_obj_in_tray("mug")
        wine_in_tray = self._check_obj_in_tray("wine")
        tray_on_mat = self._check_tray_on_table_mat()
        all_in_tray = mug_in_tray and wine_in_tray

        info["mug_in_tray"] = mug_in_tray
        info["wine_in_tray"] = wine_in_tray
        info["all_items_in_tray"] = all_in_tray
        info["tray_on_table_mat"] = tray_on_mat
        info["tray_distance_to_mat"] = self._get_tray_distance_to_mat()
        info["task_success"] = self._check_success()

        return reward, done, info

    def reward(self, action=None):
        try:
            reward = 0.0

            if self._check_obj_in_tray("mug"):
                reward += 3.0
            if self._check_obj_in_tray("wine"):
                reward += 3.0

            all_in_tray = (
                self._check_obj_in_tray("mug") and
                self._check_obj_in_tray("wine")
            )
            if all_in_tray:
                reward += 4.0

                distance = self._get_tray_distance_to_mat()
                if distance < float('inf'):
                    reward += 1.0 / (distance + 0.1)

            if self._check_tray_on_table_mat():
                reward += 5.0
                if all_in_tray:
                    reward += 10.0

            return reward
        except Exception:
            return 0.0

    def _check_success(self):
        try:
            mug_in_tray = self._check_obj_in_tray("mug")
            wine_in_tray = self._check_obj_in_tray("wine")
            tray_on_mat = self._check_tray_on_table_mat()
            gripper_away = OU.gripper_obj_far(self, "tray")

            return mug_in_tray and wine_in_tray and tray_on_mat and gripper_away
        except Exception:
            return False


class DamageablePrepareBreakfast(RSDamageableEnvironment, PrepareBreakfast):
    """PrepareBreakfast with damage tracking enabled."""

    def __init__(self, *args, **kwargs):
        super().__init__(task_name="prepare_breakfast", *args, **kwargs)
