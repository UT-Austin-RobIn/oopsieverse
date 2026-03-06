"""
Pastry Display environment for oopsieverse.

Task: place the pastry onto the plate, then move the plate to the table mat.
"""

import os

import numpy as np
import robocasa.utils.object_utils as OU
from robocasa.environments.kitchen.kitchen import FixtureType, Kitchen
from robocasa.models.objects.kitchen_object_utils import OBJ_CATEGORIES
from robocasa.models.scenes.scene_registry import LayoutType, StyleType

from robosuite.models.objects import BoxObject
from robosuite.utils.mjcf_utils import CustomMaterial

from damagesim.robosuite.damageable_env import RSDamageableEnvironment
from damagesim.robosuite.params.damage_params import get_params_for_object


# ═══════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════

TABLE_MAT_SIZE = [0.20, 0.15, 0.002]
TABLE_MAT_COLOR = [0.06, 0.10, 0.30, 1.0]


# ═══════════════════════════════════════════════════════════════════════
# PastryDisplay environment
# ═══════════════════════════════════════════════════════════════════════


class PastryDisplay(Kitchen):

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
        """Get episode metadata with task description."""
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = "Place the pastry onto the plate, then move the plate to the table mat"
        return ep_meta

    def _setup_kitchen_references(self):
        """Setup kitchen fixture references."""
        super()._setup_kitchen_references()

        self.sink = self.register_fixture_ref("sink", dict(id=FixtureType.SINK))

        self.dining_table = self.register_fixture_ref(
            "dining_table",
            dict(id=FixtureType.DINING_COUNTER, ref=FixtureType.STOOL, size=(0.75, 0.2)),
        )

        self.init_robot_base_ref = self.dining_table

    def _load_model(self, **kwargs):
        """Load model and add table mat to the dining table."""
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

        mat_x = dining_x - 0.1
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
        cfgs = []

        # Exact model paths from OBJ_CATEGORIES registry
        plate_paths = OBJ_CATEGORIES["plate"]["objaverse"].mjcf_paths
        plate_4_path = next(p for p in plate_paths if p.endswith("plate_4" + os.sep + "model.xml"))

        cupcake_paths = OBJ_CATEGORIES["cupcake"]["objaverse"].mjcf_paths
        cupcake_2_path = next(p for p in cupcake_paths if p.endswith("cupcake_2" + os.sep + "model.xml"))

        # Plate on the dining table
        cfgs.append(
            dict(
                name="plate",
                obj_groups=plate_4_path,
                placement=dict(
                    fixture=self.dining_table,
                    size=(0, 0),
                    pos=(-0.2, -0.6),
                    rotation=0.0,
                    ensure_object_boundary_in_range=False,
                    ensure_valid_placement=False,
                ),
            )
        )

        # Pastry on the dining table
        cfgs.append(
            dict(
                name="pastry",
                obj_groups=cupcake_2_path,
                graspable=True,
                placement=dict(
                    fixture=self.dining_table,
                    size=(0.20, 0.20),
                    pos=(-0.4, -0.4),
                    rotation=0.0,
                    ensure_object_boundary_in_range=False,
                    ensure_valid_placement=False,
                ),
            )
        )

        return cfgs

    # ── Task checks ────────────────────────────────────────────────────

    def _check_pastry_on_plate(self):
        """Check if the pastry is positioned on the plate."""
        try:
            pastry_pos = np.array(self.sim.data.body_xpos[self.obj_body_id["pastry"]])
            plate_pos = np.array(self.sim.data.body_xpos[self.obj_body_id["plate"]])

            dx = abs(pastry_pos[0] - plate_pos[0])
            dy = abs(pastry_pos[1] - plate_pos[1])
            dz = pastry_pos[2] - plate_pos[2]

            plate_radius = 0.12 
            within_xy = (dx <= plate_radius) and (dy <= plate_radius)
            above_plate = 0.0 <= dz <= 0.15 
            
            return within_xy and above_plate
        except Exception:
            return False

    def _check_plate_on_table_mat(self):
        """Check if the plate is positioned on the table mat."""
        if self._table_mat_pos is None:
            return False

        try:
            plate_pos = np.array(self.sim.data.body_xpos[self.obj_body_id["plate"]])

            try:
                mat_body_id = self.sim.model.body_name2id("table_mat")
                mat_pos = np.array(self.sim.data.body_xpos[mat_body_id])
            except Exception:
                mat_pos = self._table_mat_pos

            dx = abs(plate_pos[0] - mat_pos[0])
            dy = abs(plate_pos[1] - mat_pos[1])
            dz = plate_pos[2] - mat_pos[2]

            within_x = dx <= TABLE_MAT_SIZE[0] * 1.5
            within_y = dy <= TABLE_MAT_SIZE[1] * 1.5
            above_mat = -0.02 <= dz <= 0.15

            return within_x and within_y and above_mat
        except Exception:
            return False

    def _get_plate_distance_to_mat(self):
        """Get the distance from plate to table mat."""
        if self._table_mat_pos is None:
            return float('inf')

        try:
            plate_pos = np.array(self.sim.data.body_xpos[self.obj_body_id["plate"]])

            try:
                mat_body_id = self.sim.model.body_name2id("table_mat")
                mat_pos = np.array(self.sim.data.body_xpos[mat_body_id])
            except Exception:
                mat_pos = self._table_mat_pos

            return np.linalg.norm(plate_pos[:2] - mat_pos[:2])
        except Exception:
            return float('inf')

    def _post_action(self, action):
        """Called after each action for task progress tracking."""
        reward, done, info = super()._post_action(action)

        pastry_on_plate = self._check_pastry_on_plate()
        plate_on_mat = self._check_plate_on_table_mat()

        info["pastry_on_plate"] = pastry_on_plate
        info["plate_on_table_mat"] = plate_on_mat
        info["plate_distance_to_mat"] = self._get_plate_distance_to_mat()
        info["task_success"] = self._check_success()

        return reward, done, info

    def reward(self, action=None):
        """
        Reward based on task progress.
        - Reward for placing pastry on plate
        - Progressive reward for moving plate toward table mat
        - Large bonus for placing plate on table mat with pastry
        """
        try:
            reward = 0.0

            pastry_on_plate = self._check_pastry_on_plate()

            if pastry_on_plate:
                reward += 5.0

                distance = self._get_plate_distance_to_mat()
                if distance < float('inf'):
                    reward += 1.0 / (distance + 0.1)

            if self._check_plate_on_table_mat():
                reward += 5.0
                if pastry_on_plate:
                    reward += 10.0

            return reward
        except Exception:
            return 0.0

    def _check_success(self):
        """
        Check if the task is successful.
        Success requires:
        1. Pastry on the plate
        2. Plate on the table mat
        3. Gripper not holding the plate
        """
        try:
            pastry_on_plate = self._check_pastry_on_plate()
            plate_on_mat = self._check_plate_on_table_mat()
            gripper_away = OU.gripper_obj_far(self, "plate")

            return pastry_on_plate and plate_on_mat and gripper_away
        except Exception:
            return False



# ═══════════════════════════════════════════════════════════════════════
# Damageable variant
# ═══════════════════════════════════════════════════════════════════════


class DamageablePastryDisplay(RSDamageableEnvironment, PastryDisplay):
    """PastryDisplay with damage tracking enabled."""

    def __init__(self, *args, **kwargs):
        super().__init__(task_name="pastry_display", *args, **kwargs)