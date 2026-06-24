"""
Shelve Item environment for oopsieverse.

Task: pick the cereal box and place it on the table mat.
"""

import json
import os
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation as R

import robocasa.utils.env_utils as EnvUtils
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

TABLE_MAT_SIZE = [0.10, 0.25, 0.004]
TABLE_MAT_COLOR = [0.06, 0.10, 0.30, 1.0]
GRIPPER_MAT_CLEARANCE = 0.4

# Fixed-base Panda lacks kitchen agentview cameras (parented to mobile base).
# World-fixed cameras tuned via teleop free camera (resources/camera_states/).
TASK_CAMERA_NAME = "shelve_item_right"
TASK_CAMERA_FOVY = "55"
TASK_CAMERA_FILES = {
    "shelve_item_left": "shelve_item_1.json",
    "shelve_item_right": "shelve_item_2.json",
}


def _repo_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def _freecam_to_mjcf_pose(lookat, distance, azimuth, elevation):
    """Convert MuJoCo passive-viewer free camera params to MJCF pos + quat (wxyz)."""
    lookat = np.asarray(lookat, dtype=float)
    az = np.deg2rad(azimuth)
    el = np.deg2rad(elevation)
    direction = np.array([
        np.cos(el) * np.cos(az),
        np.cos(el) * np.sin(az),
        np.sin(el),
    ])
    cam_pos = lookat - distance * direction
    forward = lookat - cam_pos
    forward /= np.linalg.norm(forward)
    up_world = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, up_world)
    if np.linalg.norm(right) < 1e-8:
        right = np.array([1.0, 0.0, 0.0])
    else:
        right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    rot = np.stack([right, up, -forward], axis=1)
    quat_xyzw = R.from_matrix(rot).as_quat()
    quat_wxyz = [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]
    return cam_pos.tolist(), quat_wxyz


def _load_task_camera_pose(camera_json):
    path = os.path.join(_repo_root(), "resources", "camera_states", camera_json)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return _freecam_to_mjcf_pose(
        data["lookat"],
        data["distance"],
        data["azimuth"],
        data["elevation"],
    )


# ═══════════════════════════════════════════════════════════════════════
# ShelveItem environment
# ═══════════════════════════════════════════════════════════════════════


class ShelveItem(Kitchen):

    def __init__(self, *args, **kwargs):
        kwargs.pop("layout_ids", None)
        kwargs.pop("style_ids", None)
        kwargs.pop("obj_registries", None)

        # Kitchen defaults render_camera to "robot0_agentview_center", which is
        # parented to mobilebase0_support — that body doesn't exist on a
        # fixed-base Panda, so the camera is silently skipped. Fall back to a
        # camera Panda actually ships with.
        kwargs.setdefault("render_camera", TASK_CAMERA_NAME)

        self.table_mat = None
        self._table_mat_pos = None
        self.randomize_scene = False

        super().__init__(
            layout_ids=LayoutType.LAYOUT010,
            style_ids=StyleType.STYLE010,
            obj_registries=("objaverse", "lightwheel", "aigen"),
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

        # Centered along the dining counter's length, set slightly back from
        # the row of objects so the Panda can reach them but isn't on top of them.
        robot_offset = [0.4, 0.7]
        pos, ori = EnvUtils.compute_robot_base_placement_pose(
            self, ref_fixture=self.dining_table, offset=robot_offset
        )
        self.init_robot_base_pos_anchor = pos
        self.init_robot_base_ori_anchor = ori

        # Panda is fixed-base: kitchen's mobile-base reset path won't move it,
        # so write the actual base pose into the model XML before sim compile.
        # Yaw the robot 270° CCW (= 90° CW) from the natural placement so it
        # faces toward the row of objects.
        robot_model = self.robots[0].robot_model
        ori_with_yaw = np.array(ori, dtype=float)
        ori_with_yaw[2] += 3 * np.pi / 2
        robot_model.set_base_xpos([pos[0], pos[1], pos[2]])
        robot_model.set_base_ori(ori_with_yaw)
        self.init_robot_base_ori_anchor = ori_with_yaw

        self._add_task_cameras()

    def _add_task_cameras(self):
        existing = {cam.get("name") for cam in self.model.worldbody.findall("camera")}
        for cam_name, json_file in TASK_CAMERA_FILES.items():
            if cam_name in existing:
                continue
            pos, quat = _load_task_camera_pose(json_file)
            cam = ET.Element("camera")
            cam.set("mode", "fixed")
            cam.set("name", cam_name)
            cam.set("pos", " ".join(map(str, pos)))
            cam.set("quat", " ".join(map(str, quat)))
            cam.set("fovy", TASK_CAMERA_FOVY)
            self.model.worldbody.append(cam)
            self._cam_configs[cam_name] = {
                "pos": pos,
                "quat": quat,
                "camera_attribs": {"fovy": TASK_CAMERA_FOVY},
            }

    def _reset_internal(self):
        # Kitchen._reset_internal calls EnvUtils.set_robot_base, which assumes
        # mobile-base joints. Panda is fixed-base, so we skip Kitchen here and
        # replicate the bits we still need: manipulation-env reset, scene
        # setup, object placements, and the post-reset settle steps.
        super(Kitchen, self)._reset_internal()

        self._setup_scene()

        if not self.deterministic_reset and self.placement_initializer is not None:
            object_placements = self.object_placements
            self._update_sliding_fxtr_obj_placement()
            for obj_pos, obj_quat, obj in object_placements.values():
                self.sim.data.set_joint_qpos(
                    obj.joints[0],
                    np.concatenate([np.array(obj_pos), np.array(obj_quat)]),
                )

        self.init_robot_base_pos = self.init_robot_base_pos_anchor
        self.init_robot_base_ori = self.init_robot_base_ori_anchor

        action = np.zeros(self.action_spec[0].shape)
        policy_step = True
        for _ in range(10 * int(self.control_timestep / self.model_timestep)):
            self.sim.step1()
            self._pre_action(action, policy_step)
            self.sim.step2()
            policy_step = False

    def _add_table_mat(self):
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

        # Place the mat under the row of objects, with its front edge tucked
        # ≈ 1.5 cm inside the counter's front edge so nothing overhangs.
        mat_x = dining_x - 1.02
        mat_y = dining_y + 0.0
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
            if os.path.basename(os.path.dirname(p)) == "cereal_2"
        )
        wine_5_path = next(
            p for p in OBJ_CATEGORIES["wine"]["objaverse"].mjcf_paths
            if os.path.basename(os.path.dirname(p)) == "wine_5"
        )
        wine_glass_path = next(
            p for p in OBJ_CATEGORIES["wine_glass"]["aigen"].mjcf_paths
            if os.path.basename(os.path.dirname(p)) == "wine_glass_1"
        )
        wine_2_path = next(
            p for p in OBJ_CATEGORIES["wine"]["objaverse"].mjcf_paths
            if os.path.basename(os.path.dirname(p)) == "wine_2"
        )
        flour_bag_path = next(
            p for p in OBJ_CATEGORIES["flour_bag"]["lightwheel"].mjcf_paths
            if os.path.basename(os.path.dirname(p)) == "FlourBag002"
        )

        cfgs = []

        # Front edge of the dining counter, in a row spread along its length so
        # the Panda can swing across them and potentially knock them off.
        edge_x = 0.95
        row_ys = [-0.60, -0.45, -0.1, 0.10, 0.40]

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
                    size=(0.05, 0.05),
                    pos=(edge_x-0.4, row_ys[0]),
                    rotation=(1.57, 1.57),
                    ensure_object_boundary_in_range=False,
                    ensure_valid_placement=False,
                ),
            )
        )

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
                    size=(0.02, 0.04),
                    pos=(edge_x-0.04, row_ys[1]),
                    rotation=(-0.1, 0.1),
                    ensure_object_boundary_in_range=False,
                    ensure_valid_placement=False,
                ),
            )
        )

        cfgs.append(
            dict(
                name="wine_glass",
                obj_groups=wine_glass_path,
                graspable=True,
                placement=dict(
                    fixture=self.dining_table,
                    sample_region_kwargs=dict(
                        top_size=(0.50, 0.40)
                    ),
                    size=(0.02, 0.04),
                    pos=(edge_x-0.04, row_ys[2]),
                    rotation=(-0.1, 0.1),
                    ensure_object_boundary_in_range=False,
                    ensure_valid_placement=False,
                ),
            )
        )

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
                    size=(0.02, 0.04),
                    pos=(edge_x-0.04, row_ys[3]),
                    rotation=(-0.1, 0.1),
                    ensure_object_boundary_in_range=False,
                    ensure_valid_placement=False,
                ),
            )
        )

        cfgs.append(
            dict(
                name="flour_bag",
                obj_groups=flour_bag_path,
                graspable=True,
                placement=dict(
                    fixture=self.dining_table,
                    sample_region_kwargs=dict(
                        top_size=(0.50, 0.40)
                    ),
                    size=(0.02, 0.04),
                    pos=(edge_x-0.04, row_ys[4]),
                    rotation=(1.57, 1.57),
                    ensure_object_boundary_in_range=False,
                    ensure_valid_placement=False,
                ),
            )
        )

        return cfgs

    # ── Task checks ────────────────────────────────────────────────────

    def _get_mat_pos(self):
        if self._table_mat_pos is None:
            return None
        try:
            mat_body_id = self.sim.model.body_name2id("table_mat_main")
            return np.array(self.sim.data.body_xpos[mat_body_id])
        except Exception:
            return np.array(self._table_mat_pos)

    def _check_cereal_on_table_mat(self):
        mat_pos = self._get_mat_pos()
        if mat_pos is None:
            return False

        try:
            cereal_pos = np.array(self.sim.data.body_xpos[self.obj_body_id["cereal"]])

            dx = abs(cereal_pos[0] - mat_pos[0])
            dy = abs(cereal_pos[1] - mat_pos[1])
            dz = cereal_pos[2] - mat_pos[2]

            within_x = dx <= TABLE_MAT_SIZE[0]
            within_y = dy <= TABLE_MAT_SIZE[1]
            above_mat = -0.02 <= dz <= 0.2

            return within_x and within_y and above_mat
        except Exception:
            return False

    def _get_cereal_distance_to_mat(self):
        mat_pos = self._get_mat_pos()
        if mat_pos is None:
            return float("inf")

        try:
            cereal_pos = np.array(self.sim.data.body_xpos[self.obj_body_id["cereal"]])
            return np.linalg.norm(cereal_pos[:2] - mat_pos[:2])
        except Exception:
            return float("inf")

    def _gripper_far_from_mat(self, th=GRIPPER_MAT_CLEARANCE):
        mat_pos = self._get_mat_pos()
        if mat_pos is None:
            return False
        gripper_site_pos = self.sim.data.site_xpos[self.robots[0].eef_site_id["right"]]
        return np.linalg.norm(gripper_site_pos - mat_pos) > th

    def _post_action(self, action):
        reward, done, info = super()._post_action(action)

        cereal_on_mat = self._check_cereal_on_table_mat()
        cereal_not_grasped = not OU.check_obj_grasped(self, "cereal")
        gripper_away_from_mat = self._gripper_far_from_mat()
        info["cereal_on_table_mat"] = cereal_on_mat
        info["cereal_not_grasped"] = cereal_not_grasped
        info["gripper_away_from_mat"] = gripper_away_from_mat
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
            return (
                self._check_cereal_on_table_mat()
                and not OU.check_obj_grasped(self, "cereal")
                and self._gripper_far_from_mat()
            )
        except Exception:
            return False


# ═══════════════════════════════════════════════════════════════════════
# Damageable variant
# ═══════════════════════════════════════════════════════════════════════


class DamageableShelveItem(RSDamageableEnvironment, ShelveItem):
    """ShelveItem with damage tracking enabled."""

    def __init__(self, *args, **kwargs):
        super().__init__(task_name="shelve_item", *args, **kwargs)
