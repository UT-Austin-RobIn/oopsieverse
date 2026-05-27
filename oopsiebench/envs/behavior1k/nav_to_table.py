"""
Task configuration for **nav_to_table**.

Scene : Rs_int (Tiago primitives-style)
Robot : Tiago (tiago0)
Damage: mechanical only (task objects of interest)
"""

import numpy as np
import omnigibson as og
import torch as th

from oopsiebench.envs.behavior1k.base import TaskConfig

ROBOT_NAME = "tiago0"
ROBOT_TYPE = "Tiago"

# ── Task objects ─────────────────────────────────────────────────────────

TASK_OBJECTS = {
    "pedestal_table": {
        "type": "DatasetObject",
        "name": "pedestal_table",
        "category": "pedestal_table",
        "model": "djflkd",
        "position": [-0.5, 0.0, 0.10],
        "orientation": [0.0, 0.0, 0.0, 1.0],
        "scale": [0.5, 0.5, 1.0],
    },
    "vase": {
        "type": "DatasetObject",
        "name": "vase",
        "category": "vase",
        "model": "uuypot",
        "position": [-0.5, -1.0, 0.10],
        "orientation": [0.0, 0.0, 0.0, 1.0],
        "scale": [0.5, 0.5, 0.5],
    },
    "swivel_chair": {
        "type": "DatasetObject",
        "name": "swivel_chair",
        "category": "swivel_chair",
        "model": "pkpcew",
        "position": [-0.5, 1.0, 0.50],
        "orientation": [0.0, 0.0, 0.0, 1.0],
        "scale": [1.0, 1.0, 1.0],
    },
    "water_bottle": {
        "type": "DatasetObject",
        "name": "water_bottle",
        "category": "bottle_of_water",
        "model": "hrzznl",
        "position": [0.0, 0.0, 1.5],
        "orientation": [0.0, 0.0, 0.0, 1.0],
        "scale": [0.9, 0.9, 0.9],
    },
}

# ── Cameras ──────────────────────────────────────────────────────────────

VIEWER_CAMERA_POS = [1.5345655679702759, -2.3398592472076416, 1.3116816282272339]
VIEWER_CAMERA_ORN = [0.605172872543335, 0.14635765552520752, 0.18393288552761078, 0.7606010437011719]

EXTERNAL_CAMERA_CONFIGS = {
    "external_sensor_0": {
        "position": [0.4859, -1.8219, 1.1402],
        "orientation": [0.5857, -0.0093, -0.0129, 0.8103],
        "horizontal_aperture": 10.0,
    },
    "external_sensor_1": {
        "position": [0.2522, 0.0470, 1.0696],
        "orientation": [0.1991, -0.1991, -0.6785, 0.6785],
        "horizontal_aperture": 30.0,
    },
    "external_sensor_2": {
        "position": [-0.7765, -0.8203, 0.9939],
        "orientation": [0.4566, -0.3285, -0.4831, 0.6710],
        "horizontal_aperture": 30.0,
    },
    "external_sensor_3": {
        "position": [1.7508, -0.0198, 1.1778],
        "orientation": [0.3821, 0.4173, 0.6080, 0.5570],
        "horizontal_aperture": 20.0,
    },
}

# ── Public entry point ───────────────────────────────────────────────────

def get_task_config() -> TaskConfig:
    return TaskConfig(
        task_name="nav_to_table",
        use_gpu_dynamics=True,
        enable_transition_rules=False,
        scene_config={
            "type": "InteractiveTraversableScene",
            "scene_model": "Rs_int",
            "include_robots": False,
            "load_object_categories": ["floors", "walls", "breakfast_table"],
        },
        robot_name=ROBOT_NAME,
        robot_type=ROBOT_TYPE,
        robot_config={
            "type": ROBOT_TYPE,
            "name": ROBOT_NAME,
            "position": [0.0, 0.0, 0.0],
            "orientation": [0.0, 0.0, 0.0, 1.0],
            "default_arm_pose": "horizontal",
            "grasping_mode": "assisted",
            "obs_modalities": ["rgb", "depth"],
            "action_normalize": False,
            "self_collisions": True,
            "controller_config": {
                "arm_left": {
                    "name": "InverseKinematicsController",
                    "command_input_limits": None,
                },
                "gripper_left": {
                    "name": "MultiFingerGripperController",
                    "command_input_limits": (0.0, 1.0),
                    "mode": "smooth",
                },
                "arm_right": {
                    "name": "InverseKinematicsController",
                    "command_input_limits": None,
                },
                "gripper_right": {
                    "name": "MultiFingerGripperController",
                    "command_input_limits": (0.0, 1.0),
                    "mode": "smooth",
                },
            },
            "exclude_sensor_names": ["left_eef_link", "right_eef_link"],
        },
        task_objects=TASK_OBJECTS,
        viewer_camera_pos=VIEWER_CAMERA_POS,
        viewer_camera_orn=VIEWER_CAMERA_ORN,
        external_camera_configs=EXTERNAL_CAMERA_CONFIGS,
        target_objects_health_with_links=[
            "pedestal_table@base_link",
            "vase@base_link",
            "swivel_chair@base_link",
        ],
        target_objects_health=["pedestal_table", "vase", "swivel_chair"],
        target_objects_forces=[
            "pedestal_table@base_link",
            "vase@base_link",
            "swivel_chair@base_link",
        ],
        force_keys=["filtered_qs_forces", "impact_forces"],
        default_collect_hdf5="demos/behavior1k/teleop_data/nav_to_table.hdf5",
        default_playback_hdf5="demos/behavior1k/playback_data/nav_to_table_playback.hdf5",
        default_video_dir="demos/behavior1k/playback_videos/nav_to_table",
    )


_BOTTLE_U_XY = 0.03
_LIFT_Z = 0.1


def reset(env):
    """Snap the water bottle onto the breakfast_table top with light XY jitter."""
    try:
        bottle = env.scene.object_registry("name", "water_bottle")
        if bottle is None:
            return

        # breakfast_table is part of the base scene; substring match handles instanced names.
        table = None
        for obj in getattr(env.scene, "objects", []) or []:
            name = (getattr(obj, "name", "") or "").lower()
            cat = (getattr(obj, "category", "") or "").lower()
            if "breakfast_table" in name or "breakfast_table" in cat:
                table = obj
                break
        if table is None:
            table = (env.scene.object_registry("category", "breakfast_table")
                     or env.scene.object_registry("name", "breakfast_table"))
        if table is None:
            return

        for _ in range(5):
            og.sim.step()

        if not hasattr(table, "aabb") or table.aabb is None:
            return
        tmin, tmax = table.aabb

        cx = (float(tmin[0]) + float(tmax[0])) * 0.5
        cy = (float(tmin[1]) + float(tmax[1])) * 0.5

        z_offset = 0.05
        if hasattr(bottle, "aabb") and bottle.aabb is not None:
            mmin, mmax = bottle.aabb
            z_offset = max(0.02, 0.5 * float(mmax[2] - mmin[2])) + 0.002

        pos = th.tensor(
            [
                cx + float(np.random.uniform(-_BOTTLE_U_XY, _BOTTLE_U_XY)),
                cy + float(np.random.uniform(-_BOTTLE_U_XY, _BOTTLE_U_XY)),
                float(tmax[2]) + float(z_offset),
            ],
            dtype=th.float32,
        )
        bottle.set_position_orientation(pos, th.tensor([0.0, 0.0, 0.0, 1.0], dtype=th.float32))
        try:
            bottle.keep_still()
        except Exception:
            pass
        bottle_pos, _ = bottle.get_position_orientation()
        env._nav_bottle_start_z = float(bottle_pos[2])
    except Exception:
        return


def task_completion_check(env):
    start_z = getattr(env, "_nav_bottle_start_z", None)
    if start_z is None:
        return False
    bottle = env.scene.object_registry("name", "water_bottle")
    if bottle is None:
        return False
    bottle_pos, _ = bottle.get_position_orientation()
    return (float(bottle_pos[2]) - start_z) >= _LIFT_Z
