"""
Task configuration for **open_drawer**.

Based on `add_firewood.py`, but:
- Robot: Tiago (tiago0) instead of FrankaPanda (franka0), same pose
- Target object: a drawer (bottom_cabinet) instead of fireplace + logs
- No logs / no thermal-flammable setup
"""

from __future__ import annotations

from oopsiebench.envs.behavior1k.base import TaskConfig

ROBOT_NAME = "tiago0"
ROBOT_TYPE = "Tiago"


TASK_OBJECTS = {
    "drawer": {
        "type": "DatasetObject",
        "name": "drawer",
        "category": "bottom_cabinet",
        # BehaviorKB page: bottom_cabinet-bamfsz.
        # In our asset pipeline, `model` is typically the dataset asset id.
        "model": "bamfsz",
        # Start it low; if the asset origin is offset, `reset()` will correct
        # it further by snapping the AABB min-z to ~0.0.
        "position": [-1.5, -2.0, 0.10],
        "orientation": [0, 0, 0, 1],
        "scale": [1.0, 0.85, 0.85],
        "fixed_base": True,
    },
}


# ── Cameras (copied from add_firewood.py) ──────────────────────────────

VIEWER_CAMERA_POS = [-0.37351322174072266, -0.9105080366134644, 0.9984497427940369]
VIEWER_CAMERA_ORN = [0.1866627037525177, 0.5293360948562622, 0.7805155515670776, 0.2752378284931183]

EXTERNAL_CAMERA_CONFIGS = {
    "external_sensor_0": {
        "position": VIEWER_CAMERA_POS,
        "orientation": VIEWER_CAMERA_ORN,
        "horizontal_aperture": 30.0,
    },
    "external_sensor_1": {
        "position": [-0.5087745785713196, -3.052588701248169, 0.9984493851661682],
        "orientation": [0.5276271104812622, 0.19144046306610107, 0.2822819948196411, 0.7779955267906189],
        "horizontal_aperture": 30.0,
    },
}


def get_task_config() -> TaskConfig:
    return TaskConfig(
        task_name="open_drawer",

        # OG macros (copied from add_firewood.py)
        use_gpu_dynamics=False,
        enable_transition_rules=False,

        # Scene (copied from add_firewood.py)
        scene_config={
            "type": "InteractiveTraversableScene",
            "scene_model": "Rs_int",
            "include_robots": False,
            "load_task_relevant_only": True,
        },

        # Robot: Tiago in the same pose as the add_firewood Franka.
        robot_name=ROBOT_NAME,
        robot_type=ROBOT_TYPE,
        robot_config={
            "type": ROBOT_TYPE,
            "name": ROBOT_NAME,
            # Back the robot up a bit relative to the drawer.
            "position": [-0.65, -2.0, 0.0],
            "orientation": [0.0, 0.0, 1.0, 0.0],

            # Tiago controller setup (copied from nav_to_table.py).
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

        # Objects: only the drawer
        task_objects=TASK_OBJECTS,

        # Cameras
        viewer_camera_pos=VIEWER_CAMERA_POS,
        viewer_camera_orn=VIEWER_CAMERA_ORN,
        external_camera_configs=EXTERNAL_CAMERA_CONFIGS,

        # Visualization / damage tracking: track the drawer only
        target_objects_health_with_links=["drawer@base_link"],
        target_objects_health=["drawer"],
        target_objects_forces=["drawer@base_link"],
        force_keys=["filtered_qs_forces", "impact_forces"],

        # Default paths
        default_collect_hdf5="demos/behavior1k/teleop_data/open_drawer.hdf5",
        default_playback_hdf5="demos/behavior1k/playback_data/open_drawer_playback.hdf5",
        default_video_dir="demos/behavior1k/playback_videos/open_drawer",
    )


def reset(env):
    """
    Task-specific reset used by teleop playback.

    Ensures the drawer isn't floating by snapping its AABB min-z to ~0.0.
    """
    if not getattr(env, "scene", None):
        return

    drawer = env.scene.object_registry("name", "drawer")
    if drawer is None:
        return

    # If the asset's origin is not aligned to the floor plane, it may appear
    # "floating". Using AABB min-z provides a robust correction.
    if hasattr(drawer, "aabb") and drawer.aabb is not None:
        try:
            aabb_min, _ = drawer.aabb
            cur_min_z = float(aabb_min[2])
        except Exception:
            cur_min_z = None
    else:
        cur_min_z = None

    if cur_min_z is not None:
        target_min_z = 0.0
        dz = target_min_z - cur_min_z
        pos, orn = drawer.get_position_orientation()
        # Preserve x/y and orientation; only shift z.
        new_pos = [float(pos[0]), float(pos[1]), float(pos[2]) + dz]
        drawer.set_position_orientation(new_pos, orn)
        if hasattr(drawer, "keep_still"):
            drawer.keep_still()


