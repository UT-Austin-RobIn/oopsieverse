"""
Task configuration for **add_firewood** (a.k.a. firewood).

Scene : Rs_int
Robot : FrankaPanda (franka0)
Damage: mechanical + thermal
"""

from oopsiebench.envs.behavior1k.base import TaskConfig

ROBOT_NAME = "franka0"
ROBOT_TYPE = "FrankaPanda"

# ── Task objects ────────────────────────────────────────────────────────

TASK_OBJECTS = {
    "fireplace": {
        "type": "DatasetObject",
        "name": "fireplace",
        "category": "wood_fireplace",
        "model": "gpnsij",
        "position": [-1.5, -2.0, 0.5],
        "orientation": [0, 0, 0, 1],
        "scale": [1.0, 0.85, 0.85],
        "fixed_base": True,
        "abilities": {
            "heatSource": {
                "temperature": 100.0,
                "heating_rate": 0.1,
                "distance_threshold": 0.12,
                "requires_toggled_on": False,
            }
        },
        "initial_state": {"temperature": 100.0},
    },
    "log_center": {
        "type": "DatasetObject",
        "name": "log_center",
        "category": "log",
        "model": "pepele",
        "position": [-1.65, -2.0, 0.15],
        "orientation": [0, 0, 0, 1],
        "scale": [0.8, 0.6, 0.6],
        "abilities": {"flammable": {}},
        "initial_state": {"onFire": True},
    },
    "log_left": {
        "type": "DatasetObject",
        "name": "log_left",
        "category": "log",
        "model": "pepele",
        "position": [-1.65, -2.15, 0.17],
        "orientation": [0, 0, 0, 1],
        "scale": [0.8, 0.6, 0.6],
        "abilities": {"flammable": {}},
        "initial_state": {"onFire": True},
    },
    "target_object": {
        "type": "DatasetObject",
        "name": "target_object",
        "category": "log",
        "model": "pepele",
        "position": [-1.0, -2.25, 0.1],
        "orientation": [0, 0, 0, 1],
        "scale": [0.7, 0.5, 0.5],
        "abilities": {"flammable": {}},
        "initial_state": {"onFire": False},
    },
}

# ── Cameras ─────────────────────────────────────────────────────────────

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


# ── Public entry point ──────────────────────────────────────────────────

def get_task_config() -> TaskConfig:
    return TaskConfig(
        task_name="add_firewood",

        # OG macros
        use_gpu_dynamics=False,
        enable_transition_rules=False,

        # Scene
        scene_config={
            "type": "InteractiveTraversableScene",
            "scene_model": "Rs_int",
            "include_robots": False,
            "load_task_relevant_only": True,
        },

        # Robot
        robot_name=ROBOT_NAME,
        robot_type=ROBOT_TYPE,
        robot_config={
            "type": ROBOT_TYPE,
            "name": ROBOT_NAME,
            "position": [-0.85, -2.0, 0.0],
            "orientation": [0.0, 0.0, 1.0, 0.0],
            "grasping_mode": "assisted",
            "obs_modalities": ["rgb", "depth"],
            "action_normalize": False,
            "self_collisions": True,
            "controller_config": {
                "arm_0": {
                    "name": "InverseKinematicsController",
                    "command_input_limits": None,
                },
                "gripper_0": {
                    "name": "MultiFingerGripperController",
                    "command_input_limits": (0.0, 1.0),
                    "mode": "smooth",
                },
            },
        },

        # Objects
        task_objects=TASK_OBJECTS,

        # Cameras
        viewer_camera_pos=VIEWER_CAMERA_POS,
        viewer_camera_orn=VIEWER_CAMERA_ORN,
        external_camera_configs=EXTERNAL_CAMERA_CONFIGS,

        # Visualization
        target_objects_health_with_links=[
            f"{ROBOT_NAME}@eef_link",
            f"{ROBOT_NAME}@panda_hand",
            f"{ROBOT_NAME}@panda_leftfinger",
            f"{ROBOT_NAME}@panda_rightfinger",
            "fireplace@base_link",
            "log_center@base_link",
            "log_left@base_link",
            "target_object@base_link",
        ],
        target_objects_health=[ROBOT_NAME],
        target_objects_temperature=[
            f"{ROBOT_NAME}@eef_link",
            f"{ROBOT_NAME}@panda_hand",
            f"{ROBOT_NAME}@panda_leftfinger",
            f"{ROBOT_NAME}@panda_rightfinger",
            "fireplace@base_link",
            "log_center@base_link",
            "log_left@base_link",
            "target_object@base_link",
        ],
        target_objects_forces=[
            f"{ROBOT_NAME}@eef_link",
            f"{ROBOT_NAME}@panda_hand",
            f"{ROBOT_NAME}@panda_leftfinger",
            f"{ROBOT_NAME}@panda_rightfinger",
        ],
        force_keys=["filtered_qs_forces", "impact_forces"],

        # Default paths
        default_collect_hdf5="demos/behavior1k/teleop_data/add_firewood.hdf5",
        default_playback_hdf5="demos/behavior1k/playback_data/add_firewood_playback.hdf5",
        default_video_dir="demos/behavior1k/playback_videos/add_firewood",
    )

