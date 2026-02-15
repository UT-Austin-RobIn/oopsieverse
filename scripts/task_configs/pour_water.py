"""
Task configuration for **pour_water**.

Scene : house_single_floor
Robot : FrankaPanda (franka0)
Damage: mechanical + electrical (water contacts)

Note: ``USE_GPU_DYNAMICS = True`` is required for fluid simulation.
"""

from scripts.task_configs.base import TaskConfig

ROBOT_NAME = "franka0"
ROBOT_TYPE = "FrankaPanda"

# ── Task objects ────────────────────────────────────────────────────────

TASK_OBJECTS = {
    "laptop": {
        "type": "DatasetObject",
        "name": "laptop",
        "category": "laptop",
        "model": "nvulcs",
        "position": [6.3, 0.2, 1.3],
        "orientation": [0.0, 0.0, 0.0, 1.0],
        "scale": [1.0, 1.0, 1.0],
    },
    "coffee_cup_1": {
        "type": "DatasetObject",
        "name": "coffee_cup_1",
        "category": "coffee_cup",
        "model": "ckkwmj",
        "position": [6.3, 0.0, 1.3],
        "orientation": [0.0, 0.0, 0.0, 1.0],
        "scale": [1.2, 1.2, 1.2],
    },
    "water_glass": {
        "type": "DatasetObject",
        "name": "water_glass",
        "category": "water_glass",
        "model": "ewgotr",
        "position": [6.3, 0.5, 1.3],
        "orientation": [0.0, 0.0, 0.0, 1.0],
        "scale": [0.9, 0.9, 0.9],
    },
}

# ── Cameras ─────────────────────────────────────────────────────────────

VIEWER_CAMERA_POS = [7.0659, -0.7141, 1.9185]
VIEWER_CAMERA_ORN = [0.4850, 0.1528, 0.2586, 0.8213]

EXTERNAL_CAMERA_CONFIGS = {
    "external_sensor_0": {
        "position": [7.3920, -0.6436, 1.7519],
        "orientation": [0.5273, 0.2970, 0.3907, 0.6936],
        "horizontal_aperture": 15.0,
    },
    "external_sensor_1": {
        "position": [7.1264, 1.1205, 2.0117],
        "orientation": [0.2131, 0.4377, 0.7853, 0.3824],
        "horizontal_aperture": 15.0,
    },
}


# ── Public entry point ──────────────────────────────────────────────────

def get_task_config() -> TaskConfig:
    return TaskConfig(
        task_name="pour_water",

        # OG macros
        use_gpu_dynamics=True,
        enable_transition_rules=False,

        # Scene
        scene_config={
            "scene_model": "house_single_floor",
            "not_load_object_categories": ["ottoman"],
            "load_room_instances": [
                "kitchen_0", "dining_room_0", "entryway_0", "living_room_0",
            ],
        },

        # Robot
        robot_name=ROBOT_NAME,
        robot_type=ROBOT_TYPE,
        robot_config={
            "type": ROBOT_TYPE,
            "name": ROBOT_NAME,
            "position": [6.8, 0.2, 1.0],
            "orientation": [0.0, 0.0, 1.0, 0.0],
            "grasping_mode": "assisted",
            "obs_modalities": ["rgb", "seg_instance", "proprio"],
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
            "laptop@base_link",
            "laptop@link_0",
        ],
        target_objects_health=[ROBOT_NAME, "laptop"],
        target_objects_water_contacts=["laptop@link_0", "laptop@base_link"],

        # Default paths
        default_collect_hdf5="resources/teleop_data/pour_water.hdf5",
        default_playback_hdf5="resources/playback_data/pour_water_playback.hdf5",
        default_video_dir="resources/videos/pour_water",
    )

