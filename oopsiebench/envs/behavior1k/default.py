"""
Default Behavior1k teleop scene: **Rs_int** with broad damage tracking.

Loads the kitchen scene only (no extra task-spawned objects). Damage tracking uses
**default mode**: every damageable object whose category is not in ``skip_categories``
(see ``damage_params.DAMAGEABLE_OBJECTS["default"]``).

Robot defaults match generic OG teleop (Tiago).
"""

from oopsiebench.envs.behavior1k.base import TaskConfig

ROBOT_NAME = "robot0"
ROBOT_TYPE = "FrankaPanda"


def get_task_config() -> TaskConfig:
    return TaskConfig(
        task_name="default",

        use_gpu_dynamics=False,
        enable_transition_rules=False,

        scene_config={
            "scene_model": "Rs_int",
        },

        robot_name=ROBOT_NAME,
        robot_type=ROBOT_TYPE,
        robot_config={
            "type": ROBOT_TYPE,
            "name": ROBOT_NAME,
            "grasping_mode": "assisted",
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
        },

        task_objects={},

        # viewer_camera_pos=[-0.37351322174072266, -0.9105080366134644, 0.9984497427940369],
        # viewer_camera_orn=[0.1866627037525177, 0.5293360948562622, 0.7805155515670776, 0.2752378284931183],
        external_camera_configs={},

        target_objects_health_with_links=[],
        target_objects_health=[],
        target_objects_forces=[],
        force_keys=[],

        default_collect_hdf5="demos/behavior1k/teleop_data/default.hdf5",
        default_playback_hdf5="demos/behavior1k/playback_data/default_playback.hdf5",
        default_video_dir="demos/behavior1k/playback_videos/default",
    )
