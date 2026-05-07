"""
Task configuration for **turn_on_stove**.

Scene : Rs_int
Robot : FrankaPanda (franka0)
Damage: mechanical + thermal
"""

import pickle
import numpy as np
import torch as th
import omnigibson as og
from omnigibson.utils import transform_utils as T
from omnigibson.controllers.controller_base import IsGraspingState
from oopsiebench.envs.behavior1k.base import TaskConfig
from omnigibson import object_states
    
ROBOT_NAME = "franka0"
ROBOT_TYPE = "FrankaPanda"

# ── Task objects ────────────────────────────────────────────────────────

TASK_OBJECTS = {
    "stove": {
        "type": "DatasetObject",
        "name": "stove",
        "category": "stove",
        "model": "igwqpj",
        "position": [-1.5, -2.0, 0.5],
        "orientation": [0, 0, 0, 1],
    }
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

INIT_STATE_PATH = None
# ── Public entry point ──────────────────────────────────────────────────

def get_task_config() -> TaskConfig:
    return TaskConfig(
        task_name="turn_on_stove",

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
        ],
        target_objects_health=[ROBOT_NAME],
        target_objects_temperature=[
            f"{ROBOT_NAME}@eef_link",
            f"{ROBOT_NAME}@panda_hand",
            f"{ROBOT_NAME}@panda_leftfinger",
            f"{ROBOT_NAME}@panda_rightfinger",
        ],
        target_objects_forces=[
            f"{ROBOT_NAME}@eef_link",
            f"{ROBOT_NAME}@panda_hand",
            f"{ROBOT_NAME}@panda_leftfinger",
            f"{ROBOT_NAME}@panda_rightfinger",
            f"{ROBOT_NAME}@panda_link7",
            f"{ROBOT_NAME}@panda_link6",
            f"{ROBOT_NAME}@panda_link5",
        ],
        force_keys=["filtered_qs_forces"],

        # Default paths
        default_collect_hdf5="demos/behavior1k/teleop_data/turn_on_stove.hdf5",
        default_playback_hdf5="demos/behavior1k/playback_data/turn_on_stove_playback.hdf5",
        default_video_dir="demos/behavior1k/playback_videos/turn_on_stove",
    )

_U_XY = 0.03
_U_YAW = 0.12
_U_ARM = 0.07

def reset(env):
    """Small uniform noise on base xyz, yaw, and arm joints; settle briefly."""
    
    # Load initial state
    if INIT_STATE_PATH is not None:
        with open(INIT_STATE_PATH, "rb") as f: state_flat_array = pickle.load(f)
        og.sim.load_state(state_flat_array, serialized=True)
    
    # Reset the robot's position and orientation
    if not getattr(env, "robots", None):
        return
    robot = env.robots[0]
    pos, orn = robot.get_position_orientation()
    pos = pos.clone()
    pos[0] += float(np.random.uniform(-_U_XY, _U_XY))
    pos[1] += float(np.random.uniform(-_U_XY, _U_XY))
    euler = T.quat2euler(orn).clone()
    euler[2] = euler[2] + float(np.random.uniform(-_U_YAW, _U_YAW))
    orn = T.euler2quat(euler)
    robot.set_position_orientation(pos, orn)

    # Reset the robot's arm joints
    q = robot.get_joint_positions().clone()
    for arm_name in robot.arm_control_idx:
        idx = robot.arm_control_idx[arm_name]
        u = (th.rand(len(idx), device=q.device, dtype=q.dtype) * 2 - 1) * _U_ARM
        q[idx] = q[idx] + u
    robot.set_joint_positions(q)
    robot.set_joint_velocities(th.zeros(robot.n_dof, device=q.device, dtype=q.dtype))
    robot.keep_still()
    
    for _ in range(10): og.sim.step()

def task_completion_check(env):
    stove = env.scene.object_registry("name", "stove")
    return stove.states[object_states.ToggledOn].get_value()
