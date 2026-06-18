"""
Task configuration for **add_firewood** (a.k.a. firewood).

Scene : Rs_int
Robot : FrankaPanda (franka0)
Damage: mechanical + thermal
"""

from __future__ import annotations

import pickle

import numpy as np
import omnigibson as og
import torch as th
from omnigibson.controllers.controller_base import IsGraspingState
from omnigibson.utils import transform_utils as T

from oopsiebench.envs.behavior1k.base import TaskConfig
from oopsiebench.envs.behavior1k.spatial_checks import gripper_far_from_object

ROBOT_NAME = "franka0"
ROBOT_TYPE = "FrankaPanda"

INIT_STATE_PATH = "oopsiebench/envs/behavior1k/init_states/add_firewood.pkl"

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

_U_XY = 0.03
_U_YAW = 0.12
_U_ARM = 0.07


def reset(env):
    """Small uniform noise on base xyz, yaw, and arm joints; settle briefly."""
    # Load initial state
    with open(INIT_STATE_PATH, "rb") as f:
        state_flat_array = pickle.load(f)
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

    for _ in range(10):
        og.sim.step()


def task_completion_check(env):
    target_object = env.scene.object_registry("name", "target_object")
    fireplace = env.scene.object_registry("name", "fireplace")
    robot = env.robots[0]
    # Get positions
    target_pos, _ = target_object.get_position_orientation()
    fireplace_pos, _ = fireplace.get_position_orientation()

    # Calculate xy distance only (ignore z)
    distance_xy = th.norm((target_pos[:2] - fireplace_pos[:2])).item()

    # Tolerance: log should be close to fireplace horizontally
    tolerance_xy = 0.25  # 25cm horizontal tolerance

    log_within_tolerance = distance_xy < tolerance_xy
    gripper_open = robot.is_grasping(candidate_obj=target_object).value == IsGraspingState.FALSE
    gripper_far = gripper_far_from_object(robot, target_object)

    return log_within_tolerance and gripper_open and gripper_far


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
            "load_object_categories": [
                "breakfast_table", "coffee_table", "straight_chair", "swivel_chair", "bed",
                "bottom_cabinet", "top_cabinet", "bookcase", "countertop",
                "dishwasher", "fridge", "oven", "microwave", "furniture_sink",
                "pedestal_sink", "shower_stall", "toilet", "standing_tv", "loudspeaker",
                "pot_plant", "mirror", "floor_lamp", "table_lamp",
                "towel_rack", "public_trash_can",
                "electric_switch", "openable_window",
            ],
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
