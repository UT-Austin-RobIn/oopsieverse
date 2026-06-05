"""
Task configuration for **open_drawer**.

Scene : Rs_int (filtered categories). Robot : FrankaPanda (franka0).

``reset(env)`` applies small uniform pose/joint noise after teleop loads state from pickle.
"""

import pickle
import numpy as np
import torch as th
import omnigibson as og
from omnigibson.utils import transform_utils as T
from omnigibson.controllers.controller_base import IsGraspingState


from oopsiebench.envs.behavior1k.base import TaskConfig
from oopsiebench.envs.behavior1k.spatial_checks import gripper_far_from_object

ROBOT_NAME = "franka0"
ROBOT_TYPE = "FrankaPanda"

TASK_OBJECTS = {}

VIEWER_CAMERA_POS = [-0.2010, -2.7257,  1.0654]
VIEWER_CAMERA_ORN = [0.6820, -0.0016, -0.0017, 0.7314]

EXTERNAL_CAMERA_CONFIGS = {
        "external_sensor_0": {
        "position": VIEWER_CAMERA_POS,
        "orientation": VIEWER_CAMERA_ORN,
        "horizontal_aperture": 20.0,
    }
}

INIT_STATE_PATH = "resources/init_states/open_drawer.pkl"

def get_task_config() -> TaskConfig:
    return TaskConfig(
        task_name="open_drawer",

        # OG macros
        use_gpu_dynamics=False,
        enable_transition_rules=False,

        # Scene
        scene_config={
            "scene_model": "Rs_int",
            "load_object_categories": ["bottom_cabinet", "floors", "walls", "ceilings"],
        },

        # Robot
        robot_name=ROBOT_NAME,
        robot_type=ROBOT_TYPE,
        robot_config={
            "type": ROBOT_TYPE,
            "name": ROBOT_NAME,
            "position": [-1.2, 0.0, 0.0],
            "orientation": [0.0, 0.0, 0.0, 1.0],
            "grasping_mode": "assisted",
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
        external_camera_configs=EXTERNAL_CAMERA_CONFIGS,
        viewer_camera_pos=VIEWER_CAMERA_POS,
        viewer_camera_orn=VIEWER_CAMERA_ORN,

        # Visualization
        target_objects_health_with_links=[
            f"{ROBOT_NAME}@eef_link",
            f"{ROBOT_NAME}@panda_hand",
            f"{ROBOT_NAME}@panda_leftfinger",
            f"{ROBOT_NAME}@panda_rightfinger",
            "bottom_cabinet@base_link",
            "bottom_cabinet@link_1",
            "bottom_cabinet@link_2",
            "bottom_cabinet@link_3",
            "bottom_cabinet@link_4",
        ],
        target_objects_health=[
            ROBOT_NAME,
            "bottom_cabinet",
        ],
        target_objects_forces=[
            f"{ROBOT_NAME}@eef_link",
            f"{ROBOT_NAME}@panda_hand",
            f"{ROBOT_NAME}@panda_leftfinger",
            f"{ROBOT_NAME}@panda_rightfinger",
            "bottom_cabinet@base_link",
            "bottom_cabinet@link_1",
            "bottom_cabinet@link_2",
            "bottom_cabinet@link_3",
            "bottom_cabinet@link_4",
        ],
        force_keys=["filtered_qs_forces", "impact_forces"],

        # Default paths
        default_collect_hdf5="demos/behavior1k/teleop_data/open_drawer.hdf5",
        default_playback_hdf5="demos/behavior1k/playback_data/open_drawer_playback.hdf5",
        default_video_dir="demos/behavior1k/playback_videos/open_drawer",
    )


_U_XY = 0.03
_U_YAW = 0.12
_U_ARM = 0.07


def reset(env):
    """Small uniform noise on base xyz, yaw, and arm joints; settle briefly."""
    
    # Load initial state
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
    # Check if any of the drawers are fully open
    drawer = env.scene.object_registry("name", "bottom_cabinet_bamfsz_1")
    drawer_joints = drawer.joints
    drawer_open = False
    for joint in drawer_joints.values():
        if (joint.get_state()[0] - joint.lower_limit) / (joint.upper_limit - joint.lower_limit) > 0.95:
            drawer_open = True
            break
    
    # Check if gripper is not holding anything
    robot = env.robots[0]
    robot_grasping = robot.is_grasping(candidate_obj=drawer).value == IsGraspingState.TRUE
    
    gripper_far = gripper_far_from_object(robot, drawer)
    return drawer_open and not robot_grasping and gripper_far
