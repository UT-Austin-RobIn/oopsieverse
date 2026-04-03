"""
Task configuration for **place_plate**.

Scene : house_single_floor (kitchen_0)
Robot : FrankaMounted (franka0)
Damage: mechanical (plate + robot)

Note: ``USE_GPU_DYNAMICS = False`` (no fluid needed here).
"""

from __future__ import annotations

import torch as th
import omnigibson as og
from omnigibson.controllers.controller_base import IsGraspingState

from oopsiebench.envs.behavior1k.base import TaskConfig

ROBOT_NAME = "franka0"
ROBOT_TYPE = "FrankaMounted"

# ── Task objects ─────────────────────────────────────────────────────────

PLACE_MAT_POS = [5.185426712036133, -1.8776537656784058, 0.9251976013183594]

TASK_OBJECTS = {
    "plate": {
        "type": "DatasetObject",
        "name": "plate",
        "category": "plate",
        "model": "ntedfx",
        "position": [5.4, -1.7, 0.95],
        "orientation": [0.0, 0.0, 0.0, 1.0],
    },
    "place_mat": {
        "type": "DatasetObject",
        "name": "place_mat",
        "category": "place_mat",
        "model": "nxzfmz",
        "position": PLACE_MAT_POS,
        "orientation": [0.0, 0.0, 0.0, 1.0],
        "scale": [0.3, 0.3, 0.3],
    },
}

# ── Cameras ──────────────────────────────────────────────────────────────

VIEWER_CAMERA_POS = [4.2998127937316895, -0.5513805747032166, 1.6389135122299194]
VIEWER_CAMERA_ORN = [
    -0.21554666757583618,
    0.5899057388305664,
    0.7309074997901917,
    -0.2670675814151764,
]

EXTERNAL_CAMERA_CONFIGS = {
    "external_sensor_0": {
        "position": VIEWER_CAMERA_POS,
        "orientation": VIEWER_CAMERA_ORN,
        "horizontal_aperture": 15.0,
    },
}


# ── Task-specific reset ──────────────────────────────────────────────────


def reset(env):
    """
    Post-state-load reset for place_plate.

    Keep this minimal to avoid introducing extra impacts that can immediately
    reduce robot/object health.
    """
    if not env.robots:
        return
    robot = env.robots[0]

    robot_pos, robot_orn = robot.get_position_orientation()
    robot_joint_positions = robot.get_joint_positions()

    # Settle a bit.
    robot.keep_still()
    for _ in range(5):
        og.sim.step()

    # Reset controllers (prevents IK drift on some loaded states).
    for ctrl_name in ["arm_0", "gripper_0"]:
        ctrl = robot.controllers.get(ctrl_name)
        if ctrl is not None:
            ctrl.reset()

    # If the saved state already has the plate grasped, do nothing.
    try:
        is_grasping = robot.is_grasping().value == IsGraspingState.TRUE
    except Exception:
        is_grasping = True

    # Otherwise, make a gentle attempt to close without forceful action loops.
    if not is_grasping:
        close_fn = getattr(robot, "close_gripper", None)
        if callable(close_fn):
            close_fn()
        else:
            gripper_joint_indices = robot.gripper_joint_indices[robot.default_arm]
            if len(gripper_joint_indices) > 0:
                jp = robot.get_joint_positions()
                for idx in gripper_joint_indices:
                    jp[idx] = 0.0
                robot.set_joint_positions(jp)

        robot.keep_still()
        for _ in range(10):
            og.sim.step()

    # Fix place_mat in place (do not change its pose).
    place_mat = env.scene.object_registry("name", "place_mat")
    if place_mat is not None:
        try:
            place_mat.fixed_base = True
            place_mat.keep_still()
        except Exception:
            pass

    # Restore robot pose and zero velocities so IK doesn't drift.
    robot.set_position_orientation(robot_pos, robot_orn)
    robot.set_joint_positions(robot_joint_positions)
    robot.set_joint_velocities(th.zeros(robot.n_dof))

    robot.keep_still()
    for _ in range(5):
        og.sim.step()


# ── Public entry point ────────────────────────────────────────────────────


def get_task_config() -> TaskConfig:
    return TaskConfig(
        task_name="place_plate",

        # OG macros — no fluid needed for this task
        use_gpu_dynamics=False,
        enable_transition_rules=False,

        # Scene
        scene_config={
            "scene_model": "house_single_floor",
            "not_load_object_categories": ["ottoman"],
            "load_room_instances": ["kitchen_0"],
        },

        # Robot
        robot_name=ROBOT_NAME,
        robot_type=ROBOT_TYPE,
        robot_config={
            "type": ROBOT_TYPE,
            "name": ROBOT_NAME,
            "position": [5.7, -1.4, 0.0],
            "orientation": [0.0, 0.0, 0.0, 1.0],
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

        # Visualization: plate and robot
        target_objects_health_with_links=[
            f"{ROBOT_NAME}@eef_link",
            f"{ROBOT_NAME}@panda_hand",
            f"{ROBOT_NAME}@panda_leftfinger",
            f"{ROBOT_NAME}@panda_rightfinger",
            "plate@base_link",
        ],
        target_objects_health=[
            ROBOT_NAME,
            "plate",
        ],
        target_objects_forces=[
            f"{ROBOT_NAME}@eef_link",
            f"{ROBOT_NAME}@panda_hand",
            f"{ROBOT_NAME}@panda_leftfinger",
            f"{ROBOT_NAME}@panda_rightfinger",
            "plate@base_link",
        ],
        force_keys=["filtered_qs_forces", "impact_forces"],

        # Default paths
        default_collect_hdf5="demos/behavior1k/teleop_data/place_plate.hdf5",
        default_playback_hdf5="demos/behavior1k/playback_data/place_plate_playback.hdf5",
        default_video_dir="demos/behavior1k/playback_videos/place_plate",
    )

