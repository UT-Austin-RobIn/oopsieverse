"""
Task configuration for **open_single_door**.

Scene : house_single_floor (same setup as pick_egg / wipe_counter / pour_water)
Robot : FrankaPanda (franka0)
Damage: mechanical + thermal
"""

from __future__ import annotations

import pickle

import numpy as np
import omnigibson as og
import torch as th
from omnigibson import object_states
from omnigibson.controllers.controller_base import IsGraspingState
from omnigibson.utils import transform_utils as T

from oopsiebench.envs.behavior1k.base import TaskConfig
from oopsiebench.envs.behavior1k.spatial_checks import gripper_far_from_object

ROBOT_NAME = "franka0"
ROBOT_TYPE = "FrankaPanda"

# INIT_STATE_PATH = "resources/init_states/open_single_door.pkl"
INIT_STATE_PATH = None

# ── Task objects ────────────────────────────────────────────────────────

TASK_OBJECTS = {
    "microwave": {
        "type": "DatasetObject",
        "name": "microwave",
        "category": "microwave",
        "model": "ihxrvr",
        # In front of the robot, where the egg / laptop spawn in the house tasks.
        # reset() seats it down onto the counter beneath this xy.
        # (shifted −y together with the robot to keep it on the counter / island.)
        "position": [6.0, 0.05, 1.3],
        "orientation": [0, 0, 0, 1],
        "fixed_base": True,
    }
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

_U_XY = 0.03
_U_YAW = 0.12
_U_ARM = 0.07


def _support_surface_top_z(env, obj) -> float:
    """Top-z of the nearest surface directly beneath obj's xy (e.g. the counter)."""
    obj_pos, _ = obj.get_position_orientation()
    obj_pos = obj_pos if isinstance(obj_pos, th.Tensor) else th.tensor(obj_pos, dtype=th.float32)
    target_z = float(obj_pos[2])

    def _inside_xy(p, amin, amax):
        return (float(amin[0]) <= float(p[0]) <= float(amax[0])
                and float(amin[1]) <= float(p[1]) <= float(amax[1]))

    excluded = {obj.name}
    for r in getattr(env, "robots", []) or []:
        if hasattr(r, "name"):
            excluded.add(r.name)

    candidates = []
    for scene_obj in getattr(env.scene, "objects", []) or []:
        if scene_obj is None or getattr(scene_obj, "name", None) in excluded:
            continue
        if not hasattr(scene_obj, "aabb") or scene_obj.aabb is None:
            continue
        amin, amax = scene_obj.aabb
        top_z = float(amax[2])
        if top_z >= target_z or not _inside_xy(obj_pos, amin, amax):
            continue
        candidates.append((target_z - top_z, top_z))

    if not candidates:
        raise RuntimeError("[open_single_door] no supporting surface found under microwave")
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]


def _seat_on_counter(env, obj):
    """Shift obj vertically so its AABB bottom rests on the surface beneath it."""
    top_z = _support_surface_top_z(env, obj)
    if object_states.AABB not in obj.states:
        return
    lower, _ = obj.states[object_states.AABB].get_value()
    pos, orn = obj.get_position_orientation()
    pos = pos.clone()
    pos[2] = pos[2] + (top_z - float(lower[2]))
    obj.set_position_orientation(pos, orn)


def reset(env):
    """Seat the microwave on the counter in front of the robot; jitter robot; settle."""

    # Load initial state
    if INIT_STATE_PATH is not None:
        with open(INIT_STATE_PATH, "rb") as f: state_flat_array = pickle.load(f)
        og.sim.load_state(state_flat_array, serialized=True)

    if not getattr(env, "robots", None):
        return
    robot = env.robots[0]

    # Drop the (fixed-base) microwave down onto the counter beneath its spawn xy.
    microwave = env.scene.object_registry("name", "microwave")
    if microwave is not None:
        try:
            _seat_on_counter(env, microwave)
        except RuntimeError as e:
            print(e)
        if hasattr(microwave, "keep_still"):
            microwave.keep_still()

    # Reset the robot's joints to the tuned reach pose, then jitter base + arm.
    RESET_JOINT_POSITIONS = th.tensor([0.0606, -1.7628, 1.5638, -2.4855, 0.1761, 2.4263, 2.1347, 0.0400, 0.0400])
    robot.set_joint_positions(RESET_JOINT_POSITIONS)

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
    microwave = env.scene.object_registry("name", "microwave")
    microwave_open = False
    joint = microwave.joints["j_leaf"]
    if (joint.get_state()[0] - joint.lower_limit) / (joint.upper_limit - joint.lower_limit) > 0.95:
        microwave_open = True

    # Check if gripper is not holding anything
    robot = env.robots[0]
    robot_grasping = robot.is_grasping(candidate_obj=microwave).value == IsGraspingState.TRUE

    gripper_far = gripper_far_from_object(robot, microwave, threshold=0.5)
    return microwave_open and not robot_grasping and gripper_far


def get_task_config() -> TaskConfig:
    return TaskConfig(
        task_name="open_single_door",

        # OG macros
        use_gpu_dynamics=False,
        enable_transition_rules=False,

        # Scene
        scene_config={
            "scene_model": "house_single_floor",
            # Drop the scene's own background microwave for this task — we spawn our
            # own "microwave" task object (which is added separately, so it stays).
            "not_load_object_categories": ["ottoman", "microwave"],
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
            # Same robot->microwave offset as the original Rs_int layout (microwave
            # ~0.69m forward / 0.60m to the robot's left), which the hardcoded reset
            # arm pose was tuned for. Both shifted −y together to keep the base on
            # the island (microwave is at [6.0, 0.05]).
            "position": [6.6912, 0.65, 1.0],
            "orientation": [0.0, 0.0, 0.9984, 0.0564],
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
            f"microwave@base_link",
            f"microwave@leaf",
            # f"microwave@glass",
        ],
        target_objects_health=[ROBOT_NAME, "microwave"],
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
            f"microwave@base_link",
            f"microwave@leaf",
            # f"microwave@glass",
        ],
        force_keys=["filtered_qs_forces"],

        # Default paths
        default_collect_hdf5="demos/behavior1k/teleop_data/open_single_door.hdf5",
        default_playback_hdf5="demos/behavior1k/playback_data/open_single_door_playback.hdf5",
        default_video_dir="demos/behavior1k/playback_videos/open_single_door",
    )
