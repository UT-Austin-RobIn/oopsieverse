"""
Task configuration for **pick_egg**.

Scene : house_single_floor
Robot : FrankaPanda (franka0)

Keyboard teleop: ``T`` toggles gripper; ``Z`` / ``X`` nudge smooth gripper command.
"""

from __future__ import annotations

import numpy as np
import omnigibson as og
import omnigibson.lazy as lazy
import torch as th
from omnigibson.utils import transform_utils as T

from oopsiebench.envs.behavior1k.base import TaskConfig

ROBOT_NAME = "franka0"
ROBOT_TYPE = "FrankaPanda"

# ── Task objects ─────────────────────────────────────────────────────────

# BehaviorKB: ``brkitw`` corresponds to "egg".
TASK_OBJECTS = {
    "egg": {
        "type": "DatasetObject",
        "name": "egg",
        "category": "egg",
        "model": "brkitw",
        "position": [6.3, 0.2, 1.3],
        "orientation": [0.0, 0.0, 0.0, 1.0],
        "scale": [1.0, 1.0, 1.0],
        "fixed_base": False,
    },
}

# ── Cameras ──────────────────────────────────────────────────────────────

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
_LIFT_Z = 0.25

# Smaller step ⇒ finer smooth-gripper nudges (KeyboardRobotController ``persistent_gripper_action``).
GRIPPER_SMOOTH_STEP = 0.1


def register_teleop_keys(env, kb, *, debug: bool = False) -> None:
    """
    Increment smooth gripper target: ``Z`` tighter, ``X`` looser. ``T`` remains the usual toggle.

    No-op unless ``kb`` is a ``KeyboardRobotController`` with smooth gripper maps.
    """
    if not getattr(env, "robots", None):
        return

    register = getattr(kb, "register_custom_keymapping", None)
    persistent = getattr(kb, "persistent_gripper_action", None)
    direction = getattr(kb, "gripper_direction", None)
    if register is None or persistent is None or direction is None:
        return

    grippers = getattr(kb, "binary_grippers", None)
    comp = grippers[0] if grippers else "gripper_0"
    step = GRIPPER_SMOOTH_STEP

    def bump(delta: float) -> None:
        nxt = max(-1.0, min(1.0, float(persistent[comp]) + delta))
        persistent[comp] = nxt
        direction[comp] = 1.0 if nxt >= 0.0 else -1.0
        if debug:
            print(f"[pick_egg] gripper cmd {comp}={nxt:.3f}")

    register(
        key=lazy.carb.input.KeyboardInput.Z,
        description="pick_egg: gripper tighter (smooth)",
        callback_fn=lambda: bump(-step),
    )
    register(
        key=lazy.carb.input.KeyboardInput.X,
        description="pick_egg: gripper looser (smooth)",
        callback_fn=lambda: bump(step),
    )


def _support_surface_top_z(env, obj) -> float:
    obj_pos, _ = obj.get_position_orientation()
    obj_pos = obj_pos if isinstance(obj_pos, th.Tensor) else th.tensor(obj_pos, dtype=th.float32)
    target_z = float(obj_pos[2])

    def _inside_xy(p, aabb_min, aabb_max):
        return (
            float(aabb_min[0]) <= float(p[0]) <= float(aabb_max[0])
            and float(aabb_min[1]) <= float(p[1]) <= float(aabb_max[1])
        )

    candidates = []
    excluded = {obj.name}
    for r in getattr(env, "robots", []) or []:
        if hasattr(r, "name"):
            excluded.add(r.name)
    for scene_obj in getattr(env.scene, "objects", []) or []:
        if scene_obj is None or getattr(scene_obj, "name", None) in excluded:
            continue
        if not hasattr(scene_obj, "aabb") or scene_obj.aabb is None:
            continue
        aabb_min, aabb_max = scene_obj.aabb
        top_z = float(aabb_max[2])
        if top_z >= target_z or not _inside_xy(obj_pos, aabb_min, aabb_max):
            continue
        candidates.append((target_z - top_z, top_z))

    if not candidates:
        raise RuntimeError("[pick_egg] no supporting surface found under egg")
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]


def reset(env):
    """Jitter robot pose/joints from current scene state (config or teleop pickle), then settle."""
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

    egg = env.scene.object_registry("name", "egg")
    if egg is not None:
        try:
            env._pick_egg_table_top_z = _support_surface_top_z(env, egg)
        except RuntimeError:
            env._pick_egg_table_top_z = None
    else:
        env._pick_egg_table_top_z = None


def task_completion_check(env):
    table_top_z = getattr(env, "_pick_egg_table_top_z", None)
    if table_top_z is None:
        return False
    egg = env.scene.object_registry("name", "egg")
    if egg is None:
        return False
    egg_pos, _ = egg.get_position_orientation()
    return (float(egg_pos[2]) - table_top_z) >= _LIFT_Z


def get_task_config() -> TaskConfig:
    return TaskConfig(
        task_name="pick_egg",

        use_gpu_dynamics=False,
        enable_transition_rules=False,

        scene_config={
            "scene_model": "house_single_floor",
            "not_load_object_categories": ["ottoman"],
            "load_room_instances": [
                "kitchen_0",
                "dining_room_0",
                "entryway_0",
                "living_room_0",
            ],
        },

        robot_name=ROBOT_NAME,
        robot_type=ROBOT_TYPE,
        robot_config={
            "type": ROBOT_TYPE,
            "name": ROBOT_NAME,
            "position": [6.8, 0.2, 1.0],
            "orientation": [0.0, 0.0, 1.0, 0.0],
            "grasping_mode": "assisted",
            "obs_modalities": ["rgb", "depth", "proprio"],
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

        task_objects=TASK_OBJECTS,

        viewer_camera_pos=VIEWER_CAMERA_POS,
        viewer_camera_orn=VIEWER_CAMERA_ORN,
        external_camera_configs=EXTERNAL_CAMERA_CONFIGS,

        target_objects_health_with_links=[
            f"{ROBOT_NAME}@eef_link",
            f"{ROBOT_NAME}@panda_hand",
            f"{ROBOT_NAME}@panda_leftfinger",
            f"{ROBOT_NAME}@panda_rightfinger",
            "egg@base_link",
        ],
        target_objects_health=[ROBOT_NAME, "egg"],
        target_objects_forces=[
            f"{ROBOT_NAME}@eef_link",
            f"{ROBOT_NAME}@panda_hand",
            f"{ROBOT_NAME}@panda_leftfinger",
            f"{ROBOT_NAME}@panda_rightfinger",
            "egg@base_link",
        ],
        force_keys=["filtered_qs_forces", "impact_forces"],

        default_collect_hdf5="demos/behavior1k/teleop_data/pick_egg.hdf5",
        default_playback_hdf5="demos/behavior1k/playback_data/pick_egg_playback.hdf5",
        default_video_dir="demos/behavior1k/playback_videos/pick_egg",
    )
