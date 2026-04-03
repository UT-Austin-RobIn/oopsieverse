"""
Task configuration for **pick_egg**.

Scene matches the simple household tasks (e.g. `pour_water`, `wipe_counter`),
but the interactive objects are intentionally minimal:
- One Franka robot (`franka0`)
- One egg on the table at the same pose used for the laptop/sponge in other tasks

No dirt, no water, no extra particles.
"""

from __future__ import annotations

import torch as th

import omnigibson as og
import omnigibson.lazy as lazy

from oopsiebench.envs.behavior1k.base import TaskConfig

ROBOT_NAME = "franka0"
ROBOT_TYPE = "FrankaPanda"

# BehaviorKB: `brkitw` corresponds to "egg".
TASK_OBJECTS = {
    "egg": {
        "type": "DatasetObject",
        "name": "egg",
        "category": "egg",
        "model": "brkitw",
        # Place where `laptop` / `sponge` are positioned in other tasks.
        "position": [6.3, 0.2, 1.3],
        "orientation": [0.0, 0.0, 0.0, 1.0],
        "scale": [1.0, 1.0, 1.0],
        # Must be movable so the robot can actually lift it.
        # We'll still snap it onto the counter-top in `reset()`.
        "fixed_base": False,
    },
}

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

# Step size for Z / X incremental gripper (must match teleop `register_pick_egg_incremental_gripper_keys`).
PICK_EGG_GRIPPER_INCREMENT = 0.06


def _pick_egg_gripper_component(action_generator):
    comps = getattr(action_generator, "binary_grippers", None) or []
    if len(comps) >= 1:
        return comps[0]
    return "gripper_0"


def persistent_value_from_close_level(close_level: float) -> float:
    """Map 0=fully open, 1=fully closed to KeyboardRobotController gripper command (-1..1)."""
    lvl = max(0.0, min(1.0, float(close_level)))
    return 1.0 - 2.0 * lvl


def sync_pick_egg_close_level_from_persistent(env, action_generator) -> None:
    """Keep `_pick_egg_gripper_close_level` aligned after binary `T` toggles."""
    if not getattr(env, "_pick_egg_incremental_gripper_registered", False):
        return
    comp = _pick_egg_gripper_component(action_generator)
    persistent = float(action_generator.persistent_gripper_action[comp])
    env._pick_egg_gripper_close_level = max(0.0, min(1.0, (1.0 - persistent) / 2.0))


def sync_teleop_gripper_after_env_reset(env, action_generator) -> None:
    """After pickle load + `reset()`, align smooth gripper command with physically open fingers."""
    if not getattr(env, "_pick_egg_incremental_gripper_registered", False):
        return
    comp = _pick_egg_gripper_component(action_generator)
    env._pick_egg_gripper_close_level = 0.0
    action_generator.persistent_gripper_action[comp] = 1.0
    action_generator.gripper_direction[comp] = 1.0


def register_pick_egg_incremental_gripper_keys(env, action_generator, *, debug: bool = False) -> None:
    """
    Z / X adjust gripper open-ness via smooth gripper command (press + repeat from carb).
    Stock `T` remains the binary open/close toggle (MultiFingerGripperController smooth, dim=1).

    Call from teleop after `KeyboardRobotController` is constructed.
    """
    if not getattr(env, "robots", None):
        if debug:
            print("[pick_egg] register incremental gripper: no robots")
        return
    robot = env.robots[0]
    env._pick_egg_incremental_gripper_registered = True
    comp = _pick_egg_gripper_component(action_generator)
    step = PICK_EGG_GRIPPER_INCREMENT

    def _bump(delta: float) -> None:
        if not getattr(env, "robots", None) or not env.robots or env.robots[0] is not robot:
            return
        if getattr(env, "scene", None) is None or env.scene.object_registry("name", "egg") is None:
            return
        lvl = float(getattr(env, "_pick_egg_gripper_close_level", 0.0))
        lvl = max(0.0, min(1.0, lvl + delta))
        env._pick_egg_gripper_close_level = lvl
        persistent = persistent_value_from_close_level(lvl)
        action_generator.persistent_gripper_action[comp] = persistent
        action_generator.gripper_direction[comp] = 1.0 if persistent >= 0.0 else -1.0
        if debug:
            print(
                f"[pick_egg] gripper inc: close_level={lvl:.3f} persistent={persistent:.3f} "
                f"comp={comp!r}"
            )

    action_generator.register_custom_keymapping(
        key=lazy.carb.input.KeyboardInput.Z,
        description="pick_egg: incremental gripper open",
        callback_fn=lambda: _bump(-step),
    )
    action_generator.register_custom_keymapping(
        key=lazy.carb.input.KeyboardInput.X,
        description="pick_egg: incremental gripper close",
        callback_fn=lambda: _bump(step),
    )
    if debug:
        print(
            f"[pick_egg] Z=open increment, X=close increment, T=binary toggle; "
            f"binary_grippers={getattr(action_generator, 'binary_grippers', None)!r}"
        )


def reset(env):
    """
    Predictable start: gripper open, egg snapped to counter. Z/X/T gripper is wired in teleop
    via `register_pick_egg_incremental_gripper_keys`.
    """
    if not getattr(env, "robots", None):
        return
    robot = env.robots[0]

    if not (
        hasattr(robot, "gripper_control_idx")
        and hasattr(robot, "finger_joint_names")
        and hasattr(robot, "joints")
        and hasattr(robot, "default_arm")
    ):
        print("[pick_egg] warning: missing robot gripper fields; cannot open fingers on reset")
    else:
        arm = robot.default_arm
        gripper_dof_idx = robot.gripper_control_idx[arm]
        finger_joint_names = robot.finger_joint_names[arm]

        open_qpos = []
        for jn in finger_joint_names:
            joint = robot.joints[jn]
            lower = float(joint.lower_limit)
            upper = float(joint.upper_limit)
            open_qpos.append(max(lower, upper))

        open_qpos_t = th.tensor(open_qpos, dtype=th.float32)
        robot.set_joint_positions(open_qpos_t, indices=gripper_dof_idx, drive=False)
        if hasattr(robot, "set_joint_velocities"):
            robot.set_joint_velocities(th.zeros_like(open_qpos_t), indices=gripper_dof_idx, drive=False)

        env._pick_egg_gripper_close_level = 0.0

    # Lower the egg so it sits on the closest supporting surface under its (x, y).
    egg = env.scene.object_registry("name", "egg") if getattr(env, "scene", None) is not None else None
    if egg is None:
        print("[pick_egg] warning: egg not found in scene; skipping egg placement")
        for _ in range(5):
            og.sim.step()
        return

    egg_pos, egg_orn = egg.get_position_orientation()
    egg_pos_t = egg_pos if isinstance(egg_pos, th.Tensor) else th.tensor(egg_pos, dtype=th.float32)

    # Compute egg min-z from AABB (most robust way to correct "floating" assets).
    if not hasattr(egg, "aabb") or egg.aabb is None:
        print("[pick_egg] warning: egg has no aabb; cannot snap to counter")
        for _ in range(5):
            og.sim.step()
        return
    egg_aabb_min, egg_aabb_max = egg.aabb
    egg_min_z = float(egg_aabb_min[2])

    egg_xy = egg_pos_t[:2]

    def _xy_inside(p_xy: th.Tensor, aabb_min, aabb_max) -> bool:
        return float(aabb_min[0]) <= float(p_xy[0]) <= float(aabb_max[0]) and float(aabb_min[1]) <= float(
            p_xy[1]
        ) <= float(aabb_max[1])

    target_surface = None
    target_top_z = -1e9
    excluded_names = {"egg"}
    if hasattr(robot, "name"):
        excluded_names.add(robot.name)

    for obj in getattr(env.scene, "objects", []) or []:
        if obj is None:
            continue
        name = getattr(obj, "name", None)
        if name in excluded_names:
            continue
        if not hasattr(obj, "aabb") or obj.aabb is None:
            continue
        aabb_min, aabb_max = obj.aabb

        # Candidate support surfaces must be below the egg (by their top-z).
        top_z = float(aabb_max[2])
        if top_z >= float(egg_pos_t[2]):
            continue
        # And must cover the egg's (x, y).
        if not _xy_inside(egg_xy, aabb_min, aabb_max):
            continue

        # Pick the closest surface below the egg.
        if top_z > target_top_z:
            target_top_z = top_z
            target_surface = obj

    if target_surface is None:
        print("[pick_egg] warning: no supporting surface found under egg; leaving z as-is")
    else:
        clearance = 0.005
        desired_egg_min_z = target_top_z + clearance
        dz = desired_egg_min_z - egg_min_z

        new_pos = [float(egg_pos_t[0]), float(egg_pos_t[1]), float(egg_pos_t[2]) + dz]
        egg.set_position_orientation(new_pos, egg_orn)
        if hasattr(egg, "keep_still"):
            egg.keep_still()
        print(
            f"[pick_egg] snapped egg: surface={target_surface.name} surface_top_z={target_top_z:.4f} "
            f"egg_min_z={egg_min_z:.4f} dz={dz:.4f}"
        )

    # Let the sim settle briefly.
    for _ in range(5):
        og.sim.step()


def get_task_config() -> TaskConfig:
    return TaskConfig(
        task_name="pick_egg",

        # OG macros
        use_gpu_dynamics=False,
        enable_transition_rules=False,

        # Scene: same as `pour_water` / `wipe_counter`
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

        # Robot
        robot_name=ROBOT_NAME,
        robot_type=ROBOT_TYPE,
        robot_config={
            "type": ROBOT_TYPE,
            "name": ROBOT_NAME,
            "position": [6.8, 0.2, 1.0],
            "orientation": [0.0, 0.0, 1.0, 0.0],
            "grasping_mode": "physical",
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
                    # `smooth` + command_dim 1 => stock `T` binary toggle in KeyboardRobotController.
                    # Z/X adjust the same smooth command via `persistent_gripper_action` (see teleop).
                    "mode": "smooth",
                    "motor_type": "position",
                    # Stronger squeeze than defaults (same gains as place_bowl reload_controllers).
                    "isaac_kp": 30000.0,
                    "isaac_kd": 15000.0,
                },
            },
        },

        # Objects
        task_objects=TASK_OBJECTS,

        # Cameras
        viewer_camera_pos=VIEWER_CAMERA_POS,
        viewer_camera_orn=VIEWER_CAMERA_ORN,
        external_camera_configs=EXTERNAL_CAMERA_CONFIGS,

        # Visualization: track robot + egg
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

        # Default paths
        default_collect_hdf5="demos/behavior1k/teleop_data/pick_egg.hdf5",
        default_playback_hdf5="demos/behavior1k/playback_data/pick_egg_playback.hdf5",
        default_video_dir="demos/behavior1k/playback_videos/pick_egg",
    )

