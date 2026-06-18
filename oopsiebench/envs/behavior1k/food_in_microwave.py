"""
Task configuration for **food_in_microwave**.

Scene : house_single_floor (same setup as pick_egg / open_single_door)
Robot : FrankaPanda (franka0)
Damage: mechanical + thermal
"""

from __future__ import annotations

import pickle

import numpy as np
import omnigibson as og
import omnigibson.lazy as lazy
import torch as th
from omnigibson import object_states
from omnigibson.utils import transform_utils as T

from oopsiebench.envs.behavior1k.base import TaskConfig
from oopsiebench.envs.behavior1k.spatial_checks import gripper_far_from_object

ROBOT_NAME = "franka0"
ROBOT_TYPE = "FrankaPanda"

# INIT_STATE_PATH = "oopsiebench/envs/behavior1k/init_states/food_in_microwave.pkl"
INIT_STATE_PATH = None

# Applied in reset() — not via link_physics_materials in TASK_OBJECTS, which mutates
# init_info and breaks HDF5 scene JSON serialization (PhysicsMaterial is not JSON-safe).
_CUPCAKE_FRICTION = {
    "static_friction": 0.2,
    "dynamic_friction": 0.15,
    "restitution": 0.0,
}

# ── Task objects ─────────────────────────────────────────────────────────
TASK_OBJECTS = {
    "microwave": {
        "type": "DatasetObject",
        "name": "microwave",
        "category": "microwave",
        "model": "ihxrvr",
        "position": [6.0412, 0.25, 1.3],
        "orientation": [0, 0, 0, 1],
        "fixed_base": True,
    },
    "bowl": {
        "type": "DatasetObject",
        "name": "bowl",
        "category": "bowl",
        "model": "oyidja",
        "position": [6.7412, -0.05, 1.3],
        "orientation": [0, 0, 0, 1],
    },
    "cupcake": {
        "type": "DatasetObject",
        "name": "cupcake",
        "category": "cupcake",
        "model": "outske",
        "position": [6.7412, -0.05, 1.35],
        "orientation": [0, 0, 0, 1],
    }
}

# ── Cameras ──────────────────────────────────────────────────────────────

VIEWER_CAMERA_POS = [7.090158462524414, 0.8846141695976257, 1.6505851745605469]
VIEWER_CAMERA_ORN = [0.17841866612434387, 0.4940887987613678, 0.8002775311470032, 0.28913477063179016]

EXTERNAL_CAMERA_CONFIGS = {
    "external_sensor_0": {
        "position": [7.090158462524414, 0.8846141695976257, 1.6505851745605469],
        "orientation": [0.17841866612434387, 0.4940887987613678, 0.8002775311470032, 0.28913477063179016],
        "horizontal_aperture": 15.0,
    },
    "external_sensor_1": {
        "position": [7.3764, 1.1205, 2.0117],
        "orientation": [0.2131, 0.4377, 0.7853, 0.3824],
        "horizontal_aperture": 15.0,
    },
}

_U_XY = 0.03
_U_YAW = 0.12
_U_ARM = 0.07

_U_BOWL_XY = 0.1
_U_BOWL_YAW = 0.1
_U_CUPCAKE_XY = 0.01
_CUPCAKE_Z_OFFSET = 0.05
_PLACEMENT_SETTLE_STEPS = 20
_MAX_PLACEMENT_RETRIES = 10

_BOWL_NOMINAL_POS = [-0.8, -2.3, 0.05]
_BOWL_NOMINAL_ORN = [0.0, 0.0, 0.0, 1.0]


def _apply_link_physics_material(obj, link_name: str, mat_name: str, **material_kwargs):
    """Override collision friction for one link (runtime only; safe for teleop HDF5 save)."""
    physics_mat = lazy.isaacsim.core.api.materials.physics_material.PhysicsMaterial(
        prim_path=f"{obj.prim_path}/Looks/{mat_name}",
        name=mat_name,
        **material_kwargs,
    )
    for msh in obj.links[link_name].collision_meshes.values():
        msh.apply_physics_material(physics_mat)


def _upright_orn(orn):
    """Level orientation (zero roll/pitch) preserving only yaw."""
    euler = T.quat2euler(orn).clone()
    euler[0] = 0.0
    euler[1] = 0.0
    return T.euler2quat(euler)


def _place_bowl_and_cupcake(bowl, cupcake, bowl_pos, bowl_orn, *, jitter_cupcake_xy=True):
    cupcake_pos = bowl_pos.clone()
    if jitter_cupcake_xy:
        cupcake_pos[0] += float(np.random.uniform(-_U_CUPCAKE_XY, _U_CUPCAKE_XY))
        cupcake_pos[1] += float(np.random.uniform(-_U_CUPCAKE_XY, _U_CUPCAKE_XY))
    cupcake_pos[2] = bowl_pos[2] + _CUPCAKE_Z_OFFSET
    bowl.set_position_orientation(bowl_pos, bowl_orn)
    # Keep the cupcake upright regardless of the bowl's settled tilt.
    cupcake.set_position_orientation(cupcake_pos, _upright_orn(bowl_orn))
    bowl.keep_still()
    cupcake.keep_still()


def _randomize_bowl_and_cupcake(bowl, cupcake):
    bowl_pos, bowl_orn = bowl.get_position_orientation()
    bowl_pos, bowl_orn = bowl_pos.clone(), bowl_orn.clone()

    for _ in range(_MAX_PLACEMENT_RETRIES):
        new_pos = bowl_pos.clone()
        new_pos[0] += float(np.random.uniform(-_U_BOWL_XY, _U_BOWL_XY))
        new_pos[1] += float(np.random.uniform(-_U_BOWL_XY, _U_BOWL_XY))
        euler = T.quat2euler(bowl_orn).clone()
        euler[2] += float(np.random.uniform(-_U_BOWL_YAW, _U_BOWL_YAW))
        _place_bowl_and_cupcake(bowl, cupcake, new_pos, T.euler2quat(euler))
        for _ in range(_PLACEMENT_SETTLE_STEPS):
            og.sim.step()
        if cupcake.states[object_states.OnTop].get_value(bowl):
            return

    fallback_pos = th.tensor(_BOWL_NOMINAL_POS, dtype=bowl_pos.dtype, device=bowl_pos.device)
    fallback_orn = th.tensor(_BOWL_NOMINAL_ORN, dtype=bowl_orn.dtype, device=bowl_orn.device)
    _place_bowl_and_cupcake(bowl, cupcake, fallback_pos, fallback_orn, jitter_cupcake_xy=False)
    for _ in range(_PLACEMENT_SETTLE_STEPS):
        og.sim.step()
    print("[food_in_microwave] bowl/cupcake placement fell back to nominal pose")


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
        raise RuntimeError("[food_in_microwave] no supporting surface found under microwave")
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
    """Seat + open the microwave on the counter, place bowl/cupcake, jitter robot."""

    # Load initial state
    if INIT_STATE_PATH is not None:
        with open(INIT_STATE_PATH, "rb") as f: state_flat_array = pickle.load(f)
        og.sim.load_state(state_flat_array, serialized=True)

    # Seat the (fixed-base) microwave onto the counter, then open its door.
    microwave = env.scene.object_registry("name", "microwave")
    if microwave is not None:
        try:
            _seat_on_counter(env, microwave)
        except RuntimeError as e:
            print(e)
        if hasattr(microwave, "keep_still"):
            microwave.keep_still()
        microwave.joints["j_leaf"].set_pos(0.9, normalized=True)

    cupcake = env.scene.object_registry("name", "cupcake")
    if cupcake is not None and not getattr(env, "_cupcake_low_friction_applied", False):
        _apply_link_physics_material(
            cupcake, "base_link", "cupcake_base_physics_mat", **_CUPCAKE_FRICTION
        )
        env._cupcake_low_friction_applied = True

    bowl = env.scene.object_registry("name", "bowl")
    if bowl is not None and cupcake is not None:
        _randomize_bowl_and_cupcake(bowl, cupcake)

    if not getattr(env, "robots", None):
        return
    robot = env.robots[0]
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

    # Leave the cupcake upright on the bowl as the final reset state.
    if cupcake is not None:
        c_pos, c_orn = cupcake.get_position_orientation()
        cupcake.set_position_orientation(c_pos, _upright_orn(c_orn))
        cupcake.keep_still()


def task_completion_check(env):
    microwave = env.scene.object_registry("name", "microwave")
    cupcake = env.scene.object_registry("name", "cupcake")
    bowl = env.scene.object_registry("name", "bowl")
    robot = env.robots[0]
    microwave_open = microwave.states[object_states.Open].get_value()
    bowl_inside = bowl.states[object_states.Inside].get_value(other=microwave)
    cupcake_inside = cupcake.states[object_states.Inside].get_value(other=microwave)
    cupcake_on_top_bowl = cupcake.states[object_states.OnTop].get_value(other=bowl)
    layout_ok = not microwave_open and cupcake_inside and bowl_inside and cupcake_on_top_bowl
    return layout_ok and gripper_far_from_object(robot, microwave, threshold=0.5)


def get_task_config() -> TaskConfig:
    return TaskConfig(
        task_name="food_in_microwave",

        # OG macros
        use_gpu_dynamics=False,
        enable_transition_rules=False,

        # Scene
        scene_config={
            "scene_model": "house_single_floor",
            # Drop the scene's own background microwave (we spawn our own task one).
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
            # Same yaw as open_single_door; the whole food layout is translated
            # rigidly (here shifted +x to bring the microwave back onto the counter),
            # preserving all relative offsets.
            "position": [6.9412, 0.65, 1.0],
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
                    "isaac_kp": 1e6,
                },
            },
        },

        # Objects
        task_objects=TASK_OBJECTS,

        # Cameras
        viewer_camera_pos=VIEWER_CAMERA_POS,
        viewer_camera_orn=VIEWER_CAMERA_ORN,
        external_camera_configs=EXTERNAL_CAMERA_CONFIGS,
        exclude_sensor_names=["eef_link"],

        # Visualization
        target_objects_health_with_links=[
            f"{ROBOT_NAME}@eef_link",
            f"{ROBOT_NAME}@panda_hand",
            f"{ROBOT_NAME}@panda_leftfinger",
            f"{ROBOT_NAME}@panda_rightfinger",
            f"microwave@base_link",
            f"microwave@link_0",
            f"microwave@glass",
            f"bowl@base_link",
            f"cupcake@base_link",
        ],
        target_objects_health=[ROBOT_NAME, "cupcake", "bowl", "microwave"],
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
            f"microwave@link_0",
            f"microwave@glass",
            f"bowl@base_link",
            f"cupcake@base_link",
        ],
        # force_keys=["impact_forces"],
        force_keys=["filtered_qs_forces"],

        # Default paths
        default_collect_hdf5="demos/behavior1k/teleop_data/food_in_microwave.hdf5",
        default_playback_hdf5="demos/behavior1k/playback_data/food_in_microwave_playback.hdf5",
        default_video_dir="demos/behavior1k/playback_videos/food_in_microwave",
    )
