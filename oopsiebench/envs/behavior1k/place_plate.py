"""
Task configuration for **place_plate**.

Scene : house_single_floor (kitchen_0)
Robot : FrankaMounted (franka0)
Damage: mechanical (plate + robot)
"""

import pickle

import numpy as np
import torch as th
import omnigibson as og
import omnigibson.lazy as lazy
from omnigibson import object_states
from omnigibson.controllers.controller_base import IsGraspingState
from omnigibson.utils import transform_utils as T

from oopsiebench.envs.behavior1k.base import TaskConfig
from oopsiebench.envs.behavior1k.spatial_checks import gripper_far_from_object

ROBOT_NAME = "franka0"
ROBOT_TYPE = "FrankaMounted"

INIT_STATE_PATH = "oopsiebench/envs/behavior1k/init_states/place_plate.pkl"

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
        "scale": [0.3, 0.3, 0.4],
        "fixed_base": True,
    },
}

# ── Cameras ──────────────────────────────────────────────────────────────

VIEWER_CAMERA_POS = [4.2998127937316895, -0.5513805747032166, 1.6389135122299194]
VIEWER_CAMERA_ORN = [-0.21554666757583618, 0.5899057388305664, 0.7309074997901917, -0.2670675814151764]

EXTERNAL_CAMERA_CONFIGS = {
    "external_sensor_0": {
        "position": VIEWER_CAMERA_POS,
        "orientation": VIEWER_CAMERA_ORN,
        "horizontal_aperture": 15.0,
    },
}

# ── Public entry point ───────────────────────────────────────────────────

def get_task_config() -> TaskConfig:
    return TaskConfig(
        task_name="place_plate",

        # OG macros
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
            "orientation": [0.0, 0.0, -0.7071067811865476, 0.7071067811865476],
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
            "plate@base_link",
        ],
        target_objects_health=[ROBOT_NAME, "plate"],
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


_U_XY = 0.05
_U_YAW = 0.12
_U_ARM = 0.2
_MAT_U_XY = 0.05
_MAT_U_YAW = 0.08
_MAX_RESET_TRIES = 20
_MIN_PLATE_CLEARANCE = 0.15

_LOW_BOUNCE_PHYSICS = {
    "static_friction": 0.8,
    "dynamic_friction": 0.7,
    "restitution": 0.0,
}


def _apply_low_bounce_material(obj, mat_name, **material_kwargs):
    """Low-restitution collision material on every link of *obj* (runtime only).

    Forces restitutionCombineMode='min' so the 0 wins instead of PhysX averaging it
    against a bouncy surface; frictionCombineMode='max' keeps friction high.
    """
    mat_prim_path = f"{obj.prim_path}/Looks/{mat_name}"
    physics_mat = lazy.isaacsim.core.api.materials.physics_material.PhysicsMaterial(
        prim_path=mat_prim_path, name=mat_name, **material_kwargs,
    )
    try:
        stage = lazy.isaacsim.core.utils.stage.get_current_stage()
        mat_prim = stage.GetPrimAtPath(mat_prim_path)
        physx_api = lazy.pxr.PhysxSchema.PhysxMaterialAPI.Apply(mat_prim)
        physx_api.CreateRestitutionCombineModeAttr().Set("min")
        physx_api.CreateFrictionCombineModeAttr().Set("max")
    except Exception as e:
        print(f"[place_plate] could not set restitution combine mode: {e}")
    for link in obj.links.values():
        for msh in link.collision_meshes.values():
            msh.apply_physics_material(physics_mat)


def _counter_top_z_under(env, xy_pos):
    """Top z of the AABB directly under ``xy_pos`` (excluding robot/plate/mat)."""
    excluded = {"place_mat", "plate"}
    for r in getattr(env, "robots", []) or []:
        if hasattr(r, "name"):
            excluded.add(r.name)
    target_z = float(xy_pos[2])
    best = None
    for obj in getattr(env.scene, "objects", []) or []:
        if obj is None or getattr(obj, "name", None) in excluded:
            continue
        if not hasattr(obj, "aabb") or obj.aabb is None:
            continue
        amin, amax = obj.aabb
        top_z = float(amax[2])
        if top_z > target_z + 0.05:
            continue
        if not (float(amin[0]) <= float(xy_pos[0]) <= float(amax[0])
                and float(amin[1]) <= float(xy_pos[1]) <= float(amax[1])):
            continue
        if best is None or top_z > best:
            best = top_z
    return best


def reset(env):
    """Settle robot, snap kinematic mat onto the counter with jitter, jitter robot pose."""
    if not env.robots:
        return
    robot = env.robots[0]

    if not getattr(env, "_low_bounce_applied", False):
        for _name in ("plate", "place_mat"):
            _obj = env.scene.object_registry("name", _name)
            if _obj is not None:
                _apply_low_bounce_material(_obj, f"{_name}_low_bounce_mat", **_LOW_BOUNCE_PHYSICS)
        env._low_bounce_applied = True

    for attempt in range(_MAX_RESET_TRIES):
        with open(INIT_STATE_PATH, "rb") as f:
            state_flat_array = pickle.load(f)
        og.sim.load_state(state_flat_array, serialized=True)
        for _ in range(5):
            og.sim.step()

        robot.keep_still()
        for _ in range(5):
            og.sim.step()

        robot_pos, robot_orn = robot.get_position_orientation()
        robot_joint_positions = robot.get_joint_positions()

        for ctrl_name in ("arm_0", "gripper_0"):
            ctrl = robot.controllers.get(ctrl_name)
            if ctrl is not None:
                ctrl.reset()

        # Close gripper unless already grasping (loaded states may already hold the plate).
        try:
            is_grasping = robot.is_grasping().value == IsGraspingState.TRUE
        except Exception:
            is_grasping = True
        if not is_grasping:
            try:
                close_action = th.zeros(robot.action_dim)
                close_action[robot.gripper_action_idx[robot.default_arm]] = -1.0
                robot.apply_action(close_action)
            except Exception:
                pass
            robot.keep_still()
            for _ in range(10):
                og.sim.step()

        # Jitter the kinematic mat and snap z to the counter below it.
        place_mat = env.scene.object_registry("name", "place_mat")
        if place_mat is not None:
            mpos = th.tensor(PLACE_MAT_POS, dtype=th.float32).clone()
            mpos[0] += float(np.random.uniform(-_MAT_U_XY, _MAT_U_XY))
            mpos[1] += float(np.random.uniform(-_MAT_U_XY, _MAT_U_XY))
            counter_top_z = _counter_top_z_under(env, mpos)
            if counter_top_z is not None:
                mat_aabb = getattr(place_mat, "aabb", None)
                half_h = 0.5 * float(mat_aabb[1][2] - mat_aabb[0][2]) if mat_aabb is not None else 0.0
                mpos[2] = counter_top_z + half_h + 0.001
            yaw = float(np.random.uniform(-_MAT_U_YAW, _MAT_U_YAW))
            morn = T.euler2quat(th.tensor([0.0, 0.0, yaw], dtype=th.float32))
            place_mat.set_position_orientation(mpos, morn)
            place_mat.keep_still()

        # Restore robot pose with light jitter.
        pos = robot_pos.clone()
        pos[0] += float(np.random.uniform(-_U_XY, _U_XY))
        pos[1] += float(np.random.uniform(-_U_XY, _U_XY))
        euler = T.quat2euler(robot_orn).clone()
        euler[2] = euler[2] + float(np.random.uniform(-_U_YAW, _U_YAW))
        q = robot_joint_positions.clone()
        for arm_name in robot.arm_control_idx:
            idx = robot.arm_control_idx[arm_name]
            u = (th.rand(len(idx), device=q.device, dtype=q.dtype) * 2 - 1) * _U_ARM
            q[idx] = q[idx] + u
        robot.set_position_orientation(pos, T.euler2quat(euler))
        robot.set_joint_positions(q)
        robot.set_joint_velocities(th.zeros(robot.n_dof))

        robot.keep_still()
        for _ in range(5):
            og.sim.step()

        plate = env.scene.object_registry("name", "plate")
        clearance = None
        if (plate is not None and place_mat is not None
                and getattr(plate, "aabb", None) is not None
                and getattr(place_mat, "aabb", None) is not None):
            clearance = float(plate.aabb[0][2]) - float(place_mat.aabb[1][2])

        clearance_str = "n/a" if clearance is None else f"{clearance:.3f}"
        print(f"[place_plate] reset attempt {attempt + 1}/{_MAX_RESET_TRIES}: plate-mat clearance = {clearance_str}")
        if clearance is None or clearance >= _MIN_PLATE_CLEARANCE:
            break
    else:
        print(f"[place_plate] WARNING: plate clearance below {_MIN_PLATE_CLEARANCE} after {_MAX_RESET_TRIES} attempts")


def task_completion_check(env):
    plate = env.scene.object_registry("name", "plate")
    place_mat = env.scene.object_registry("name", "place_mat")
    if (
        plate is None
        or place_mat is None
        or not getattr(env, "robots", None)
        or object_states.OnTop not in plate.states
    ):
        return False
    on_top = bool(plate.states[object_states.OnTop].get_value(other=place_mat))
    return on_top and gripper_far_from_object(env.robots[0], plate)
