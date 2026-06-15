"""
Task configuration for **wipe_counter** (sponge-only).

Scene : house_single_floor
Robot : FrankaPanda (franka0)
"""

from __future__ import annotations

import pickle

import numpy as np
import omnigibson as og
import torch as th
from omnigibson.objects.stateful_object import StatefulObject
from omnigibson.utils import transform_utils as T
from omnigibson.utils.bddl_utils import get_system_name_by_synset

from oopsiebench.envs.behavior1k.base import TaskConfig
from oopsiebench.envs.behavior1k.spatial_checks import eef_world_position_or_raise

ROBOT_NAME = "franka0"
ROBOT_TYPE = "FrankaPanda"

INIT_STATE_PATH = "resources/init_states/wipe_counter.pkl"

# Particle systems to record in teleop HDF5 (not scene-default water).
TRANSITION_SYSTEMS = ("dust",)

# ── Task objects ─────────────────────────────────────────────────────────

TASK_OBJECTS = {
    "sponge": {
        "type": "DatasetObject",
        "name": "sponge",
        "category": "sponge",
        "model": "qewotb",
        "position": [6.3, 0.2, 1.3],
        "orientation": [0.0, 0.0, 0.0, 1.0],
        "scale": [1.0, 1.0, 1.0],
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

# Gripper must be lifted this far above the wiped surface to finish (matches pick_egg).
_LIFT_Z = 0.25

_DIRT_SYNSETS = ("dust.n.01", "dirt.n.02")


def _patch_stateful_object_texture_dtype_guard():
    """Guard against Float/Double mismatch in StatefulObject._update_texture_change."""
    if getattr(StatefulObject, "_oopsie_texture_dtype_patch_applied", False):
        return

    def _patched(self, object_state):
        if object_state is None:
            albedo_add = 0.0
            diffuse_tint = th.tensor([1.0, 1.0, 1.0], dtype=th.float32)
        else:
            albedo_add, diffuse_tint = object_state.get_texture_change_params()
            if not isinstance(diffuse_tint, th.Tensor):
                diffuse_tint = th.tensor(diffuse_tint, dtype=th.float32)
        for material in self.materials:
            if material.albedo_add != albedo_add:
                material.albedo_add = albedo_add
            mat_tint = material.diffuse_tint
            cast_tint = diffuse_tint.to(dtype=mat_tint.dtype, device=mat_tint.device)
            if not th.allclose(mat_tint, cast_tint):
                material.diffuse_tint = cast_tint

    StatefulObject._update_texture_change = _patched
    StatefulObject._oopsie_texture_dtype_patch_applied = True


def _spawn_counter_dirt(env):
    """Find the counter under the sponge, generate a compact dirt stain on it, cache the group."""
    sponge = env.scene.object_registry("name", "sponge")
    if sponge is None:
        raise RuntimeError("[wipe_counter] sponge not found")

    # Resolve a dirt particle system.
    dirt_system = None
    for syn in _DIRT_SYNSETS:
        system_name = get_system_name_by_synset(syn)
        if system_name in env.scene.available_systems:
            dirt_system = env.scene.get_system(system_name, force_init=True)
            break
    if dirt_system is None:
        raise RuntimeError(f"[wipe_counter] no dirt system available among {_DIRT_SYNSETS}")

    # Find the closest surface (top-z) directly below the sponge.
    sponge_pos, _ = sponge.get_position_orientation()
    if not isinstance(sponge_pos, th.Tensor):
        sponge_pos = th.tensor(sponge_pos, dtype=th.float32)
    target_z = float(sponge_pos[2])
    excluded = {sponge.name}
    for r in getattr(env, "robots", []) or []:
        if hasattr(r, "name"):
            excluded.add(r.name)
    surface = None
    best_dz = None
    for obj in getattr(env.scene, "objects", []) or []:
        if obj is None or getattr(obj, "name", None) in excluded:
            continue
        if not hasattr(obj, "aabb") or obj.aabb is None:
            continue
        amin, amax = obj.aabb
        top_z = float(amax[2])
        if top_z >= target_z:
            continue
        if not (float(amin[0]) <= float(sponge_pos[0]) <= float(amax[0])
                and float(amin[1]) <= float(sponge_pos[1]) <= float(amax[1])):
            continue
        dz = target_z - top_z
        if best_dz is None or dz < best_dz:
            best_dz = dz
            surface = obj
    if surface is None:
        raise RuntimeError("[wipe_counter] no supporting surface found under sponge")

    group = dirt_system.get_group_name(obj=surface) if hasattr(dirt_system, "get_group_name") else surface.name
    if hasattr(dirt_system, "groups") and group not in dirt_system.groups:
        dirt_system.create_attachment_group(obj=surface)
    if hasattr(dirt_system, "groups") and group in dirt_system.groups:
        dirt_system.remove_all_group_particles(group=group)

    dirt_system.generate_group_particles_on_object(group=group, max_samples=96, min_samples_for_success=1)
    og.sim.render()

    # Pack particles into a compact, dense circular stain near the sponge.
    if hasattr(dirt_system, "get_group_particles_position_orientation") and hasattr(
        dirt_system, "set_group_particles_position_orientation"
    ):
        pos, orn = dirt_system.get_group_particles_position_orientation(group=group)
        if isinstance(pos, th.Tensor) and pos.ndim == 2 and pos.shape[0] > 0:
            _, amax = surface.aabb
            pos = pos.clone()
            stain_cx = float(sponge_pos[0]) - 0.10
            stain_cy = float(sponge_pos[1])
            stain_r = 0.035
            n_pts = pos.shape[0]
            angles = th.linspace(0.0, 2.0 * th.pi, n_pts, dtype=th.float32, device=pos.device)
            radii = th.rand(n_pts, dtype=th.float32, device=pos.device).pow(2.0) * stain_r
            pos[:, 0] = stain_cx + radii * th.cos(angles)
            pos[:, 1] = stain_cy + radii * th.sin(angles)
            pos[:, 2] = float(amax[2]) + 0.002
            dirt_system.set_group_particles_position_orientation(group=group, positions=pos, orientations=orn)
            og.sim.render()

    env._wipe_counter_dirt = (dirt_system, group)
    # Cache the wiped surface's top z so completion can require a gripper lift above it.
    env._wipe_counter_surface_top_z = float(surface.aabb[1][2])


def reset(env):
    """Load init pickle, jitter robot pose/joints, spawn counter dirt, settle."""
    _patch_stateful_object_texture_dtype_guard()

    with open(INIT_STATE_PATH, "rb") as f:
        state_flat_array = pickle.load(f)
    og.sim.load_state(state_flat_array, serialized=True)

    if not getattr(env, "robots", None):
        return
    robot = env.robots[0]
    pos, orn = robot.get_position_orientation()
    pos = pos.clone()
    pos[0] += float(np.random.uniform(-_U_XY, _U_XY))
    pos[1] += float(np.random.uniform(-_U_XY, _U_XY))
    euler = T.quat2euler(orn).clone()
    euler[2] = euler[2] + float(np.random.uniform(-_U_YAW, _U_YAW))
    robot.set_position_orientation(pos, T.euler2quat(euler))

    q = robot.get_joint_positions().clone()
    for arm_name in robot.arm_control_idx:
        idx = robot.arm_control_idx[arm_name]
        u = (th.rand(len(idx), device=q.device, dtype=q.dtype) * 2 - 1) * _U_ARM
        q[idx] = q[idx] + u
    robot.set_joint_positions(q)
    robot.set_joint_velocities(th.zeros(robot.n_dof, device=q.device, dtype=q.dtype))
    robot.keep_still()

    _spawn_counter_dirt(env)

    for _ in range(10):
        og.sim.step()


def task_completion_check(env):
    cached = getattr(env, "_wipe_counter_dirt", None)
    if cached is None:
        return False
    dirt_system, group = cached
    if int(dirt_system.num_group_particles(group=group)) != 0:
        return False
    # Also require the gripper to be lifted up off the counter (same height as pick_egg).
    surface_top_z = getattr(env, "_wipe_counter_surface_top_z", None)
    if surface_top_z is None or not getattr(env, "robots", None):
        return False
    eef_z = float(eef_world_position_or_raise(env.robots[0])[2])
    return (eef_z - surface_top_z) >= _LIFT_Z


def get_task_config() -> TaskConfig:
    return TaskConfig(
        task_name="wipe_counter",
        use_gpu_dynamics=False,
        enable_transition_rules=False,
        scene_config={
            "scene_model": "house_single_floor",
            "not_load_object_categories": ["ottoman"],
            "load_room_instances": [
                "kitchen_0", "dining_room_0", "entryway_0", "living_room_0",
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
        target_objects_health_with_links= [
            f"{ROBOT_NAME}@eef_link",
            f"{ROBOT_NAME}@panda_hand",
            f"{ROBOT_NAME}@panda_leftfinger",
            f"{ROBOT_NAME}@panda_rightfinger",
        ],
        target_objects_health=[ROBOT_NAME],
        target_objects_forces=[
            f"{ROBOT_NAME}@eef_link",
            f"{ROBOT_NAME}@panda_hand",
            f"{ROBOT_NAME}@panda_leftfinger",
            f"{ROBOT_NAME}@panda_rightfinger",
        ],
        force_keys=["filtered_qs_forces", "impact_forces"],
        default_collect_hdf5="demos/behavior1k/teleop_data/wipe_counter.hdf5",
        default_playback_hdf5="demos/behavior1k/playback_data/wipe_counter_playback.hdf5",
        default_video_dir="demos/behavior1k/playback_videos/wipe_counter",
    )
