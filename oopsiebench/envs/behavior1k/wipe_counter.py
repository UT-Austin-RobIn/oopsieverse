"""
Task configuration for **wipe_counter** (sponge-only).

This task is intentionally minimal right now:
- One Franka robot
- One sponge placed at a fixed pose (matching the `laptop` pose in `pour_water.py`)
"""

from __future__ import annotations

import omnigibson as og
import torch as th

from oopsiebench.envs.behavior1k.base import TaskConfig
from omnigibson.utils.bddl_utils import get_system_name_by_synset
from omnigibson.objects.stateful_object import StatefulObject

ROBOT_NAME = "franka0"
ROBOT_TYPE = "FrankaPanda"

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


def _patch_stateful_object_texture_dtype_guard():
    """
    Local safety patch (project-side only): guard against Float/Double mismatch in
    StatefulObject._update_texture_change's th.allclose call.
    """
    if getattr(StatefulObject, "_oopsie_texture_dtype_patch_applied", False):
        return

    def _patched_update_texture_change(self, object_state):
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

    StatefulObject._update_texture_change = _patched_update_texture_change
    StatefulObject._oopsie_texture_dtype_patch_applied = True
    print("[wipe_counter] Applied StatefulObject texture dtype guard patch")


def reset(env):
    _patch_stateful_object_texture_dtype_guard()

    if not getattr(env, "robots", None):
        return
    robot = env.robots[0]

    sponge = env.scene.object_registry("name", "sponge")
    if sponge is None:
        raise RuntimeError("[wipe_counter] sponge object not found in scene by name='sponge'")

    sponge.set_position_orientation([6.3, 0.2, 1.3], [0.0, 0.0, 0.0, 1.0])

    # Let the sim settle briefly so objects do not start in an unstable intermediate state.
    if hasattr(sponge, "keep_still"):
        for _ in range(10):
            sponge.keep_still()
            og.sim.step()
    else:
        for _ in range(10):
            og.sim.step()

    # Start with gripper closed (requested behavior).
    try:
        close_action = th.zeros(robot.action_dim)
        close_action[robot.gripper_action_idx[robot.default_arm]] = -1.0
        for _ in range(8):
            env.step(close_action)
    except Exception as e:
        print(f"[wipe_counter] warning: failed to enforce closed gripper at reset: {e}")

    spawn_dirt = bool(getattr(env, "_teleop_loaded_from_pkl", False))
    if not spawn_dirt:
        # If we don't have a compatible saved state loaded yet, keep the initial
        # state "dirt-free" so saving/loading doesn't depend on runtime particle setup.
        print("[wipe_counter] skipping dirt spawn (waiting for pkl load)")
        return

    # Spawn visual dirt on the supporting surface (counter/table) under sponge.
    # Try dust first (matches clean_pan examples), then dirt.
    candidate_synsets = ["dust.n.01", "dirt.n.02"]
    dirt_system = None
    chosen_synset = None
    for syn in candidate_synsets:
        system_name = get_system_name_by_synset(syn)
        if system_name in env.scene.available_systems:
            dirt_system = env.scene.get_system(system_name, force_init=True)
            chosen_synset = syn
            break

    if dirt_system is None:
        raise RuntimeError(f"[wipe_counter][dirt] none of {candidate_synsets} resolved to available systems")

    # Pick the surface underneath sponge using AABB checks.
    sponge_pos, _ = sponge.get_position_orientation()
    sponge_pos = sponge_pos if isinstance(sponge_pos, th.Tensor) else th.tensor(sponge_pos, dtype=th.float32)
    target_z = float(sponge_pos[2])

    def _inside_xy(p, aabb_min, aabb_max):
        return float(aabb_min[0]) <= float(p[0]) <= float(aabb_max[0]) and float(aabb_min[1]) <= float(p[1]) <= float(aabb_max[1])

    candidates = []
    excluded = {sponge.name}
    for r in getattr(env, "robots", []) or []:
        if hasattr(r, "name"):
            excluded.add(r.name)
    for obj in getattr(env.scene, "objects", []) or []:
        if obj is None or getattr(obj, "name", None) in excluded:
            continue
        if not hasattr(obj, "aabb") or obj.aabb is None:
            continue
        aabb_min, aabb_max = obj.aabb
        # only consider surfaces below the objects
        top_z = float(aabb_max[2])
        if top_z >= target_z:
            continue
        inside_sponge = _inside_xy(sponge_pos, aabb_min, aabb_max)
        if not inside_sponge:
            continue
        dz = target_z - top_z
        candidates.append((dz, obj, inside_sponge, aabb_min, aabb_max))

    if not candidates:
        raise RuntimeError("[wipe_counter][dirt] no supporting surface found under sponge")

    candidates.sort(key=lambda x: x[0])  # closest top-z below sponge
    dz, surface, inside_sponge, smin, smax = candidates[0]
    print(
        f"[wipe_counter][dirt] chosen surface={surface.name} category={getattr(surface, 'category', None)} "
        f"dz={dz} inside_sponge={inside_sponge} "
        f"aabb_min={smin} aabb_max={smax}"
    )

    if hasattr(dirt_system, "get_group_name"):
        group = dirt_system.get_group_name(obj=surface)
    else:
        group = surface.name

    if hasattr(dirt_system, "groups") and group not in dirt_system.groups:
        dirt_system.create_attachment_group(obj=surface)

    # Clear only this chosen-surface group if it already has particles.
    if hasattr(dirt_system, "groups") and group in dirt_system.groups:
        dirt_system.remove_all_group_particles(group=group)

    # Keep the stain compact and dense (not scattered across the full surface).
    n_samples = 96
    success = dirt_system.generate_group_particles_on_object(
        group=group,
        max_samples=n_samples,
        min_samples_for_success=1,
    )
    og.sim.render()
    n_after = dirt_system.num_group_particles(group=group)
    print(
        f"[wipe_counter][dirt] synset={chosen_synset} system={dirt_system.name} "
        f"group={group} success={success} n_after={n_after}"
    )

    # Force particles near top of chosen surface AABB and pack XY into a tiny circle
    # to make a dense "stain" instead of a broad scatter.
    if n_after > 0 and hasattr(dirt_system, "get_group_particles_position_orientation") and hasattr(
        dirt_system, "set_group_particles_position_orientation"
    ):
        pos, orn = dirt_system.get_group_particles_position_orientation(group=group)
        if isinstance(pos, th.Tensor) and pos.ndim == 2 and pos.shape[0] > 0:
            _, surface_aabb_max = surface.aabb
            pos = pos.clone()

            # Stain center: slightly left of sponge on the support surface.
            stain_cx = float(sponge_pos[0]) - 0.10
            stain_cy = float(sponge_pos[1])
            stain_r = 0.035

            n_pts = pos.shape[0]
            angles = th.linspace(0.0, 2.0 * th.pi, n_pts, dtype=th.float32, device=pos.device)
            # Bias radii toward center so stain looks denser in the middle.
            radii = th.rand(n_pts, dtype=th.float32, device=pos.device).pow(2.0) * stain_r
            pos[:, 0] = stain_cx + radii * th.cos(angles)
            pos[:, 1] = stain_cy + radii * th.sin(angles)
            pos[:, 2] = float(surface_aabb_max[2]) + 0.002
            dirt_system.set_group_particles_position_orientation(group=group, positions=pos, orientations=orn)
            og.sim.render()
            print(
                f"[wipe_counter][dirt] compact stain center=({stain_cx:.4f},{stain_cy:.4f}) radius={stain_r} "
                f"top_z={float(surface_aabb_max[2]) + 0.002} "
                f"min={pos.min(dim=0).values} max={pos.max(dim=0).values}"
            )


def get_task_config() -> TaskConfig:
    return TaskConfig(
        task_name="wipe_counter",
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
        target_objects_health_with_links=[],
        target_objects_health=[],
        target_objects_forces=[],
        force_keys=["filtered_qs_forces", "impact_forces"],
        default_collect_hdf5="demos/behavior1k/teleop_data/wipe_counter.hdf5",
        default_playback_hdf5="demos/behavior1k/playback_data/wipe_counter_playback.hdf5",
        default_video_dir="demos/behavior1k/playback_videos/wipe_counter",
    )
