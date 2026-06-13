"""
Task configuration for **nav_to_table**.

Scene : Rs_int (Tiago primitives-style)
Robot : Tiago (tiago0)
Damage: mechanical only (task objects of interest)
"""

import pickle

import numpy as np
import omnigibson as og
import torch as th

from oopsiebench.envs.behavior1k.base import TaskConfig

ROBOT_NAME = "tiago0"
ROBOT_TYPE = "Tiago"

# ── Task objects ─────────────────────────────────────────────────────────

TASK_OBJECTS = {
    "pedestal_table": {
        "type": "DatasetObject",
        "name": "pedestal_table",
        "category": "pedestal_table",
        "model": "djflkd",
        "position": [-0.5, 0.0, 0.10],
        "orientation": [0.0, 0.0, 0.0, 1.0],
        "scale": [0.5, 0.5, 1.0],
    },
    "vase": {
        "type": "DatasetObject",
        "name": "vase",
        "category": "vase",
        "model": "uuypot",
        "position": [-0.5, -1.0, 0.10],
        "orientation": [0.0, 0.0, 0.0, 1.0],
        "scale": [0.5, 0.5, 0.5],
    },
    "swivel_chair": {
        "type": "DatasetObject",
        "name": "swivel_chair",
        "category": "swivel_chair",
        "model": "pkpcew",
        "position": [-0.5, 1.0, 0.50],
        "orientation": [0.0, 0.0, 0.0, 1.0],
        "scale": [1.0, 1.0, 1.0],
    },
    # Fixed barrier right of the pedestal (fixed_base keeps it out of the serialized state).
    "coffee_table": {
        "type": "DatasetObject",
        "name": "coffee_table",
        "category": "coffee_table",
        "model": "fqluyq",
        "position": [-0.4763, -1.2196, 0.2838],
        "orientation": [0.0, 0.0, 1.0, 0.0],
        "scale": [1.1712, 1.0266, 0.9481],
        "fixed_base": True,
    },
    "water_bottle": {
        "type": "DatasetObject",
        "name": "water_bottle",
        "category": "bottle_of_water",
        "model": "hrzznl",
        "position": [0.0, 0.0, 1.5],
        "orientation": [0.0, 0.0, 0.0, 1.0],
        "scale": [0.9, 0.9, 0.9],
    },
}

# ── Cameras ──────────────────────────────────────────────────────────────

VIEWER_CAMERA_POS = [1.5345655679702759, -2.3398592472076416, 1.3116816282272339]
VIEWER_CAMERA_ORN = [0.605172872543335, 0.14635765552520752, 0.18393288552761078, 0.7606010437011719]

EXTERNAL_CAMERA_CONFIGS = {
    # One stationary camera at the viewer pose (world_fixed) — the only one the video uses.
    "external_sensor_0": {
        "position": VIEWER_CAMERA_POS,
        "orientation": VIEWER_CAMERA_ORN,
        "horizontal_aperture": 20.995,  # matches the OG viewer FOV
        "world_fixed": True,
    },
}

# ── Public entry point ───────────────────────────────────────────────────

def get_task_config() -> TaskConfig:
    return TaskConfig(
        task_name="nav_to_table",
        use_gpu_dynamics=True,
        enable_transition_rules=False,
        scene_config={
            "type": "InteractiveTraversableScene",
            "scene_model": "Rs_int",
            "include_robots": False,
            "load_object_categories": ["floors", "walls", "breakfast_table"],
        },
        robot_name=ROBOT_NAME,
        robot_type=ROBOT_TYPE,
        robot_config={
            "type": ROBOT_TYPE,
            "name": ROBOT_NAME,
            "position": [0.0, 0.0, 0.0],
            "orientation": [0.0, 0.0, 0.0, 1.0],
            "default_arm_pose": "horizontal",
            "grasping_mode": "assisted",
            "obs_modalities": ["rgb", "depth"],
            "action_normalize": False,
            "self_collisions": True,
            "controller_config": {
                "arm_left": {
                    "name": "InverseKinematicsController",
                    "command_input_limits": None,
                },
                "gripper_left": {
                    "name": "MultiFingerGripperController",
                    "command_input_limits": (0.0, 1.0),
                    "mode": "smooth",
                },
                "arm_right": {
                    "name": "InverseKinematicsController",
                    "command_input_limits": None,
                },
                "gripper_right": {
                    "name": "MultiFingerGripperController",
                    "command_input_limits": (0.0, 1.0),
                    "mode": "smooth",
                },
            },
            "exclude_sensor_names": ["left_eef_link", "right_eef_link"],
        },
        task_objects=TASK_OBJECTS,
        viewer_camera_pos=VIEWER_CAMERA_POS,
        viewer_camera_orn=VIEWER_CAMERA_ORN,
        external_camera_configs=EXTERNAL_CAMERA_CONFIGS,
        target_objects_health_with_links=[
            "pedestal_table@base_link",
            "vase@base_link",
            "swivel_chair@base_link",
        ],
        target_objects_health=["pedestal_table", "vase", "swivel_chair"],
        target_objects_forces=[
            "pedestal_table@base_link",
            "vase@base_link",
            "swivel_chair@base_link",
        ],
        force_keys=["filtered_qs_forces", "impact_forces"],
        # Let the scene settle on playback load before damage tracking starts, so the
        # vase isn't destroyed by the teleport/impact artifact at step 0 (was 4).
        post_playback_env_setup=lambda env: setattr(env, "playback_init_skip_steps", 30),
        default_collect_hdf5="demos/behavior1k/teleop_data/nav_to_table.hdf5",
        default_playback_hdf5="demos/behavior1k/playback_data/nav_to_table_playback.hdf5",
        default_video_dir="demos/behavior1k/playback_videos/nav_to_table",
    )


_BOTTLE_U_XY = 0.03
_LIFT_Z = 0.1

INIT_STATE_PATH = "resources/init_states/nav_to_table.pkl"


def reset(env):
    """Restore the saved init state (robot behind the pedestal table, vase on it),
    then snap the water bottle onto the breakfast_table top with light XY jitter."""
    # Restore the hand-authored arrangement. The harness has already called
    # env.reset() before this, so loading the serialized sim state is enough
    # (mirrors add_firewood / wipe_counter / open_drawer reset()).
    with open(INIT_STATE_PATH, "rb") as f:
        state_flat_array = pickle.load(f)
    og.sim.load_state(state_flat_array, serialized=True)
    for _ in range(5):
        og.sim.step()

    try:
        bottle = env.scene.object_registry("name", "water_bottle")
        if bottle is None:
            return

        # breakfast_table is part of the base scene; substring match handles instanced names.
        table = None
        for obj in getattr(env.scene, "objects", []) or []:
            name = (getattr(obj, "name", "") or "").lower()
            cat = (getattr(obj, "category", "") or "").lower()
            if "breakfast_table" in name or "breakfast_table" in cat:
                table = obj
                break
        if table is None:
            table = (env.scene.object_registry("category", "breakfast_table")
                     or env.scene.object_registry("name", "breakfast_table"))
        if table is None:
            return

        for _ in range(5):
            og.sim.step()

        if not hasattr(table, "aabb") or table.aabb is None:
            return
        tmin, tmax = table.aabb

        cx = (float(tmin[0]) + float(tmax[0])) * 0.5
        cy = (float(tmin[1]) + float(tmax[1])) * 0.5

        z_offset = 0.05
        if hasattr(bottle, "aabb") and bottle.aabb is not None:
            mmin, mmax = bottle.aabb
            z_offset = max(0.02, 0.5 * float(mmax[2] - mmin[2])) + 0.002

        pos = th.tensor(
            [
                cx + float(np.random.uniform(-_BOTTLE_U_XY, _BOTTLE_U_XY)),
                cy + float(np.random.uniform(-_BOTTLE_U_XY, _BOTTLE_U_XY)),
                float(tmax[2]) + float(z_offset),
            ],
            dtype=th.float32,
        )
        bottle.set_position_orientation(pos, th.tensor([0.0, 0.0, 0.0, 1.0], dtype=th.float32))
        try:
            bottle.keep_still()
        except Exception:
            pass
        bottle_pos, _ = bottle.get_position_orientation()
        env._nav_bottle_start_z = float(bottle_pos[2])
    except Exception:
        return


def playback_reset(env):
    """reset() (which seats the vase on the pedestal) isn't called during playback — re-apply
    it so objects start arranged instead of teleporting in via the state replay."""
    reset(env)


def task_completion_check(env):
    start_z = getattr(env, "_nav_bottle_start_z", None)
    if start_z is None:
        return False
    bottle = env.scene.object_registry("name", "water_bottle")
    if bottle is None:
        return False
    bottle_pos, _ = bottle.get_position_orientation()
    return (float(bottle_pos[2]) - start_z) >= _LIFT_Z


def register_teleop_keys(env, kb):
    """Teleop post-setup hook: show the robot's head camera in a docked side viewport.

    Named ``register_teleop_keys`` because that is the only task hook teleop runs after
    ``setup_viewport_layout`` (which hides the robot-camera viewport). UI-only; does not
    affect HDF5 collection. ``kb`` is unused.
    """
    import omnigibson.lazy as lazy
    from omnigibson.sensors import VisionSensor
    from omnigibson.utils.ui_utils import dock_window

    try:
        cam = next((s for s in env.robots[0].sensors.values()
                    if isinstance(s, VisionSensor)), None)
        if cam is None:
            print("[nav_to_table] no robot camera found; skipping head-camera viewport")
            return
        cam.viewer_visibility = True
        dock_window(
            space=lazy.omni.ui.Workspace.get_window("DockSpace"),
            name=cam._viewport.name,
            location=lazy.omni.ui.DockPosition.LEFT,
            ratio=0.3,
        )
        for _ in range(5):
            og.sim.render()
        print(f"[nav_to_table] head camera '{cam.name}' shown in '{cam._viewport.name}'")
    except Exception as e:  # viewport tweaks must never break teleop
        print(f"[nav_to_table] could not show head-camera viewport: {e}")
