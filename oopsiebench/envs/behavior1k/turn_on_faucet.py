"""
Task configuration for **turn_on_faucet**.

Uses the native kitchen faucet in house_single_floor / kitchen_0 (``drop_in_sink_awvzkn_0``,
the same sink fill_bowl fills from). The faucet starts OFF; the task is to reach its knob
to turn it on, then back the gripper off.

Scene : house_single_floor (kitchen_0)
Robot : FrankaMounted (franka0)
"""

from __future__ import annotations

import numpy as np
import omnigibson as og
import torch as th
from omnigibson.object_states import ToggledOn
from omnigibson.object_states import particle_modifier as _particle_modifier
from omnigibson.utils import transform_utils as T

from oopsiebench.envs.behavior1k.base import TaskConfig
from oopsiebench.envs.behavior1k.spatial_checks import eef_world_position_or_raise

# Faster faucet water stream (set before the particle system reads the macros).
_WATER_PARTICLES_PER_STEP = 25
_WATER_STEPS_PER_EMIT = 1
with _particle_modifier.m.unlocked():
    _particle_modifier.m.MAX_PHYSICAL_PARTICLES_APPLIED_PER_STEP = _WATER_PARTICLES_PER_STEP
    _particle_modifier.m.N_STEPS_PER_APPLICATION = _WATER_STEPS_PER_EMIT

ROBOT_NAME = "franka0"
ROBOT_TYPE = "FrankaMounted"

FAUCET_NAME = "drop_in_sink_awvzkn_0"  # native kitchen faucet (toggleable)

ROBOT_POSITION = [5.7, -1.25, 0.0]
ROBOT_ORIENTATION = [0.0, 0.0, -0.7071067811865476, 0.7071067811865476]

# Stand-off toggle assist on the knob, and the retreat distance to finish.
_TOGGLE_ASSIST_DIST = 0.07
_GRIPPER_FAR_FROM_KNOB_M = 0.4

# ── Cameras ──────────────────────────────────────────────────────────────

VIEWER_CAMERA_POS = [6.764060974121094, -1.9225226640701294, 1.3960963487625122]
VIEWER_CAMERA_ORN = [0.44636261463165283, 0.4237414598464966, 0.542643666267395, 0.5716130137443542]

EXTERNAL_CAMERA_CONFIGS = {
    "external_sensor_0": {
        "position": VIEWER_CAMERA_POS,
        "orientation": VIEWER_CAMERA_ORN,
        "horizontal_aperture": 15.0,
    },
}

_U_XY = 0.05
_U_YAW = 0.12
_U_ARM = 0.2
_MAX_RESET_TRIES = 20


def _faucet(env):
    return env.scene.object_registry("name", FAUCET_NAME)


def _set_faucet_off(env):
    """Start episodes with the faucet OFF."""
    sink = _faucet(env)
    if sink is None or not hasattr(sink, "states"):
        return
    try:
        st = sink.states.get(ToggledOn)
        if st is not None:
            st.set_value(False)
    except Exception:
        pass


def _toggle_marker_pos(env):
    """World position of the faucet's togglebutton marker (the knob), or None."""
    sink = _faucet(env)
    if sink is None or ToggledOn not in sink.states:
        return None
    try:
        return sink.states[ToggledOn].link.get_position_orientation()[0]
    except Exception:
        return None


def _assist_turn_on(env):
    sink = _faucet(env)
    if sink is None or ToggledOn not in sink.states or not getattr(env, "robots", None):
        return
    if bool(sink.states[ToggledOn].get_value()):
        return  # already on
    marker = _toggle_marker_pos(env)
    if marker is None:
        return
    eef = eef_world_position_or_raise(env.robots[0])
    marker = marker.reshape(-1)[:3].to(dtype=eef.dtype, device=eef.device)
    if float(th.norm(eef[:3] - marker)) <= _TOGGLE_ASSIST_DIST:
        try:
            sink.states[ToggledOn].set_value(True)
        except Exception:
            pass


def reset(env):
    """Faucet OFF, light robot pose/arm jitter, controllers sync."""
    if not getattr(env, "robots", None):
        return
    robot = env.robots[0]

    _set_faucet_off(env)

    base_pos, base_orn = robot.get_position_orientation()
    base_q = robot.get_joint_positions().clone()

    for attempt in range(_MAX_RESET_TRIES):
        pos = base_pos.clone()
        pos[0] += float(np.random.uniform(-_U_XY, _U_XY))
        pos[1] += float(np.random.uniform(-_U_XY, _U_XY))
        euler = T.quat2euler(base_orn).clone()
        euler[2] = euler[2] + float(np.random.uniform(-_U_YAW, _U_YAW))
        robot.set_position_orientation(pos, T.euler2quat(euler))

        q = base_q.clone()
        for arm_name in robot.arm_control_idx:
            idx = robot.arm_control_idx[arm_name]
            u = (th.rand(len(idx), device=q.device, dtype=q.dtype) * 2 - 1) * _U_ARM
            q[idx] = q[idx] + u
        robot.set_joint_positions(q)
        robot.set_joint_velocities(th.zeros(robot.n_dof, device=q.device, dtype=q.dtype))
        robot.keep_still()

        for ctrl_name in ("arm_0", "gripper_0"):
            ctrl = robot.controllers.get(ctrl_name)
            if ctrl is not None:
                ctrl.reset()

        for _ in range(10):
            og.sim.step()

        env._reset_damage_tracking()
        for _ in range(3):
            og.sim.step()
        update_health = getattr(env, "_update_all_health", None)
        if callable(update_health):
            try:
                update_health()
            except Exception:
                pass
        robot_health = float((env.get_env_health() or {}).get(robot.name, 100.0))
        print(f"[turn_on_faucet] reset attempt {attempt + 1}/{_MAX_RESET_TRIES}: robot health = {robot_health:.1f}")
        if robot_health >= 100.0:
            break
    else:
        print(f"[turn_on_faucet] WARNING: robot still damaged after {_MAX_RESET_TRIES} attempts")

    env._reset_damage_tracking()
    _set_faucet_off(env)  # ensure it's still off after settling


def task_completion_check(env):
    # Flip the faucet on when the gripper reaches its knob (teleop assist).
    _assist_turn_on(env)

    sink = _faucet(env)
    if sink is None or not getattr(env, "robots", None) or ToggledOn not in sink.states:
        return False
    if not bool(sink.states[ToggledOn].get_value()):
        return False

    # Distance from the gripper to the faucet *knob* (the button it presses), not the
    # sink's transform origin — so the required gap is measured from what you touched.
    marker = _toggle_marker_pos(env)
    if marker is None:
        return False
    eef = eef_world_position_or_raise(env.robots[0])
    marker = marker.reshape(-1)[:3].to(dtype=eef.dtype, device=eef.device)
    return float(th.norm(eef[:3] - marker)) > _GRIPPER_FAR_FROM_KNOB_M


def get_task_config() -> TaskConfig:
    return TaskConfig(
        task_name="turn_on_faucet",

        use_gpu_dynamics=True,
        # Must stay True so ToggledOn follows the robot touching the faucet knob.
        enable_transition_rules=True,
        physics_frequency=120.0,

        scene_config={
            "scene_model": "house_single_floor",
            "not_load_object_categories": ["ottoman"],
            "load_room_instances": ["kitchen_0"],
        },

        robot_name=ROBOT_NAME,
        robot_type=ROBOT_TYPE,
        robot_config={
            "type": ROBOT_TYPE,
            "name": ROBOT_NAME,
            "position": ROBOT_POSITION,
            "orientation": ROBOT_ORIENTATION,
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

        task_objects={},

        viewer_camera_pos=VIEWER_CAMERA_POS,
        viewer_camera_orn=VIEWER_CAMERA_ORN,
        external_camera_configs=EXTERNAL_CAMERA_CONFIGS,

        target_objects_health_with_links=[
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

        default_collect_hdf5="demos/behavior1k/teleop_data/turn_on_faucet.hdf5",
        default_playback_hdf5="demos/behavior1k/playback_data/turn_on_faucet_playback.hdf5",
        default_video_dir="demos/behavior1k/playback_videos/turn_on_faucet",
    )
