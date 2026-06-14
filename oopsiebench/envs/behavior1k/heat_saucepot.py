"""
Task configuration for **heat_saucepot**.

Uses the native cooktop from house_single_floor (the ``burner_mjvqii_0`` under the
range hood). The saucepot starts on the back-right burner; the task is to turn the
burner on (via its right-most knob) and move the pot onto the front-right burner,
then back the gripper off.

Scene : house_single_floor (kitchen)
Robot : FrankaMounted (franka0)
"""

from __future__ import annotations

import numpy as np
import omnigibson as og
import torch as th
from omnigibson import object_states
from omnigibson.object_states import HeatSourceOrSink
from omnigibson.utils import transform_utils as T

from oopsiebench.envs.behavior1k.base import TaskConfig
from oopsiebench.envs.behavior1k.spatial_checks import (
    gripper_far_from_object,
    eef_world_position_or_raise,
)

ROBOT_NAME = "franka0"
ROBOT_TYPE = "FrankaMounted"

BURNER_NAME = "burner_mjvqii_0"  # native house_single_floor cooktop
BURNER_XY = [4.17, -0.51]

ROBOT_POSITION = [4.8, -0.51, 0.0]
ROBOT_ORIENTATION = [0.0, 0.0, 1.0, 0.0]
_ORN_Z_PLUS_90_DEG = [0.0, 0.0, 0.7071067811865476, 0.7071067811865476]

# Burner heat source, re-tamed in reset() (its defaults would cook the gripper at the knobs).
_BURNER_HEAT_RADIUS = 0.12
_BURNER_HEAT_TEMP = 300.0

# Burner-frame xy of the heat-source links: pot starts back-right, goal is front-right.
_POT_SPAWN_LOCAL_XY = [-0.1123, 0.2542]
_TARGET_BURNER_LOCAL_XY = [0.0748, 0.2532]
_POT_ON_BURNER_XY_MAX_M = 0.08

# Stand-off toggle assist on the right-most knob.
_TOGGLE_BUTTON_LOCAL_XYZ = [0.2216, 0.1322, 0.0182]
_TOGGLE_ASSIST_DIST = 0.07

_GRIPPER_FAR_M = 0.2

# ── Task objects ─────────────────────────────────────────────────────────

TASK_OBJECTS = {
    "saucepot": {
        "type": "DatasetObject",
        "name": "saucepot",
        "category": "saucepot",
        "model": "fbfmwt",
        # Above the burner; reset() repositions + seats it onto the cooktop surface.
        "position": [BURNER_XY[0], BURNER_XY[1], 1.15],
        "scale": [0.5, 0.5, 0.5],
        "orientation": _ORN_Z_PLUS_90_DEG,
    },
}

# ── Cameras ──────────────────────────────────────────────────────────────

VIEWER_CAMERA_POS = [5.453732490539551, -1.596856713294983, 1.899999976158142]
VIEWER_CAMERA_ORN = [0.485004723072052, 0.152801513671875, 0.258602499961853, 0.8213080167770386]

EXTERNAL_CAMERA_CONFIGS = {
    "external_sensor_0": {
        "position": VIEWER_CAMERA_POS,
        "orientation": VIEWER_CAMERA_ORN,
        "horizontal_aperture": 20.0,
    },
}

_U_XY = 0.05
_U_YAW = 0.12
_U_ARM = 0.2


def _cooktop_local_to_world(burner, local):
    """World position of a burner-frame xy (or xyz) point, transformed by the cooktop pose."""
    vec = list(local)
    if len(vec) == 2:
        vec = vec + [0.0]
    bpos, born = burner.get_position_orientation()
    local_t = th.tensor(vec, dtype=bpos.dtype, device=bpos.device)
    return bpos + T.quat_apply(born, local_t)


def _support_surface_top_z(env, obj) -> float:
    """Top-z of the nearest surface directly beneath obj's xy (here: the cooktop)."""
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
        raise RuntimeError("[heat_saucepot] no supporting surface found under saucepot")
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]


def _seat_on_surface(env, obj):
    """Shift obj vertically so its AABB bottom rests on the surface beneath it."""
    top_z = _support_surface_top_z(env, obj)
    if object_states.AABB not in obj.states:
        return
    lower, _ = obj.states[object_states.AABB].get_value()
    pos, orn = obj.get_position_orientation()
    pos = pos.clone()
    pos[2] = pos[2] + (top_z - float(lower[2]))
    obj.set_position_orientation(pos, orn)


def _tame_burner_heat(env):
    """Tighten + cool the native burner's live heat source (re-applied each reset/playback)."""
    burner = env.scene.object_registry("name", BURNER_NAME)
    if burner is None or not hasattr(burner, "states"):
        return
    hs = burner.states.get(HeatSourceOrSink)
    if hs is None:
        return
    hs.distance_threshold = _BURNER_HEAT_RADIUS
    hs._temperature = _BURNER_HEAT_TEMP


def _place_saucepot(env):
    """Position the saucepot on its start (back-right) burner and seat it on the cooktop."""
    saucepot = env.scene.object_registry("name", "saucepot")
    burner = env.scene.object_registry("name", BURNER_NAME)
    if saucepot is None:
        return
    if burner is not None:
        world = _cooktop_local_to_world(burner, _POT_SPAWN_LOCAL_XY)
        sp, so = saucepot.get_position_orientation()
        sp = sp.clone()
        sp[0], sp[1] = world[0], world[1]
        saucepot.set_position_orientation(sp, so)
    try:
        _seat_on_surface(env, saucepot)
    except RuntimeError as e:
        print(e)
    if hasattr(saucepot, "keep_still"):
        saucepot.keep_still()


def _burner_on(burner) -> bool:
    if burner is None or object_states.ToggledOn not in burner.states:
        return False
    if not burner.states[object_states.ToggledOn].get_value():
        return False
    if HeatSourceOrSink not in burner.states:
        return True
    return bool(burner.states[HeatSourceOrSink].get_value())


def _pot_on_burner(burner, saucepot) -> bool:
    """OnTop the cooktop AND over the front-right target burner (not just anywhere)."""
    if burner is None or saucepot is None or object_states.OnTop not in saucepot.states:
        return False
    try:
        if not bool(saucepot.states[object_states.OnTop].get_value(other=burner)):
            return False
    except (KeyError, AttributeError):
        return False
    target = _cooktop_local_to_world(burner, _TARGET_BURNER_LOCAL_XY)
    pot_xy = saucepot.get_position_orientation()[0][:2].to(dtype=target.dtype, device=target.device)
    return float(th.norm(pot_xy - target[:2])) <= _POT_ON_BURNER_XY_MAX_M


def _assist_turn_on(env):
    """Toggle the burner on once the gripper reaches the right-most knob."""
    burner = env.scene.object_registry("name", BURNER_NAME)
    if burner is None or object_states.ToggledOn not in burner.states or not getattr(env, "robots", None):
        return
    if bool(burner.states[object_states.ToggledOn].get_value()):
        return  # already on
    marker = _cooktop_local_to_world(burner, _TOGGLE_BUTTON_LOCAL_XYZ)
    eef = eef_world_position_or_raise(env.robots[0])
    marker = marker.reshape(-1)[:3].to(dtype=eef.dtype, device=eef.device)
    if float(th.norm(eef[:3] - marker)) <= _TOGGLE_ASSIST_DIST:
        try:
            burner.states[object_states.ToggledOn].set_value(True)
        except Exception:
            pass


def reset(env):
    """Tame the burner, place the saucepot on its start burner, jitter robot, settle."""
    _tame_burner_heat(env)
    _place_saucepot(env)

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


def playback_reset(env):
    """Match teleop's initial setup at playback start — reset() isn't run during playback,
    so without this the burner stays untamed and the pot starts at its raw spawn pose."""
    _tame_burner_heat(env)
    _place_saucepot(env)


def task_completion_check(env):
    # Toggle the burner on when the gripper reaches the right-most knob (teleop assist).
    _assist_turn_on(env)

    burner = env.scene.object_registry("name", BURNER_NAME)
    saucepot = env.scene.object_registry("name", "saucepot")
    if burner is None or saucepot is None or not getattr(env, "robots", None):
        return False

    pot_ok = _burner_on(burner) and _pot_on_burner(burner, saucepot)
    gripper_far = gripper_far_from_object(env.robots[0], saucepot, threshold=_GRIPPER_FAR_M)
    return pot_ok and gripper_far


def get_task_config() -> TaskConfig:
    return TaskConfig(
        task_name="heat_saucepot",

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

        task_objects=TASK_OBJECTS,

        viewer_camera_pos=VIEWER_CAMERA_POS,
        viewer_camera_orn=VIEWER_CAMERA_ORN,
        external_camera_configs=EXTERNAL_CAMERA_CONFIGS,

        target_objects_health_with_links=[
            f"{ROBOT_NAME}@eef_link",
            f"{ROBOT_NAME}@panda_hand",
            f"{ROBOT_NAME}@panda_leftfinger",
            f"{ROBOT_NAME}@panda_rightfinger",
            "saucepot@base_link",
        ],
        target_objects_health=[ROBOT_NAME, "saucepot"],
        target_objects_temperature=[
            f"{ROBOT_NAME}@eef_link",
            f"{ROBOT_NAME}@panda_hand",
            f"{ROBOT_NAME}@panda_leftfinger",
            f"{ROBOT_NAME}@panda_rightfinger",
            "saucepot@base_link",
        ],
        target_objects_forces=[
            f"{ROBOT_NAME}@eef_link",
            f"{ROBOT_NAME}@panda_hand",
            f"{ROBOT_NAME}@panda_leftfinger",
            f"{ROBOT_NAME}@panda_rightfinger",
            "saucepot@base_link",
        ],
        force_keys=["filtered_qs_forces"],

        default_collect_hdf5="demos/behavior1k/teleop_data/heat_saucepot.hdf5",
        default_playback_hdf5="demos/behavior1k/playback_data/heat_saucepot_playback.hdf5",
        default_video_dir="demos/behavior1k/playback_videos/heat_saucepot",
    )
