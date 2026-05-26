"""
Task configuration for **turn_on_stove**.

Scene : Rs_int

Robot (flip ``TURN_ON_STOVE_ROBOT`` below):
    **FrankaPanda** — Fixed base Panda on an optional elevated ``robot_platform``
    slab (helps reach rear burners without a mobile base).

    **Tiago** — Holonomic mobile base plus dual IK arms + grippers.

    OmniGibson does not ship a Panda+Omron (mobile Omni + Panda arm) preset;
    Tiago is the closest bundled substitute when you must drive the base.

Task success (when used): knob ON, gripper retracted from stove, ``saucepot`` ``OnTop`` the stove and
within the burner ``HeatSourceOrSink`` coupling used for temperature propagation.
"""

import copy
import pickle

import numpy as np
import torch as th
import omnigibson as og
from omnigibson.utils import transform_utils as T
from omnigibson.utils.bddl_utils import OBJECT_TAXONOMY

from oopsiebench.envs.behavior1k.base import TaskConfig
from oopsiebench.envs.behavior1k.spatial_checks import gripper_far_from_object
from omnigibson import object_states

# ``FrankaPanda`` | ``Tiago``
TURN_ON_STOVE_ROBOT = "FrankaPanda"
if TURN_ON_STOVE_ROBOT not in ("FrankaPanda", "Tiago"):
    raise ValueError(
        f"TURN_ON_STOVE_ROBOT must be 'FrankaPanda' or 'Tiago'; got {TURN_ON_STOVE_ROBOT!r}"
    )


# ── OmniGibson taxonomy merge for stove heat coupling ─────────────────────────
_STOVE_SYNSET = OBJECT_TAXONOMY.get_synset_from_category("stove")
if _STOVE_SYNSET is None:
    raise RuntimeError(
        "OBJECT_TAXONOMY has no synset for category 'stove'; cannot merge abilities."
    )
_STOVE_ABILITIES = copy.deepcopy(OBJECT_TAXONOMY.get_abilities(_STOVE_SYNSET))
_STOVE_ABILITIES.setdefault("heatSource", {})
_STOVE_ABILITIES["heatSource"]["distance_threshold"] = 0.15

# Thermal coupling sphere is generous; completion uses a tighter horizontal gate so a pot sitting
# on another part of THIS stove (rear deck / inactive burner plane) cannot count unless it is near
# the active heatsource meta-link anchor (matches ``HeatSourceOrSink._update`` sphere center logic).
_TASK_POT_VS_BURNER_XY_MAX_M = 0.055
# Pot bottom must be at least this much above the heatsource anchor z (slack for asset alignment).
_TASK_POT_BOTTOM_ABOVE_BURNER_Z_MIN_M = -0.04

# Articulated stove hinges — PhysX joint friction bump each reset (see pour_water laptop).
_STOVE_JOINT_FRICTION = 5.0e7

# +90° about world Z — OmniGibson ``set_position_orientation`` uses quaternion ``x,y,z,w``.
_ORN_Z_PLUS_90_DEG = [0.0, 0.0, 0.7071067811865476, 0.7071067811865476]

_FIXED_OBJS_REST = {
    "stove": {
        "type": "DatasetObject",
        "name": "stove",
        "category": "stove",
        "model": "igwqpj",
        "position": [-1.5, -2.0, 0.45],
        "orientation": [0, 0, 0, 1],
        "abilities": _STOVE_ABILITIES,
    },
    "saucepot": {
        "type": "DatasetObject",
        "name": "saucepot",
        "category": "saucepot",
        "model": "fbfmwt",
        "position": [-1.6, -1.75, 0.8],
        "scale": [0.5, 0.5, 0.5],
        "orientation": _ORN_Z_PLUS_90_DEG,
    },
}


def _branch_setup():
    """
    Populate robot identity, EE arm key for ``gripper_far_from_object``,
    start pose, and optional Franka platform geometry.
    """
    if TURN_ON_STOVE_ROBOT == "FrankaPanda":
        platform_xy = [-1.0, -2.0]
        platform_scale_xy = 0.4
        platform_scale_z = 0.3
        plat_half_z = platform_scale_z / 2.0
        robot_base_z = float(platform_scale_z)
        return dict(
            robot_name="franka0",
            robot_type="FrankaPanda",
            teleop_arm="default",
            robot_position=[platform_xy[0], platform_xy[1], robot_base_z],
            robot_orientation=[0.0, 0.0, 1.0, 0.0],
            franka_platform={
                "scale_xy": platform_scale_xy,
                "scale_z": platform_scale_z,
                "z_center": plat_half_z,
                "xy": platform_xy,
            },
        )

    # Tiago — floor spawn, holonomic base; right arm biased toward stove -x layout.
    return dict(
        robot_name="tiago0",
        robot_type="Tiago",
        teleop_arm="right",
        robot_position=[-0.6, -2.2, 0.0],
        robot_orientation=[0.0, 0.0, 1.0, 0.0],
        franka_platform=None,
    )


SETUP = _branch_setup()

ROBOT_NAME = SETUP["robot_name"]
ROBOT_TYPE = SETUP["robot_type"]
TELEOP_ARM_FOR_STOVE = SETUP["teleop_arm"]


def _build_task_objects() -> dict:
    objs = dict(_FIXED_OBJS_REST)
    plat = SETUP["franka_platform"]
    if plat is not None:
        out = {
            "robot_platform": {
                "type": "PrimitiveObject",
                "name": "robot_platform",
                "primitive_type": "Cube",
                "category": "object",
                "scale": [plat["scale_xy"], plat["scale_xy"], plat["scale_z"]],
                "fixed_base": True,
                "position": [plat["xy"][0], plat["xy"][1], plat["z_center"]],
                "orientation": [0.0, 0.0, 0.0, 1.0],
                "rgba": [0.42, 0.41, 0.45, 1.0],
            },
            **objs,
        }
        return out
    return objs


TASK_OBJECTS = _build_task_objects()


def _viz_link_targets():
    rn = ROBOT_NAME
    if TURN_ON_STOVE_ROBOT == "FrankaPanda":
        return dict(
            health_with_links=[
                f"{rn}@eef_link",
                f"{rn}@panda_hand",
                f"{rn}@panda_leftfinger",
                f"{rn}@panda_rightfinger",
            ],
            temperature=[
                f"{rn}@eef_link",
                f"{rn}@panda_hand",
                f"{rn}@panda_leftfinger",
                f"{rn}@panda_rightfinger",
            ],
            forces=[
                f"{rn}@eef_link",
                f"{rn}@panda_hand",
                f"{rn}@panda_leftfinger",
                f"{rn}@panda_rightfinger",
                f"{rn}@panda_link7",
                f"{rn}@panda_link6",
                f"{rn}@panda_link5",
            ],
        )
    return dict(
        health_with_links=[
            f"{rn}@gripper_right_link",
            f"{rn}@gripper_right_left_finger_link",
            f"{rn}@gripper_right_right_finger_link",
            f"{rn}@arm_right_7_link",
            f"{rn}@arm_right_6_link",
            f"{rn}@arm_right_5_link",
        ],
        temperature=[
            f"{rn}@gripper_right_link",
            f"{rn}@gripper_right_left_finger_link",
            f"{rn}@gripper_right_right_finger_link",
            f"{rn}@arm_right_7_link",
        ],
        forces=[
            f"{rn}@gripper_right_link",
            f"{rn}@gripper_right_left_finger_link",
            f"{rn}@gripper_right_right_finger_link",
            f"{rn}@arm_right_7_link",
            f"{rn}@arm_right_6_link",
            f"{rn}@arm_right_5_link",
        ],
    )


_VIZ = _viz_link_targets()


def _robot_config() -> dict:
    common = dict(
        type=ROBOT_TYPE,
        name=ROBOT_NAME,
        position=SETUP["robot_position"],
        orientation=SETUP["robot_orientation"],
        grasping_mode="assisted",
        obs_modalities=["rgb", "depth"],
        action_normalize=False,
        self_collisions=True,
    )
    if TURN_ON_STOVE_ROBOT == "FrankaPanda":
        return {
            **common,
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
        }
    # Tiago
    return {
        **common,
        "default_arm_pose": "horizontal",
        "controller_config": {
            "base": {
                "name": "HolonomicBaseJointController",
                "motor_type": "position",
                "command_input_limits": None,
                "use_impedances": False,
            },
            "trunk": {
                "name": "JointController",
                "motor_type": "position",
                "command_input_limits": None,
                "use_delta_commands": False,
                "use_impedances": False,
            },
            "camera": {
                "name": "JointController",
                "motor_type": "position",
                "command_input_limits": None,
                "use_delta_commands": False,
                "use_impedances": False,
            },
            "arm_left": {
                "name": "InverseKinematicsController",
                "command_input_limits": None,
            },
            "arm_right": {
                "name": "InverseKinematicsController",
                "command_input_limits": None,
            },
            "gripper_left": {
                "name": "MultiFingerGripperController",
                "command_input_limits": (0.0, 1.0),
                "mode": "smooth",
            },
            "gripper_right": {
                "name": "MultiFingerGripperController",
                "command_input_limits": (0.0, 1.0),
                "mode": "smooth",
            },
        },
    }


# ── Cameras ─────────────────────────────────────────────────────────────────
VIEWER_CAMERA_POS = [-0.7963, -0.8716,  1.2569]
VIEWER_CAMERA_ORN = [0.0930, 0.4701, 0.8610, 0.1704]

EXTERNAL_CAMERA_CONFIGS = {
    "external_sensor_0": {
        "position": VIEWER_CAMERA_POS,
        "orientation": VIEWER_CAMERA_ORN,
        "horizontal_aperture": 30.0,
    },
    "external_sensor_1": {
        "position": [-0.5087745785713196, -3.052588701248169, 0.9984493851661682],
        "orientation": [0.5276271104812622, 0.19144046306610107, 0.2822819948196411, 0.7779955267906189],
        "horizontal_aperture": 30.0,
    },
}

INIT_STATE_PATH = None


def get_task_config() -> TaskConfig:
    return TaskConfig(
        task_name="turn_on_stove",
        use_gpu_dynamics=False,
        enable_transition_rules=False,
        scene_config={
            "type": "InteractiveTraversableScene",
            "scene_model": "Rs_int",
            "include_robots": False,
            "load_task_relevant_only": True,
        },
        robot_name=ROBOT_NAME,
        robot_type=ROBOT_TYPE,
        robot_config=_robot_config(),
        task_objects=TASK_OBJECTS,
        viewer_camera_pos=VIEWER_CAMERA_POS,
        viewer_camera_orn=VIEWER_CAMERA_ORN,
        external_camera_configs=EXTERNAL_CAMERA_CONFIGS,
        target_objects_health_with_links=_VIZ["health_with_links"],
        target_objects_health=[ROBOT_NAME],
        target_objects_temperature=_VIZ["temperature"],
        target_objects_forces=_VIZ["forces"],
        force_keys=["filtered_qs_forces"],
        default_collect_hdf5="demos/behavior1k/teleop_data/turn_on_stove.hdf5",
        default_playback_hdf5="demos/behavior1k/playback_data/turn_on_stove_playback.hdf5",
        default_video_dir="demos/behavior1k/playback_videos/turn_on_stove",
    )


_U_XY = 0.03
_U_YAW = 0.12
_U_ARM = 0.07
_U_POT_XY = 0.03


def _heat_source_world_anchor(heat) -> th.Tensor:
    """
    Mirror ``HeatSourceOrSink._update`` anchor so task geometry matches Thermal coupling sphere center.

    OmniGibson source (non-``requires_inside``):
        ``link.aabb_center`` if ``link == _default_link`` else ``link.get_position_orientation()[0]``.
    """
    link = heat.link
    default_link = heat._default_link
    if link == default_link:
        return link.aabb_center
    pos, _ = link.get_position_orientation()
    return pos


def _pot_geometry_center_world(saucepot):
    """Return (bbox_center_xyz, bbox_lower_corner_xyz); fall back to object root pose if no ``AABB``."""
    try:
        if object_states.AABB in saucepot.states:
            lb, ub = saucepot.states[object_states.AABB].get_value()
            lb_d = lb.to(dtype=th.float32)
            ub_d = ub.to(dtype=th.float32)
            return (lb_d + ub_d) * 0.5, lb_d
    except Exception:
        pass
    pw, _ = saucepot.get_position_orientation()
    p = pw.to(dtype=th.float32)
    return p, p


def tighten_stove_joints(env):
    stove = env.scene.object_registry("name", "stove")
    if stove is None or not hasattr(stove, "joints"):
        return
    for joint in stove.joints.values():
        joint.friction = float(_STOVE_JOINT_FRICTION)
        if hasattr(joint, "keep_still"):
            joint.keep_still()
    if hasattr(stove, "keep_still"):
        stove.keep_still()


def _pot_over_active_burner(stove, saucepot):
    """
    True when the knob is ON, ``saucepot`` is kinematically marked ``OnTop`` the stove cooktop,
    the stove's ``HeatSourceOrSink`` is logically active (``get_value()``), and pot geometry sits
    within a tight XY window of the heatsource anchor (same point ``HeatSourceOrSink._update`` uses
    for its overlap sphere) with sensible vertical slack.
    """
    if stove is None or saucepot is None:
        return False

    try:
        if object_states.OnTop not in saucepot.states:
            return False
        if object_states.ToggledOn not in stove.states:
            return False

        knob_on = stove.states[object_states.ToggledOn].get_value()
        placed_on_surface = saucepot.states[object_states.OnTop].get_value(other=stove)

        if not knob_on:
            return False
        if not placed_on_surface:
            return False

        if object_states.HeatSourceOrSink not in stove.states:
            return True

        heat = stove.states[object_states.HeatSourceOrSink]
        # ``get_value``: toggles / door requirements satisfied so the element is logically “on”.
        if not heat.get_value():
            return False

        anchor = _heat_source_world_anchor(heat).to(dtype=th.float32)
        pot_ctr, pot_bot = _pot_geometry_center_world(saucepot)
        pot_ctr_f = pot_ctr.to(dtype=th.float32)
        pot_bot_f = pot_bot.to(dtype=th.float32)
        xy_sep = float(th.norm(pot_ctr_f[:2] - anchor[:2]))
        if xy_sep > float(_TASK_POT_VS_BURNER_XY_MAX_M):
            print("pot not over active burner")
            return False
        else:
            print("pot in xy window of active burner")
            return True
        # min_bot_z = float(anchor[2]) + float(_TASK_POT_BOTTOM_ABOVE_BURNER_Z_MIN_M)
        # return float(pot_bot_f[2]) >= min_bot_z
    except (KeyError, AttributeError):
        return False


def reset(env):
    """Small uniform noise on robot pose / arm joints + ``saucepot`` spawn offset; settle briefly."""

    if INIT_STATE_PATH is not None:
        with open(INIT_STATE_PATH, "rb") as f:
            state_flat_array = pickle.load(f)
        og.sim.load_state(state_flat_array, serialized=True)

    saucepot = env.scene.object_registry("name", "saucepot")
    if saucepot is not None:
        pos, orn = saucepot.get_position_orientation()
        pos = pos.clone()
        pos[0] += float(np.random.uniform(-_U_POT_XY, _U_POT_XY))
        pos[1] += float(np.random.uniform(-_U_POT_XY, _U_POT_XY))
        saucepot.set_position_orientation(pos, orn)
        if hasattr(saucepot, "keep_still"):
            saucepot.keep_still()

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
    aci = robot.arm_control_idx
    if hasattr(aci, "items"):
        for arm_key, idx in aci.items():
            if arm_key == "combined":
                continue
            u = (th.rand(len(idx), device=q.device, dtype=q.dtype) * 2 - 1) * _U_ARM
            q[idx] = q[idx] + u
    robot.set_joint_positions(q)
    robot.set_joint_velocities(th.zeros(robot.n_dof, device=q.device, dtype=q.dtype))
    robot.keep_still()

    tighten_stove_joints(env)

    for _ in range(10):
        og.sim.step()


def task_completion_check(env):
    stove = env.scene.object_registry("name", "stove")
    saucepot = env.scene.object_registry("name", "saucepot")
    robot = env.robots[0]

    if stove is None or robot is None:
        return False

    pot_ok = _pot_over_active_burner(stove, saucepot)
    gripper_far = gripper_far_from_object(
        robot,
        stove,
        threshold=0.5,
        arm=TELEOP_ARM_FOR_STOVE,
    )
    return pot_ok and gripper_far
