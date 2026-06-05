"""
Task configuration for **turn_on_faucet**.

Scene : ``Rs_int`` (interactive traversable), sink near the workspace center.

Lone object: furniture_sink ``czyfhq`` at a **large scale** in the middle of the navigable area.
Spawn with ``fixed_base=False`` so it settles on the floor, then ``reset`` locks it with ``fixed_base=True``.

Robot: **FrankaMounted** ``franka0`` on the floor **behind** the sink ( −Y ), facing the fixture so the
arm can reach the lever. ``enable_transition_rules`` must stay **True** so ``ToggledOn`` can follow
physical / affordance interactions (keyboard teleop toggle on the faucet).
"""

from __future__ import annotations

import numpy as np
import omnigibson as og
import torch as th
from omnigibson.object_states import ToggledOn
from omnigibson.utils import transform_utils as T

from oopsiebench.envs.behavior1k.base import TaskConfig
from oopsiebench.envs.behavior1k.spatial_checks import gripper_far_from_object

ROBOT_NAME = "franka0"
ROBOT_TYPE = "FrankaMounted"

# Enlarged sink (user request: scale *up* vs default 1,1,1).
SINK_SCALE = [1, 1, 1.25]

# Near the middle of Rs_int’s main walkable pocket (floor plane ~ z = 0 for mounted robots).
SINK_POSITION = [-1.05, -2.05, 0.02]
SINK_ORIENTATION = [0.0, 0.0, 0.0, 1.0]

# Mounted Franka behind the faucet approach (−Y), rotated to face +Y toward the sink / lever.
ROBOT_POSITION = [-0.45, -2.05, 0.0]
ROBOT_ORIENTATION = [0.0, 0.0, 1.0, 0.0]

# ── Task objects ─────────────────────────────────────────────────────────

TASK_OBJECTS = {
    "sink": {
        "type": "DatasetObject",
        "name": "sink",
        "category": "furniture_sink",
        "model": "czyfhq",
        "position": SINK_POSITION,
        "orientation": SINK_ORIENTATION,
        "scale": SINK_SCALE,
        # Settle unconstrained first; ``reset`` enables fixed_base after physics steps.
        "fixed_base": False,
    },
}

# ── Cameras ──────────────────────────────────────────────────────────────

# Eye/orientation from teleop TAB (`viewer_camera`).
VIEWER_CAMERA_POS = [-0.7933, -1.0910,  1.8558]
VIEWER_CAMERA_ORN = [-0.0039,  0.4483,  0.8938, -0.0079]

EXTERNAL_CAMERA_CONFIGS = {
    "external_sensor_0": {
        "position": VIEWER_CAMERA_POS,
        "orientation": VIEWER_CAMERA_ORN,
        "horizontal_aperture": 18.0,
    },
}


# ── Reset ───────────────────────────────────────────────────────────────

# Small XY nudges on later teleop reset retries (meters) to break sink/robot overlap.
_RESET_SINK_XY_JITTER = (
    (0.0, 0.0),
    (0.02, 0.0),
    (-0.02, 0.0),
    (0.0, 0.02),
    (0.0, -0.02),
    (0.015, -0.015),
)

_U_XY = 0.05
_U_YAW = 0.12
_U_ARM = 0.2


def _nudge_sink_for_retry(sink, attempt: int) -> None:
    if attempt <= 0:
        return
    dx, dy = _RESET_SINK_XY_JITTER[attempt % len(_RESET_SINK_XY_JITTER)]
    if dx == 0.0 and dy == 0.0:
        return
    try:
        sink.fixed_base = False
        pos, orn = sink.get_position_orientation()
        pos = pos.clone()
        pos[0] += dx
        pos[1] += dy
        sink.set_position_orientation(pos, orn)
    except Exception:
        pass


def _try_set_faucet_off(env) -> None:
    """Start episodes with water OFF when ``ToggledOn`` exists."""
    try:
        from omnigibson.object_states import ToggledOn
    except Exception:
        return

    sink = env.scene.object_registry("name", "sink")
    if sink is None or not hasattr(sink, "states"):
        return

    try:
        st = sink.states.get(ToggledOn)
        if st is not None:
            st.set_value(False)
    except Exception:
        pass


def reset(env):
    """Light settle, lock sink base, controllers sync + faucet OFF."""
    if not getattr(env, "robots", None):
        return
    attempt = int(getattr(env, "_reset_settle_attempt", 0))
    robot = env.robots[0]
    sink = env.scene.object_registry("name", "sink") if getattr(env, "scene", None) else None

    if sink is not None:
        _nudge_sink_for_retry(sink, attempt)

    robot.keep_still()
    if sink is not None:
        try:
            sink.keep_still()
        except Exception:
            pass

    for _ in range(15):
        og.sim.step()

    if sink is not None:
        try:
            sink.fixed_base = True
            sink.keep_still()
        except Exception:
            pass

    _try_set_faucet_off(env)

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

    robot.keep_still()
    robot.set_joint_velocities(th.zeros(robot.n_dof))
    for ctrl_name in ("arm_0", "gripper_0"):
        ctrl = robot.controllers.get(ctrl_name)
        if ctrl is not None:
            ctrl.reset()


def task_completion_check(env):
    sink = env.scene.object_registry("name", "sink")
    if sink is None or not getattr(env, "robots", None) or ToggledOn not in sink.states:
        return False
    faucet_on = bool(sink.states[ToggledOn].get_value())
    return faucet_on and gripper_far_from_object(env.robots[0], sink, threshold=0.75)


def get_task_config() -> TaskConfig:
    return TaskConfig(
        task_name="turn_on_faucet",

        use_gpu_dynamics=True,
        enable_transition_rules=True,
        physics_frequency=120.0,

        scene_config={
            "type": "InteractiveTraversableScene",
            "scene_model": "Rs_int",
            "include_robots": False,
            "load_task_relevant_only": True,
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
