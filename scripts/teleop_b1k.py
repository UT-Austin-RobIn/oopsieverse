#!/usr/bin/env python3
"""
Keyboard teleop for Behavior1k-style OmniGibson tasks.

Creates the env from the TaskConfig, loads the initial state from a pickle,
then lets you teleoperate (focus the viewer).

Usage:
    # Simple teleop (1 episode, saves video on quit)
    python scripts/teleop_b1k.py --task_name shelve_item

    # Collect 5 episodes to HDF5 for later playback
    python scripts/teleop_b1k.py --task_name shelve_item \\
        --collect_hdf5_path demos/behavior1k/teleop_data/shelve_item.hdf5 --n_episodes 5

Keys:
    TAB       — end current episode (resets env, starts next episode)
    ESC       — quit (saves completed episodes to HDF5; discards in-progress)
    BACKSPACE — discard current trajectory and start over (no save, no count)
    S         — save serialized state to init_states and breakpoint
"""

from __future__ import annotations

import argparse
import importlib
import sys
import os
import pickle
from datetime import datetime
from typing import List, Optional

os.environ.setdefault("CARB_LOG_CHANNELS", "omni.physx.plugin=off")

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import torch as th
import omnigibson as og
import omnigibson.lazy as lazy
from omnigibson.macros import gm
from omnigibson.utils.ui_utils import KeyboardRobotController
from omnigibson.envs.data_wrapper import flatten_obs
from omnigibson.controllers.controller_base import IsGraspingState

from damagesim.omnigibson.damageable_env import (
    OGDamageableEnvironment,
    OGDamageableDataCollectionWrapper,
)
from damagesim.utils.visualization import (
    save_rgb_camera_video,
    save_rgb_health_video_with_overlay,
    save_rgb_force_video,
    save_rgb_temperature_video,
    apply_playback_health_tint_to_frames,
)
from damagesim.utils.misc_utils import setup_viewport_layout, setup_robot_eef_visualization


# --task_name picks which module to import from this package
TASK_CONFIG_PACKAGE = "oopsiebench.envs.behavior1k"

# Default task when --task_name omitted: Rs_int, damage = all except skip_categories

TASK_REGISTRY = {
    "default": "default",
    "shelve_item": "shelve_item",
    "add_firewood": "add_firewood",
    "firewood": "add_firewood",
    "pour_water": "pour_water",
    "open_drawer": "open_drawer",
    "wipe_counter": "wipe_counter",
    "nav_to_table": "nav_to_table",
    "pick_egg": "pick_egg",
    "fill_bowl": "fill_bowl",
    "place_plate": "place_plate",
    "turn_on_faucet": "turn_on_faucet",
    "heat_saucepot": "heat_saucepot",
    "open_single_door": "open_single_door",
    "food_in_microwave": "food_in_microwave",
}

# Global variables for teleop
QUIT_REQUESTED = [False]
EPISODE_DONE = [False]
DISCARD_REQUESTED = [False]

VIDEO_CAMERA_TYPE = "external"
VIDEO_CAMERA_NAME = "external_sensor0"


def load_task_config(task_name: str):
    """Import the task config module and return its (TaskConfig, module)."""
    if task_name not in TASK_REGISTRY:
        available = ", ".join(sorted(TASK_REGISTRY.keys()))
        raise ValueError(f"Unknown task '{task_name}'. Available tasks: {available}")
    mod = importlib.import_module(f"{TASK_CONFIG_PACKAGE}.{TASK_REGISTRY[task_name]}")
    return mod.get_task_config(), mod


def save_state_to_pkl(task_name: str):
    state = og.sim.dump_state(serialized=True)
    init_dir = os.path.join(
        _REPO_ROOT, "oopsiebench", "envs", "behavior1k", "init_states",
    )
    os.makedirs(init_dir, exist_ok=True)
    path = os.path.join(init_dir, f"{task_name}_temp.pkl")
    with open(path, "wb") as f:
        pickle.dump(state, f)
    print(f"[teleop] Saved serialized state to {path}")
    breakpoint()


def load_state_from_pkl(env, task_name: str, task_module=None, *, run_task_reset: bool = True):
    # Used by some task modules (e.g. wipe_counter) to decide whether to run
    # runtime-only setup after restoring a saved sim state.
    setattr(env, "_teleop_loaded_from_pkl", False)

    init_dir = os.path.join(
        _REPO_ROOT, "oopsiebench", "envs", "behavior1k", "init_states",
    )
    path = os.path.join(init_dir, f"{task_name}.pkl")
    if not os.path.isfile(path):
        print(f"[teleop] No init-state pickle at {path}; using config-defined reset instead.")
        env.reset()
        if task_module is not None and hasattr(task_module, "reset") and callable(task_module.reset):
            task_module.reset(env)
        return

    with open(path, "rb") as f:
        state = pickle.load(f)

    env.reset()
    scene_file = getattr(env, "scene_file", None)
    if scene_file is not None:
        env.scene.restore(scene_file, update_initial_file=True)

    if not og.sim.is_playing():
        og.sim.play()
    og.sim.load_state(state, serialized=True)
    setattr(env, "_teleop_loaded_from_pkl", True)
    for _ in range(10):
        og.sim.step()

    if run_task_reset and task_module is not None and hasattr(task_module, "reset") and callable(task_module.reset):
        task_module.reset(env)


def build_env_config(task_cfg):
    scene_config = dict(task_cfg.scene_config)
    if "type" not in scene_config:
        scene_config["type"] = "InteractiveTraversableScene"

    return {
        "env": {
            "action_frequency": task_cfg.action_frequency,
            "rendering_frequency": task_cfg.rendering_frequency,
            "physics_frequency": task_cfg.physics_frequency,
        },
        "scene": scene_config,
        "robots": [dict(task_cfg.robot_config)],
        "objects": [dict(obj) for obj in task_cfg.task_objects.values()],
        "task": {"type": "DummyTask", "activity_name": task_cfg.task_name},
    }


def build_external_sensors_config(task_cfg, robot_name: str, robot_type: str,
                                  image_height: int = 1280, image_width: int = 1280):
    """
    Build the list-of-dicts external_sensors config so the env includes external
    cameras (e.g. from task_cfg.external_camera_configs). Same structure as in
    scripts/playback.py so HDF5 contains the same external camera obs.
    """
    sensors = []
    for name, cam_cfg in task_cfg.external_camera_configs.items():
        idx = name.split("_")[-1]
        prim_path = (
            f"/controllable__damageable{robot_type.lower()}"
            f"__{robot_name}/base_link/external_sensor{idx}"
        )
        sensors.append({
            "sensor_type": "VisionSensor",
            "name": f"external_sensor{idx}",
            "relative_prim_path": prim_path,
            "modalities": ["rgb", "seg_instance"],
            "sensor_kwargs": {
                "image_height": image_height,
                "image_width": image_width,
                "horizontal_aperture": cam_cfg.get("horizontal_aperture", 15.0),
            },
            "position": th.tensor(cam_cfg["position"], dtype=th.float32),
            "orientation": th.tensor(cam_cfg["orientation"], dtype=th.float32),
            "pose_frame": cam_cfg.get("frame", "world"),
        })
    return sensors


def capture_rgb(camera="viewer", env=None, *, return_seg: bool = False):
    """RGB frame (H, W, 3) uint8. camera: viewer | external_sensor* | robot sensor name | sensor object."""
    if hasattr(camera, "get_obs"):
        sensor = camera
    elif camera in ("viewer", "viewer_camera"):
        sensor = og.sim.viewer_camera
    else:
        sensor = None
        node = env
        while node is not None and sensor is None:
            sensor = (getattr(node, "_external_sensors", None) or {}).get(camera)
            for r in getattr(node, "robots", None) or ():
                sensor = sensor or r.sensors.get(camera)
            node = getattr(node, "env", None)
        if sensor is None:
            raise ValueError(f"Unknown camera: {camera}")
    obs, sensor_info = sensor.get_obs()
    frame = obs["rgb"]
    if isinstance(frame, th.Tensor):
        frame = frame.cpu().numpy()
    else:
        frame = np.array(frame)
    if frame.shape[-1] == 4:
        frame = frame[:, :, :3]
    if frame.dtype != np.uint8:
        frame = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)

    if not return_seg:
        return frame

    seg = obs.get("seg_instance")
    if seg is None:
        raise ValueError(f"Camera {camera} did not return seg_instance (needed for --playback_health_tint)")
    if isinstance(seg, th.Tensor):
        seg = seg.cpu().numpy()
    else:
        seg = np.array(seg)
    return frame, seg, sensor_info


_VIEWER_CAM_STEP = 0.075


def _quat_rotate_vec(q: th.Tensor, v: th.Tensor) -> th.Tensor:
    """Rotate vector *v* by unit quaternion *q* (x, y, z, w)."""
    qv = q[:3]
    w = q[3]
    uv = th.cross(qv, v)
    uuv = th.cross(qv, uv)
    return v + 2.0 * (w * uv + uuv)


def viewer_camera_print_pose_and_break() -> None:
    cam = og.sim.viewer_camera
    pos, orn = cam.get_position_orientation()
    pe = pos.flatten().tolist()[:3]
    oe = orn.flatten().tolist()[:4]
    print("[viewer_camera] position:", pe)
    print("[viewer_camera] orientation (x,y,z,w):", oe)
    breakpoint()


def viewer_camera_nudge(forward: float, right: float, up_world: float) -> None:
    """Translate viewer camera in its horizontal plane (+world Z via Q/E)."""
    cam = og.sim.viewer_camera
    pos, orn = cam.get_position_orientation()
    pos = pos.flatten()[:3].to(dtype=th.float32)
    orn = orn.flatten()[:4].to(dtype=th.float32)
    look_local = th.tensor([0.0, 0.0, -1.0], dtype=th.float32)
    fwd_w = _quat_rotate_vec(orn, look_local)
    fwd_w = fwd_w.clone()
    fwd_w[2] = 0.0
    fn = fwd_w.norm()
    if fn > 1e-6:
        fwd_w = fwd_w / fn
    else:
        fwd_w = th.tensor([1.0, 0.0, 0.0], dtype=th.float32)
    right_w = th.cross(th.tensor([0.0, 0.0, 1.0], dtype=th.float32), fwd_w)
    rn = right_w.norm()
    if rn > 1e-6:
        right_w = right_w / rn
    up_w = th.tensor([0.0, 0.0, 1.0], dtype=th.float32)
    delta = (
        fwd_w * float(forward) * _VIEWER_CAM_STEP
        + right_w * float(right) * _VIEWER_CAM_STEP
        + up_w * float(up_world) * _VIEWER_CAM_STEP
    )
    cam.set_position_orientation(position=pos + delta, orientation=orn)


def save_video(teleop_frames, teleop_health_records, target_objects_for_overlay,
                task_cfg, fps=30, overlay_position="bottom_center", overlay_layout="column",
                teleop_force_records=None, teleop_temperature_records=None,
                playback_health_tint: bool = False,
                teleop_seg_frames: Optional[List[np.ndarray]] = None,
                teleop_obs_info_list: Optional[List[dict]] = None,
                video_camera_type: str = VIDEO_CAMERA_TYPE,
                video_camera_name: str = VIDEO_CAMERA_NAME):
    """Save the collected frames as an MP4 (with health overlay if available).
    Also saves force / temperature side-by-side plots when configured and data exists."""
    if not teleop_frames:
        return
    video_dir = os.path.join(
        _REPO_ROOT, "demos", "behavior1k", "teleop_videos",
    )
    os.makedirs(video_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(video_dir, f"teleop_{task_cfg.task_name}_{timestamp}")
    imgs = np.array(teleop_frames)

    health_for_overlay = None
    if target_objects_for_overlay and teleop_health_records:
        health_for_overlay = {}
        for name in target_objects_for_overlay:
            arr = teleop_health_records.get(name)
            if arr is None:
                continue
            arr = np.array(arr, dtype=np.float64)
            if len(arr) < len(teleop_frames):
                arr = np.concatenate([[100.0], arr])
            if len(arr) > len(teleop_frames):
                arr = arr[: len(teleop_frames)]
            health_for_overlay[name] = arr

    if playback_health_tint and health_for_overlay and teleop_seg_frames and teleop_obs_info_list:
        imgs = apply_playback_health_tint_to_frames(
            imgs,
            teleop_seg_frames,
            teleop_obs_info_list,
            list(health_for_overlay.keys()),
            health_for_overlay,
            camera_type=video_camera_type,
            camera_name=video_camera_name,
        )

    if health_for_overlay and len(health_for_overlay) > 0:
        save_rgb_health_video_with_overlay(
            output_path,
            imgs,
            target_objects=list(health_for_overlay.keys()),
            health=health_for_overlay,
            fps=fps,
            position=overlay_position,
            layout=overlay_layout,
        )
        print(f"[teleop] Saved {len(teleop_frames)} frames with health overlay to {output_path}.mp4")
    else:
        save_rgb_camera_video(output_path, imgs, fps=fps)
        print(f"[teleop] Saved {len(teleop_frames)} frames to {output_path}.mp4")

    # Force plot video (when applicable)
    target_objects_forces = getattr(task_cfg, "target_objects_forces", None) or []
    force_keys = getattr(task_cfg, "force_keys", None) or ["filtered_qs_forces"]
    if target_objects_forces and teleop_force_records:
        forces = {}
        for obj_name in target_objects_forces:
            obj_data = teleop_force_records.get(obj_name, {})
            forces[obj_name] = {fk: list(obj_data.get(fk, [])) for fk in force_keys}
        # Trim/pad to match frame count
        n_frames = len(teleop_frames)
        has_data = False
        for obj_name in target_objects_forces:
            for fk in force_keys:
                arr = forces.get(obj_name, {}).get(fk, [])
                if len(arr) > n_frames:
                    forces[obj_name][fk] = arr[:n_frames]
                elif len(arr) < n_frames:
                    forces[obj_name][fk] = arr + [0.0] * (n_frames - len(arr))
                if len(forces[obj_name][fk]) == n_frames:
                    has_data = True
        if has_data and forces:
            forces_path = output_path + "_forces.mp4"
            save_rgb_force_video(
                output_video_path=forces_path,
                imgs=imgs,
                target_objects=target_objects_forces,
                data=forces,
                forces_to_plot=force_keys,
                fps=fps,
            )
            print(f"[teleop] Saved force plot video to {forces_path}")

    # Temperature plot (thermal evaluator outputs on robot links)
    target_objects_temperature = getattr(task_cfg, "target_objects_temperature", None) or []
    temperature_plot_keys = getattr(task_cfg, "temperature_plot_keys", None) or ["temperature"]
    if target_objects_temperature and teleop_temperature_records:
        temps = {}
        for obj_name in target_objects_temperature:
            obj_data = teleop_temperature_records.get(obj_name, {})
            temps[obj_name] = {k: list(obj_data.get(k, [])) for k in temperature_plot_keys}
        n_frames = len(teleop_frames)
        has_temp = False
        for obj_name in target_objects_temperature:
            for k in temperature_plot_keys:
                arr = temps.get(obj_name, {}).get(k, [])
                if len(arr) > n_frames:
                    temps[obj_name][k] = arr[:n_frames]
                elif len(arr) < n_frames:
                    temps[obj_name][k] = arr + [0.0] * (n_frames - len(arr))
                if len(temps[obj_name][k]) == n_frames:
                    has_temp = True
        if has_temp and temps:
            temp_path = output_path + "_temperature.mp4"
            save_rgb_temperature_video(
                output_video_path=temp_path,
                imgs=imgs,
                target_objects=target_objects_temperature,
                data=temps,
                temperature_keys=tuple(temperature_plot_keys),
                fps=fps,
            )
            print(f"[teleop] Saved temperature plot video to {temp_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Keyboard teleop for Behavior1k tasks.")
    p.add_argument("--task_name", type=str, default="default",
                   help="Task name (e.g. shelve_item, pour_water, add_firewood).")
    p.add_argument("--collect_hdf5_path", type=str, default=None,
                   help="If specified, save teleop demos to this HDF5 for later playback. Otherwise resorts to a default path")
    p.add_argument("--live_feedback", action="store_true",
                   help="Show live health bars and object coloring during teleop.")
    p.add_argument("--save_video", action="store_true",
                   help="Save an MP4 of the viewer at exit (default: False). If not set, sim optimization is used and no video is saved.")
    p.add_argument("--save_obs_to_hdf5", action="store_true",
                   help="Typically images are saved only during playback. But, use this if you want to save images during teleop.")
    p.add_argument("--n_episodes", type=int, default=3,
                   help="Number of teleop episodes to run (default: 1).")
    p.add_argument("--skip_hdf5_save", action="store_true",
                   help="Skip saving the HDF5 file (default: False).")
    p.add_argument("--teleop_device", type=str, default="keyboard",
                   help="Teleop device (default: keyboard).")
    p.add_argument("--overlay_links", action="store_true",
                   help="Show one health bar per link in saved video (default: one per object).")
    p.add_argument(
        "--overlay_position",
        type=str,
        default="bottom_left",
        choices=[ "bottom_left", "bottom_center", "bottom_right", "top_left", "top_center", "top_right", "center"],
        help="Position of the health bar overlay in the saved video (default: bottom_center).",
    )
    p.add_argument(
        "--overlay_layout",
        type=str,
        default="column",
        choices=["column", "row"],
        help="Layout of health bars in saved video: column (default) or row.",
    )
    p.add_argument(
        "--playback_health_tint",
        action="store_true",
        help=(
            "Disable live diffuse_tint coloring and apply playback-style seg-mask "
            "red tint when saving video (requires --save_video)."
        ),
    )
    return p.parse_args()


MAX_RESET_RETRIES = 5
def discard_in_progress_traj(env):
    """Drop the current in-memory trajectory without writing to HDF5."""
    if hasattr(env, "discard_current_traj"):
        env.discard_current_traj()


def reset_env(env, task_cfg, task_mod):
    """
    Reset the environment to the initial state.
    """

    # Set the viewer camera position and orientation
    if task_cfg.viewer_camera_pos is not None and task_cfg.viewer_camera_orn is not None:
        og.sim.viewer_camera.set_position_orientation(
            position=th.tensor(task_cfg.viewer_camera_pos, dtype=th.float32),
            orientation=th.tensor(task_cfg.viewer_camera_orn, dtype=th.float32),
        )

    env.reset()
    # Call task specific reset
    if task_mod is not None and hasattr(task_mod, "reset") and callable(task_mod.reset):
        task_mod.reset(env)

    env._reset_damage_tracking()
    for _ in range(5): og.sim.step()
    env_health = env.get_env_health()
    damaged = {k: v for k, v in (env_health or {}).items() if v < 100.0}
    if damaged:
        print(f"health not clean: {damaged}")


class TeleopWrapper:
    """
    Wrapper for the teleop system to control the teleop session and the robot controller.
    """
    def __init__(self, env, robot, task_cfg, task_mod, init_grasp=False, **kwargs):
        self.env = env
        self.robot = robot
        self.task_cfg = task_cfg
        self.task_mod = task_mod
        self.init_grasp = init_grasp
        self.teleop_device = kwargs["teleop_device"]
        self.save_video = kwargs["save_video"]
        self.save_to_hdf5 = not kwargs["skip_hdf5_save"]

        # For video saving
        self.teleop_frames = []
        self.teleop_seg_frames = []
        self.teleop_obs_info_list = []
        self.teleop_health_records = {}
        self.teleop_force_records = {}
        self.teleop_temperature_records = {}
        self.overlay_links = kwargs["overlay_links"]
        self.overlay_position = kwargs["overlay_position"]
        self.overlay_layout = kwargs["overlay_layout"]
        self.playback_health_tint = kwargs.get("playback_health_tint", False)
        self.video_camera_type = VIDEO_CAMERA_TYPE
        self.video_camera_name = VIDEO_CAMERA_NAME
        self.target_objects_forces = getattr(self.task_cfg, "target_objects_forces", None) or []
        self.force_keys = getattr(self.task_cfg, "force_keys", None) or ["filtered_qs_forces"]
        self.target_objects_temperature = getattr(self.task_cfg, "target_objects_temperature", None) or []
        self.temperature_plot_keys = getattr(self.task_cfg, "temperature_plot_keys", None) or ["temperature"]
        self.setup_video_saving()

        # setup teleop interface
        self.teleop_interface = self.setup_teleop_interface()
        # Session keys + viewer WASD: same KeyboardRobotController as robot when using keyboard.
        self.keyboard_interface = self.setup_keyboard_interface()


    def setup_teleop_interface(self):
        if self.teleop_device == "keyboard":
            teleop_interface = KeyboardRobotController(robot=self.robot)
        elif self.teleop_device == "spacemouse":
            # Telemoma (optional dependency) — only required for non-keyboard teleop.
            from telemoma.configs.base_config import teleop_config
            from omnigibson.utils.teleop_utils import TeleopSystem

            from damagesim.omnigibson.telemoma_spacemouse import register_scaled_spacemouse
            from damagesim.omnigibson.og_teleop import patch_holonomic_teleop_trunk_and_camera

            register_scaled_spacemouse()

            arm_teleop_method = self.teleop_device
            base_teleop_method = self.teleop_device
            # # Franka config: uses arm_0 instead of arm_left/arm_right
            teleop_config.arm_0_controller = arm_teleop_method
            # # Tiago config:
            teleop_config.arm_left_controller = arm_teleop_method
            teleop_config.arm_right_controller = arm_teleop_method
            teleop_config.base_controller = base_teleop_method
            teleop_config.torso_controller = base_teleop_method
            teleop_config.interface_kwargs["spacemouse"] = {
                "arm_speed_scaledown": 0.01,
                "base_speed_scaledown": 0.03,
            }
            patch_holonomic_teleop_trunk_and_camera(self.robot)
            teleop_interface = TeleopSystem(config=teleop_config, robot=self.robot, show_control_marker=False)
            teleop_interface.start()
        else:
            raise ValueError(f"Unknown teleop device: {self.teleop_device}")
        return teleop_interface
    
    def reset_teleop_wrapper(self):
        if self.teleop_device == "keyboard":
            self.teleop_interface.persistent_gripper_action[self.teleop_interface.binary_grippers[0]] = -1.0 if self.init_grasp else 1.0
        elif self.teleop_device == "spacemouse":
            if self.init_grasp:
                self.teleop_interface.interfaces["spacemouse"].actions["left"][-1] = 0.0
                self.teleop_interface.interfaces["spacemouse"].actions["right"][-1] = 0.0
            self.last_grasp_action = 0.0 if self.init_grasp else 1.0
        else:
            raise ValueError(f"Unknown teleop device: {self.teleop_device}")

        # Reset for video saving
        self.teleop_frames = []
        self.teleop_seg_frames = []
        self.teleop_obs_info_list = []
        self.teleop_health_records = {}
        self.teleop_force_records = {}
        self.teleop_temperature_records = {}

    def setup_keyboard_interface(self):
        # Keyboard teleop: reuse the robot KeyboardRobotController so custom mappings fire each step.
        # SpaceMouse: separate lightweight keyboard controller for session keys only.
        if self.teleop_device == "keyboard":
            keyboard_interface = self.teleop_interface
        else:
            keyboard_interface = KeyboardRobotController(robot=self.robot)

        def save_state_and_break():
            save_state_to_pkl(self.task_cfg.task_name)

        def on_esc():
            QUIT_REQUESTED[0] = True

        def on_episode_done():
            EPISODE_DONE[0] = True

        def on_backspace():
            DISCARD_REQUESTED[0] = True

        def do_reset():
            reset_env(self.env, self.task_cfg, self.task_mod)

        Ki = lazy.carb.input.KeyboardInput

        keyboard_interface.register_custom_keymapping(
            key=lazy.carb.input.KeyboardInput.S,
            description="Save serialized state to init_states and breakpoint",
            callback_fn=save_state_and_break,
        )
        keyboard_interface.register_custom_keymapping(
            key=Ki.ESCAPE,
            description="Quit immediately and save video",
            callback_fn=on_esc,
        )
        keyboard_interface.register_custom_keymapping(
            key=Ki.TAB,
            description="Print viewer_camera pose and breakpoint",
            callback_fn=viewer_camera_print_pose_and_break,
        )
        keyboard_interface.register_custom_keymapping(
            key=Ki.K,
            description="End current episode (reset & start next)",
            callback_fn=on_episode_done,
        )
        keyboard_interface.register_custom_keymapping(
            key=Ki.BACKSPACE,
            description="Discard current trajectory and start over (no save)",
            callback_fn=on_backspace,
        )

        keyboard_interface.register_custom_keymapping(
            key=Ki.W,
            description="Viewer camera forward",
            callback_fn=lambda: viewer_camera_nudge(1.0, 0.0, 0.0),
        )
        keyboard_interface.register_custom_keymapping(
            key=Ki.S,
            description="Viewer camera back",
            callback_fn=lambda: viewer_camera_nudge(-1.0, 0.0, 0.0),
        )
        keyboard_interface.register_custom_keymapping(
            key=Ki.A,
            description="Viewer camera left",
            callback_fn=lambda: viewer_camera_nudge(0.0, -1.0, 0.0),
        )
        keyboard_interface.register_custom_keymapping(
            key=Ki.D,
            description="Viewer camera right",
            callback_fn=lambda: viewer_camera_nudge(0.0, 1.0, 0.0),
        )
        keyboard_interface.register_custom_keymapping(
            key=Ki.Q,
            description="Viewer camera down (world)",
            callback_fn=lambda: viewer_camera_nudge(0.0, 0.0, -1.0),
        )
        keyboard_interface.register_custom_keymapping(
            key=Ki.E,
            description="Viewer camera up (world)",
            callback_fn=lambda: viewer_camera_nudge(0.0, 0.0, 1.0),
        )
        return keyboard_interface
    
    def get_action(self):
        if self.teleop_device == "keyboard":
            action, _ = self.teleop_interface.get_teleop_action()
        elif self.teleop_device == "spacemouse":
            action = self.teleop_interface.get_action(self.teleop_interface.get_obs())
        else:
            raise ValueError(f"Unknown teleop device: {self.teleop_device}")
        return action

    def setup_video_saving(self):
        if self.overlay_links:
            self.target_objects_for_overlay = (
                getattr(self.task_cfg, "target_objects_health_with_links", None)
                or getattr(self.task_cfg, "target_objects_health", None)
                or []
            )
        else:
            self.target_objects_for_overlay = (
                getattr(self.task_cfg, "target_objects_health", None) or []
            )

    def record_health_step(self, health_arr, health_list_link_names):
        """Extract per-step health into teleop_health_records (in-place)."""
        if not health_list_link_names or health_arr is None:
            return
        if hasattr(health_arr, "cpu"):
            health_arr = health_arr.cpu().numpy()
        else:
            health_arr = np.asarray(health_arr)
        
        link_vals = {}
        for idx, link_name in enumerate(health_list_link_names):
            val = float(health_arr[idx]) if idx < len(health_arr) else 100.0
            link_vals[link_name] = max(0.0, min(100.0, val))
        if self.overlay_links:
            for name in self.target_objects_for_overlay:
                if name not in link_vals:
                    continue
                self.teleop_health_records.setdefault(name, []).append(link_vals[name])
        else:
            for obj_name in self.target_objects_for_overlay:
                vals = [v for k, v in link_vals.items() if k.startswith(f"{obj_name}@")]
                per_step = min(vals) if vals else 100.0
                self.teleop_health_records.setdefault(obj_name, []).append(per_step)
    
    def record_forces_step(self, damage_info):
        """Extract per-step forces from damage_info and append to teleop_force_records (in-place)."""
        if not self.target_objects_forces or not self.force_keys or not damage_info:
            return
        for obj_name in self.target_objects_forces:
            parts = obj_name.split("@", 1)
            if len(parts) != 2:
                for fk in self.force_keys:
                    self.teleop_force_records.setdefault(obj_name, {}).setdefault(fk, []).append(0.0)
                continue
            obj_key, link_key = parts
            obj_info = damage_info.get(obj_key, {})
            link_info = obj_info.get(link_key, {})
            mechanical = link_info.get("mechanical", {})
            for fk in self.force_keys:
                val = mechanical.get(fk, 0.0)
                self.teleop_force_records.setdefault(obj_name, {}).setdefault(fk, []).append(val)

    def record_temperature_step(self, damage_info):
        """Append thermal evaluator fields per step (temperature, thresholds, …)."""
        if not self.target_objects_temperature or not self.temperature_plot_keys or not damage_info:
            return
        for qual in self.target_objects_temperature:
            parts = qual.split("@", 1)
            if len(parts) != 2:
                for tk in self.temperature_plot_keys:
                    self.teleop_temperature_records.setdefault(qual, {}).setdefault(tk, []).append(0.0)
                continue
            obj_key, link_key = parts
            obj_info = damage_info.get(obj_key, {})
            link_info = obj_info.get(link_key, {})
            thermal = link_info.get("thermal", {})
            for tk in self.temperature_plot_keys:
                raw = thermal.get(tk)
                try:
                    val = float(raw) if raw is not None else 0.0
                except (TypeError, ValueError):
                    val = 0.0
                self.teleop_temperature_records.setdefault(qual, {}).setdefault(tk, []).append(val)

    def record_step(self, obs, info):
        health_list_link_names = getattr(self.env, "health_list_link_names", None) or []
        health_arr = obs.get("health")
        self.record_health_step(
            health_arr,
            health_list_link_names,
        )
        damage_info = info.get("damage_info", {})
        self.record_forces_step(damage_info)
        self.record_temperature_step(damage_info)
        if self.playback_health_tint:
            frame, seg, sensor_info = capture_rgb(
                camera=self.video_camera_name, env=self.env, return_seg=True,
            )
            self.teleop_frames.append(frame)
            self.teleop_seg_frames.append(seg)
            self.teleop_obs_info_list.append(
                {self.video_camera_type: {self.video_camera_name: sensor_info}}
            )
        else:
            frame = capture_rgb(camera=self.video_camera_name, env=self.env)
            self.teleop_frames.append(frame)

    def on_episode_done(self):
        if self.save_video:
            save_video(
                self.teleop_frames,
                self.teleop_health_records,
                self.target_objects_for_overlay,
                self.task_cfg,
                overlay_position=self.overlay_position,
                overlay_layout=self.overlay_layout,
                teleop_force_records=self.teleop_force_records,
                teleop_temperature_records=self.teleop_temperature_records,
                playback_health_tint=self.playback_health_tint,
                teleop_seg_frames=self.teleop_seg_frames,
                teleop_obs_info_list=self.teleop_obs_info_list,
                video_camera_type=self.video_camera_type,
                video_camera_name=self.video_camera_name,
            )


def main():
    args = parse_args()
    task_cfg, task_mod = load_task_config(args.task_name)

    gm.USE_GPU_DYNAMICS = task_cfg.use_gpu_dynamics
    gm.ENABLE_TRANSITION_RULES = task_cfg.enable_transition_rules

    # =============================================
    # ====== Build the DamageableEnvironment ======
    # =============================================
    env_config = build_env_config(task_cfg)
    save_to_hdf5 = not args.skip_hdf5_save
    
    # External cameras are needed either to save obs into the HDF5, or to record
    # the video (VIDEO_CAMERA_NAME points at an external sensor, e.g. external_sensor0).
    need_external_sensors = (save_to_hdf5 and args.save_obs_to_hdf5) or args.save_video
    if need_external_sensors and getattr(task_cfg, "external_camera_configs", None):
        env_config["env"]["external_sensors"] = build_external_sensors_config(
            task_cfg, task_cfg.robot_name, task_cfg.robot_type,
            image_height=1280, image_width=1280,
        )
    base_env = OGDamageableEnvironment(configs=env_config)

    if save_to_hdf5:
        if args.collect_hdf5_path is None:
            args.collect_hdf5_path = os.path.join(
                _REPO_ROOT, "demos", "behavior1k", "teleop_data", f"{args.task_name}.hdf5"
            )
        os.makedirs(os.path.dirname(args.collect_hdf5_path) or ".", exist_ok=True)
        transition_systems = getattr(task_mod, "TRANSITION_SYSTEMS", ())
        env = OGDamageableDataCollectionWrapper(
            env=base_env,
            output_path=args.collect_hdf5_path,
            only_successes=False,
            save_video=args.save_video,
            save_obs_to_hdf5=args.save_obs_to_hdf5,
            transition_systems=transition_systems,
        )
    else:
        env = base_env
    # ================================================

    # Get the robot
    robot = env.robots[0]

    # Reset the environment to the initial state
    reset_env(env, task_cfg, task_mod)

    setup_viewport_layout()
    setup_robot_eef_visualization(robot, env.scene)

    # Setup teleop wrapper
    init_grasp = robot.is_grasping().value == IsGraspingState.TRUE
    teleop_wrapper = TeleopWrapper(
        env=env,
        robot=robot,
        task_cfg=task_cfg,
        task_mod=task_mod,
        init_grasp=init_grasp,
        **vars(args),
    )       

    # Optional task-specific teleop key bindings (if a task module defines ``register_teleop_keys``).
    if hasattr(task_mod, "register_teleop_keys") and callable(task_mod.register_teleop_keys):
        task_mod.register_teleop_keys(env, teleop_wrapper.teleop_interface)

    # Setup live health visualization if enabled
    if args.live_feedback:
        env.enable_health_visualization()

    n_episodes = args.n_episodes
    completed_episodes = 0

    # Loop through episodes
    while completed_episodes < n_episodes:
        print("\n" + "="*80)
        print(f"[TELEOP] Running episode {completed_episodes + 1}/{n_episodes}…")
        print("Press TAB to end an episode (and save if save_to_hdf5 is True), "
              "ESC to quit (discards in-progress), BACKSPACE to discard and restart.")
        print("Press c to continue")
        print("="*80 + "\n")
        teleop_wrapper.reset_teleop_wrapper()
        breakpoint()
                
        # Loop through steps of the current episode
        while True:
            if teleop_wrapper.teleop_device == "keyboard":
                action, _ = teleop_wrapper.teleop_interface.get_teleop_action()
            else:
                _, _ = teleop_wrapper.keyboard_interface.get_teleop_action()
                action = teleop_wrapper.get_action()
            if QUIT_REQUESTED[0] or EPISODE_DONE[0] or DISCARD_REQUESTED[0]:
                break
            obs, reward, terminated, truncated, info = env.step(action.clone())
            if args.save_video:
                teleop_wrapper.record_step(obs, info)

            if hasattr(task_mod, "task_completion_check") and task_mod.task_completion_check(env):
                for _ in range(20):
                    obs, reward, terminated, truncated, info = env.step(action.clone())
                    if args.save_video:
                        teleop_wrapper.record_step(obs, info)
                print(f"[TELEOP] Task completed. Ending episode.")
                EPISODE_DONE[0] = True
                break

        if QUIT_REQUESTED[0]:
            break

        if DISCARD_REQUESTED[0]:
            print(f"[TELEOP] Discarded episode {completed_episodes + 1}. Resetting and starting over…")
            if save_to_hdf5:
                discard_in_progress_traj(env)
            teleop_wrapper.reset_teleop_wrapper()
            reset_env(env, task_cfg, task_mod)
            DISCARD_REQUESTED[0] = False

        if EPISODE_DONE[0]:
            breakpoint()
            teleop_wrapper.on_episode_done()

            completed_episodes += 1
            print(f"[TELEOP] Episode {completed_episodes}/{n_episodes} finished "
                f"({len(teleop_wrapper.teleop_frames)} frames total).")

            # Flush the episode trajectory to HDF5 if save_to_hdf5
            if save_to_hdf5 and hasattr(env, "flush_current_traj"):
                env.task._success = True
                env.flush_current_traj()

            # Reset for next episode (if more remain)
            if completed_episodes < n_episodes:
                reset_env(env, task_cfg, task_mod)

            EPISODE_DONE[0] = False

    if save_to_hdf5 and hasattr(env, "save_data"):
        if QUIT_REQUESTED[0]:
            discard_in_progress_traj(env)
        env.save_data()
        print(f"[teleop] HDF5 saved to {args.collect_hdf5_path}")

    if args.live_feedback:
        env.disable_health_visualization()

    og.shutdown()


if __name__ == "__main__":
    main()
