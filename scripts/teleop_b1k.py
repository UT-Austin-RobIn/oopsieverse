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
    ESC       — quit immediately (saves video + HDF5 if save_to_hdf5)
    BACKSPACE — discard current trajectory and start over (no save, no count)
    R         — reset to initial state (mid-episode)
    S         — save serialized state to init_states and breakpoint
"""

from __future__ import annotations

import argparse
import importlib
import sys
import os
import pickle
from datetime import datetime

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
from telemoma.configs.base_config import teleop_config
from omnigibson.utils.teleop_utils import TeleopSystem
from omnigibson.controllers.controller_base import IsGraspingState

from damagesim.omnigibson.damageable_env import (
    OGDamageableEnvironment,
    OGDamageableDataCollectionWrapper,
)
from damagesim.utils.visualization import (
    save_rgb_camera_video,
    save_rgb_health_video_with_overlay,
    save_rgb_force_video,
)
from utils.misc_utils import setup_viewport_layout


# --task_name picks which module to import from this package
TASK_CONFIG_PACKAGE = "oopsiebench.envs.behavior1k"

TASK_REGISTRY = {
    "shelve_item": "shelve_item",
    "add_firewood": "add_firewood",
    "firewood": "add_firewood",
    "pour_water": "pour_water",
}

# Global variables for teleop
QUIT_REQUESTED = [False]
EPISODE_DONE = [False]
DISCARD_REQUESTED = [False]


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
        _REPO_ROOT, "resources", "init_states",
    )
    os.makedirs(init_dir, exist_ok=True)
    path = os.path.join(init_dir, f"{task_name}_temp.pkl")
    with open(path, "wb") as f:
        pickle.dump(state, f)
    print(f"[teleop] Saved serialized state to {path}")
    breakpoint()


def load_state_from_pkl(env, task_name: str, task_module=None):
    init_dir = os.path.join(
        _REPO_ROOT, "resources", "init_states",
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
    for _ in range(10):
        og.sim.step()

    if task_module is not None and hasattr(task_module, "reset") and callable(task_module.reset):
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
            "pose_frame": "world",
        })
    return sensors


def capture_viewer_rgb():
    """Capture current viewer camera as RGB numpy array (H, W, 3) uint8."""
    obs, _ = og.sim.viewer_camera.get_obs()
    frame = obs["rgb"]
    if isinstance(frame, th.Tensor):
        frame = frame.cpu().numpy()
    else:
        frame = np.array(frame)
    if frame.shape[-1] == 4:
        frame = frame[:, :, :3]
    if frame.dtype != np.uint8:
        frame = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)
    return frame


def save_video(teleop_frames, teleop_health_records, target_objects_for_overlay,
                task_cfg, fps=30, overlay_position="bottom_center", overlay_layout="column",
                teleop_force_records=None):
    """Save the collected frames as an MP4 (with health overlay if available).
    Also saves force plot video when target_objects_forces is set and force data exists."""
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


def parse_args():
    p = argparse.ArgumentParser(description="Keyboard teleop for Behavior1k tasks.")
    p.add_argument("--task_name", type=str, required=True,
                   help="Task name (e.g. shelve_item, pour_water, add_firewood).")
    p.add_argument("--collect_hdf5_path", type=str, default=None,
                   help="If specified, save teleop demos to this HDF5 for later playback. Otherwise resorts to a default path")
    p.add_argument("--live_feedback", action="store_true",
                   help="Show live health bars and object coloring during teleop.")
    p.add_argument("--save_video", action="store_true",
                   help="Save an MP4 of the viewer at exit (default: False). If not set, sim optimization is used and no video is saved.")
    p.add_argument("--save_obs_to_hdf5", action="store_true",
                   help="Typically images are saved only during playback. But, use this if you want to save images during teleop.")
    p.add_argument("--n_episodes", type=int, default=1,
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
    return p.parse_args()


MAX_RESET_RETRIES = 5
def reset_env(env, task_cfg, task_mod):
    """
    Reset the environment to the initial state.
    """
    # TODO: Check if multiple attempts are needed
    for _attempt in range(1, MAX_RESET_RETRIES + 1):

        # Set the viewer camera position and orientation
        og.sim.viewer_camera.set_position_orientation(
            position=th.tensor(task_cfg.viewer_camera_pos, dtype=th.float32),
            orientation=th.tensor(task_cfg.viewer_camera_orn, dtype=th.float32),
        )

        # Load the initial state from the pickle file if it exists, else just reset the environment
        load_state_from_pkl(env, task_name=task_cfg.task_name, task_module=task_mod)
        for _ in range(10): og.sim.step()

        env._reset_damage_tracking()
        for _ in range(5): og.sim.step()
        env_health = env.get_env_health()
        all_clean = (
            all(h >= 100.0 for h in env_health.values())
            if env_health else True
        )
        if all_clean:
            if _attempt > 1:
                print(f"[teleop] Initial state clean after {_attempt} attempts.")
            break
        
        damaged = {k: v for k, v in (env_health or {}).items() if v < 100.0}
        print(f"[teleop] Initial load attempt {_attempt}/{MAX_RESET_RETRIES}: "
              f"health not clean: {damaged}. Retrying load…")
    else:
        raise RuntimeError(f"[teleop] Initial health max retries reached. All objects are not at 100% health.")


class TeleopWrapper:
    """
    Wrapper for the teleop system to control the teleop session and the robot controller.
    """
    def __init__(self, env, robot, task_cfg, init_grasp=False, **kwargs):
        self.env = env
        self.robot = robot
        self.task_cfg = task_cfg
        self.init_grasp = init_grasp
        self.teleop_device = kwargs["teleop_device"]
        self.save_video = kwargs["save_video"]
        self.save_to_hdf5 = not kwargs["skip_hdf5_save"]

        # For video saving
        self.teleop_frames = []
        self.teleop_health_records = {}
        self.teleop_force_records = {}
        self.overlay_links = kwargs["overlay_links"]
        self.overlay_position = kwargs["overlay_position"]
        self.overlay_layout = kwargs["overlay_layout"]
        self.target_objects_forces = getattr(self.task_cfg, "target_objects_forces", None) or []
        self.force_keys = getattr(self.task_cfg, "force_keys", None) or ["filtered_qs_forces"]
        self.setup_video_saving()

        # setup teleop interface
        self.teleop_interface = self.setup_teleop_interface()
        # setup keyboard interface to control the teleop session
        self.keyboard_interface = self.setup_keyboard_interface()


    def setup_teleop_interface(self):
        if self.teleop_device == "keyboard":
            teleop_interface = KeyboardRobotController(robot=self.robot)
        elif self.teleop_device == "spacemouse":
            # Telemoma
            arm_teleop_method = self.teleop_device
            base_teleop_method = self.teleop_device
            # # Franka config: uses arm_0 instead of arm_left/arm_right
            teleop_config.arm_0_controller = arm_teleop_method
            # # Tiago config:
            teleop_config.arm_left_controller = arm_teleop_method
            teleop_config.arm_right_controller = arm_teleop_method
            teleop_config.base_controller = base_teleop_method
            teleop_config.interface_kwargs["spacemouse"] = {"arm_speed_scaledown": 0.01}
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
        self.teleop_health_records = {}
        self.teleop_force_records = {}

        if self.save_to_hdf5 and hasattr(self.env, "current_traj_history"):
            self.env.current_traj_history = []
    
    def setup_keyboard_interface(self):
        # This is only used for controlling the teleop session and not for the robot controller
        keyboard_interface = KeyboardRobotController(robot=self.robot)

        def save_state_and_break():
            save_state_to_pkl(self.task_cfg.task_name)
        
        def on_esc():
            QUIT_REQUESTED[0] = True

        def on_tab():
            EPISODE_DONE[0] = True

        def on_backspace():
            DISCARD_REQUESTED[0] = True

        keyboard_interface.register_custom_keymapping(
            key=lazy.carb.input.KeyboardInput.R,
            description="Reset to initial state from pickle",
            callback_fn=reset_env,
        )
        keyboard_interface.register_custom_keymapping(
            key=lazy.carb.input.KeyboardInput.S,
            description="Save serialized state to init_states and breakpoint",
            callback_fn=save_state_and_break,
        )
        keyboard_interface.register_custom_keymapping(
            key=lazy.carb.input.KeyboardInput.ESCAPE,
            description="Quit immediately and save video",
            callback_fn=on_esc,
        )
        keyboard_interface.register_custom_keymapping(
            key=lazy.carb.input.KeyboardInput.TAB,
            description="End current episode (reset & start next)",
            callback_fn=on_tab,
        )
        keyboard_interface.register_custom_keymapping(
            key=lazy.carb.input.KeyboardInput.BACKSPACE,
            description="Discard current trajectory and start over (no save)",
            callback_fn=on_backspace,
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
    
    def record_step(self, obs, info):
        health_list_link_names = getattr(self.env, "health_list_link_names", None) or []
        health_arr = obs.get("health")
        self.record_health_step(
            health_arr,
            health_list_link_names,
        )
        damage_info = info.get("damage_info", {})
        self.record_forces_step(damage_info)
        frame = capture_viewer_rgb()
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
    
    if save_to_hdf5 and args.save_obs_to_hdf5 and getattr(task_cfg, "external_camera_configs", None):
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
        env = OGDamageableDataCollectionWrapper(
            env=base_env,
            output_path=args.collect_hdf5_path,
            only_successes=False,
            save_video=args.save_video, 
            save_obs_to_hdf5=args.save_obs_to_hdf5
        )
    else:
        env = base_env
    # ================================================

    # Get the robot
    robot = env.robots[0]

    # Reset the environment to the initial state
    reset_env(env, task_cfg, task_mod)

    # Setup teleop wrapper
    init_grasp = robot.is_grasping().value == IsGraspingState.TRUE
    teleop_wrapper = TeleopWrapper(
        env=env,
        robot=robot,
        task_cfg=task_cfg,
        init_grasp=init_grasp,
        **vars(args),
        )

    # Setup live health visualization if enabled
    if args.live_feedback:
        env.enable_health_visualization()

    n_episodes = args.n_episodes
    completed_episodes = 0

    # Loop through episodes
    while completed_episodes < n_episodes:
        print("\n" + "="*80)
        print(f"[TELEOP] Running episode {completed_episodes + 1}/{n_episodes}…")
        print("Press TAB to end an episode (and save if save_to_hdf5 is True), ESC to quit, BACKSPACE to discard and restart.")
        print("Press c to continue")
        print("="*80 + "\n")
        teleop_wrapper.reset_teleop_wrapper()
        breakpoint()
                
        # Loop through steps of the current episode
        while True:
            action = teleop_wrapper.get_action()
            if QUIT_REQUESTED[0] or EPISODE_DONE[0] or DISCARD_REQUESTED[0]:
                break
            obs, reward, terminated, truncated, info = env.step(action.clone())
            if args.save_video:
                teleop_wrapper.record_step(obs, info)

        if QUIT_REQUESTED[0]:
            break

        if DISCARD_REQUESTED[0]:
            print(f"[TELEOP] Discarded episode {completed_episodes + 1}. Resetting and starting over…")
            reset_env(env, task_cfg, task_mod)
            DISCARD_REQUESTED[0] = False

        if EPISODE_DONE[0]:
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
        env.save_data()
        print(f"[teleop] HDF5 saved to {args.collect_hdf5_path}")

    if args.live_feedback:
        env.disable_health_visualization()

    og.shutdown()


if __name__ == "__main__":
    main()
