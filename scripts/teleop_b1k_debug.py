#!/usr/bin/env python3
"""
Debug variant of `teleop_b1k.py`: same flow, plus extra prints (health deltas, pick_egg gripper).

For `pick_egg`, skips `ensure_gripper_closed` / `ensure_gripper_persistent_closed` so the hand
starts open; registers Z/X incremental gripper on the `KeyboardRobotController`.

Usage:
    python scripts/teleop_b1k_debug.py --task_name pick_egg --skip_hdf5_save

See `teleop_b1k.py` for full key help and HDF5 / video options.
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

from damagesim.omnigibson.damageable_env import (
    OGDamageableEnvironment,
    OGDamageableDataCollectionWrapper,
)
from damagesim.utils.visualization import (
    save_rgb_camera_video,
    save_rgb_health_video_with_overlay,
    save_rgb_force_video,
)

class _TeleopDataCollectionWrapper(OGDamageableDataCollectionWrapper):
    """Subclass that optionally saves obs/info to HDF5 and conditionally
    skips sim optimizations for video recording."""

    def __init__(self, *args, save_video=False, save_extra_obs=False, **kwargs):
        self._save_video = save_video
        self._save_extra_obs = save_extra_obs
        super().__init__(*args, **kwargs)

    def _optimize_sim_for_data_collection(self, viewport_camera_path):
        if self._save_video:
            return
        super()._optimize_sim_for_data_collection(viewport_camera_path)

    def _parse_step_data(self, action, obs, reward, terminated, truncated, info):
        step_data = super()._parse_step_data(action, obs, reward, terminated, truncated, info)
        if self._save_extra_obs:
            # process_traj_to_hdf5 expects flat obs: modality_key -> tensor per step
            step_data["obs"] = flatten_obs(obs)
            step_data["info"] = info
        return step_data


# --task_name picks which module to import from this package
TASK_CONFIG_PACKAGE = "oopsiebench.envs.behavior1k"

TASK_REGISTRY = {
    "shelve_item": "shelve_item",
    "add_firewood": "add_firewood",
    "firewood": "add_firewood",
    "pour_water": "pour_water",
    "wipe_counter": "wipe_counter",
    "nav_to_table": "nav_to_table",
    "place_bowl": "place_bowl",
    "place_plate": "place_plate",
    "open_drawer": "open_drawer",
    "pick_egg": "pick_egg",
}


def load_task_config(task_name: str):
    """Import the task config module and return its (TaskConfig, module)."""
    if task_name not in TASK_REGISTRY:
        available = ", ".join(sorted(TASK_REGISTRY.keys()))
        raise ValueError(f"Unknown task '{task_name}'. Available tasks: {available}")
    mod = importlib.import_module(f"{TASK_CONFIG_PACKAGE}.{TASK_REGISTRY[task_name]}")
    return mod.get_task_config(), mod


def ensure_gripper_closed(env):
    if not env.robots:
        return
    robot = env.robots[0]
    close_fn = getattr(robot, "close_gripper", None)
    if callable(close_fn):
        try:
            close_fn()
        except Exception:
            pass


def ensure_gripper_persistent_closed(action_generator):
    try:
        binary_grippers = getattr(action_generator, "binary_grippers", [])
        gripper_direction = getattr(action_generator, "gripper_direction", None)
        persistent = getattr(action_generator, "persistent_gripper_action", None)
        if gripper_direction is None or persistent is None:
            return
        for comp in binary_grippers:
            gripper_direction[comp] = -1.0
            persistent[comp] = -1.0
    except Exception:
        pass


def _maybe_ensure_gripper_closed(env, task_name: str):
    """`pick_egg` starts with an open gripper from task reset; do not force-close."""
    if task_name == "pick_egg":
        print("[teleop_b1k_debug] pick_egg: skipping ensure_gripper_closed (start open)")
        return
    ensure_gripper_closed(env)


def _maybe_ensure_gripper_persistent_closed(action_generator, task_name: str):
    if task_name == "pick_egg":
        print("[teleop_b1k_debug] pick_egg: skipping ensure_gripper_persistent_closed")
        return
    ensure_gripper_persistent_closed(action_generator)


def _snapshot_env_health(base_env) -> dict | None:
    gh = base_env.get_env_health()
    return dict(gh) if gh else None


def _debug_print_health_drops(prev: dict | None, cur: dict | None, step_i: int) -> None:
    if not prev or not cur:
        return
    for k, v in cur.items():
        pv = float(prev.get(k, 100.0))
        fv = float(v)
        if fv < pv - 0.01:
            print(f"[teleop_b1k_debug] step {step_i}: health {k!r} {pv:.2f} -> {fv:.2f}")


def save_state_to_temp_pkl(task_name: str):
    state = og.sim.dump_state(serialized=True)
    init_dir = os.path.join(
        _REPO_ROOT, "resources", "init_states",
    )
    os.makedirs(init_dir, exist_ok=True)
    temp_path = os.path.join(init_dir, f"{task_name}_temp.pkl")
    with open(temp_path, "wb") as f:
        pickle.dump(state, f)
    print(f"[teleop] Saved serialized state to temp init-state {temp_path}")


def load_state_from_pkl(env, task_name: str, task_module=None):
    init_dir = os.path.join(
        _REPO_ROOT, "resources", "init_states",
    )
    canonical_path = os.path.join(init_dir, f"{task_name}.pkl")
    temp_path = os.path.join(init_dir, f"{task_name}_temp.pkl")
    path = canonical_path if os.path.isfile(canonical_path) else (
        temp_path if os.path.isfile(temp_path) else None
    )

    # Used by some task modules to decide whether to "re-setup" runtime-only
    # visuals (like particles) after loading a saved sim state.
    setattr(env, "_teleop_loaded_from_pkl", False)

    if path is None:
        print(f"[teleop] No init-state pickle found for '{task_name}'. Falling back to config reset.")
        env.reset()
        if task_module is not None and hasattr(task_module, "reset") and callable(task_module.reset):
            task_module.reset(env)
        return

    if path == temp_path:
        print(f"[teleop] Using temp init-state pickle at {path}")
    else:
        print(f"[teleop] Loading init-state pickle at {path}")

    with open(path, "rb") as f:
        state = pickle.load(f)

    env.reset()
    scene_file = getattr(env, "scene_file", None)
    if scene_file is not None:
        env.scene.restore(scene_file, update_initial_file=True)

    if not og.sim.is_playing():
        og.sim.play()

    loaded = False
    # First try: serialized state (what S-key saves)
    try:
        og.sim.load_state(state, serialized=True)
        loaded = True
        setattr(env, "_teleop_loaded_from_pkl", True)
    except Exception as e_serialized:
        print(f"[teleop] Failed loading serialized state from {path}: {e_serialized}")
        # Second try: dict-like non-serialized state
        try:
            og.sim.load_state(state, serialized=False)
            print(f"[teleop] Loaded non-serialized state from {path}")
            loaded = True
            setattr(env, "_teleop_loaded_from_pkl", True)
        except Exception as e_nonserialized:
            print(
                f"[teleop] Failed loading init-state pickle {path} (serialized and non-serialized). "
                f"Rebuilding compatible state."
            )

            env.reset()
            if task_module is not None and hasattr(task_module, "reset") and callable(task_module.reset):
                # During rebuilding we want runtime-only visuals (like dirt particles)
                # to be skipped so the repaired state stays as compatible as possible.
                setattr(env, "_teleop_loaded_from_pkl", False)
                task_module.reset(env)
            repaired_state = og.sim.dump_state(serialized=True)
            # Always write to canonical path for the next run
            with open(canonical_path, "wb") as f:
                pickle.dump(repaired_state, f)
            # Also keep the temp backup in sync
            with open(temp_path, "wb") as f:
                pickle.dump(repaired_state, f)
            og.sim.load_state(repaired_state, serialized=True)
            setattr(env, "_teleop_loaded_from_pkl", True)
            loaded = True

    if not loaded:
        raise RuntimeError(f"[teleop] Unexpected: init-state load did not complete for task '{task_name}'")

    for _ in range(10):
        og.sim.step()

    # Some tasks need to re-apply runtime-only setup (e.g. spawn particles)
    # after the saved sim state has been restored.
    if getattr(env, "_teleop_loaded_from_pkl", False) and task_module is not None and hasattr(task_module, "reset") and callable(task_module.reset):
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
    p.add_argument("--save_extra_obs", action="store_true",
                   help="When collecting to HDF5, also record external camera obs (rgb, seg_instance) like playback.")
    p.add_argument("--n_episodes", type=int, default=1,
                   help="Number of teleop episodes to run (default: 1).")
    p.add_argument("--skip_hdf5_save", action="store_true",
                   help="Skip saving the HDF5 file (default: False).")
    
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


def _record_forces_step(damage_info, target_objects_forces, force_keys, teleop_force_records):
    """Extract per-step forces from damage_info and append to teleop_force_records (in-place)."""
    if not target_objects_forces or not force_keys or not damage_info:
        return
    for obj_name in target_objects_forces:
        parts = obj_name.split("@", 1)
        if len(parts) != 2:
            for fk in force_keys:
                teleop_force_records.setdefault(obj_name, {}).setdefault(fk, []).append(0.0)
            continue
        obj_key, link_key = parts
        obj_info = damage_info.get(obj_key, {})
        link_info = obj_info.get(link_key, {})
        mechanical = link_info.get("mechanical", {})
        for fk in force_keys:
            val = mechanical.get(fk, 0.0)
            teleop_force_records.setdefault(obj_name, {}).setdefault(fk, []).append(val)


def _record_health_step(health_arr, health_list_link_names, overlay_links,
                        target_objects_for_overlay, teleop_health_records):
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
    if overlay_links:
        for name in target_objects_for_overlay:
            if name not in link_vals:
                continue
            teleop_health_records.setdefault(name, []).append(link_vals[name])
    else:
        for obj_name in target_objects_for_overlay:
            vals = [v for k, v in link_vals.items() if k.startswith(f"{obj_name}@")]
            per_step = min(vals) if vals else 100.0
            teleop_health_records.setdefault(obj_name, []).append(per_step)


MAX_RESET_RETRIES = 5


def _get_base_env(env):
    """Unwrap to the underlying OGDamageableEnvironment."""
    e = env
    while hasattr(e, 'env') and not isinstance(e, OGDamageableEnvironment):
        e = e.env
    return e


def _ensure_health_reset(base_env):
    """
    Re-baseline damage tracking after a full state load so that evaluators
    reference the current (post-load) positions/velocities. Settles physics
    for a few extra steps, then re-baselines again.

    Returns True when every tracked object reports full health (100).
    """
    base_env._reset_damage_tracking()
    for _ in range(5):
        og.sim.step()
    base_env._reset_damage_tracking()
    env_health = base_env.get_env_health()
    if not env_health:
        return True
    return all(h >= 100.0 for h in env_health.values())


def _save_video(teleop_frames, teleop_health_records, target_objects_for_overlay,
                task_cfg, teleop_fps=30, overlay_position="bottom_center", overlay_layout="column",
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
            fps=teleop_fps,
            position=overlay_position,
            layout=overlay_layout,
        )
        print(f"[teleop] Saved {len(teleop_frames)} frames with health overlay to {output_path}.mp4")
    else:
        save_rgb_camera_video(output_path, imgs, fps=teleop_fps)
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
                fps=teleop_fps,
            )
            print(f"[teleop] Saved force plot video to {forces_path}")


def main():
    args = parse_args()
    task_cfg, task_mod = load_task_config(args.task_name)

    gm.USE_GPU_DYNAMICS = task_cfg.use_gpu_dynamics
    gm.ENABLE_TRANSITION_RULES = task_cfg.enable_transition_rules

    env_config = build_env_config(task_cfg)
    collecting = not args.skip_hdf5_save
    if collecting and args.save_extra_obs and getattr(task_cfg, "external_camera_configs", None):
        env_config["env"]["external_sensors"] = build_external_sensors_config(
            task_cfg, task_cfg.robot_name, task_cfg.robot_type,
            image_height=1280, image_width=1280,
        )
    base_env = OGDamageableEnvironment(configs=env_config)

    if collecting:
        if args.collect_hdf5_path is None:
            args.collect_hdf5_path = os.path.join(
                _REPO_ROOT, "demos", "behavior1k", "teleop_data", f"{args.task_name}.hdf5"
            )
        os.makedirs(os.path.dirname(args.collect_hdf5_path) or ".", exist_ok=True)
        env = _TeleopDataCollectionWrapper(
            env=base_env,
            output_path=args.collect_hdf5_path,
            only_successes=False,
            save_video=args.save_video,
            save_extra_obs=args.save_extra_obs,
        )
    else:
        env = base_env

    the_base_env = _get_base_env(env)

    load_state_from_pkl(env, task_name=task_cfg.task_name, task_module=task_mod)

    og.sim.viewer_camera.set_position_orientation(
        position=th.tensor(task_cfg.viewer_camera_pos, dtype=th.float32),
        orientation=th.tensor(task_cfg.viewer_camera_orn, dtype=th.float32),
    )
    _maybe_ensure_gripper_closed(env, task_cfg.task_name)

    for _ in range(10):
        og.sim.step()

    for _attempt in range(1, MAX_RESET_RETRIES + 1):
        _ensure_health_reset(the_base_env)
        env_health = the_base_env.get_env_health()
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
        load_state_from_pkl(env, task_name=task_cfg.task_name, task_module=task_mod)
        og.sim.viewer_camera.set_position_orientation(
            position=th.tensor(task_cfg.viewer_camera_pos, dtype=th.float32),
            orientation=th.tensor(task_cfg.viewer_camera_orn, dtype=th.float32),
        )
        _maybe_ensure_gripper_closed(env, task_cfg.task_name)
        for _ in range(10):
            og.sim.step()
    else:
        the_base_env._reset_damage_tracking()
        print(f"[teleop] WARNING: Initial health max retries reached. Forced health reset.")

    robot = env.robots[0]
    action_generator = KeyboardRobotController(robot=robot)
    _maybe_ensure_gripper_persistent_closed(action_generator, task_cfg.task_name)

    def save_state_and_break():
        save_state_to_temp_pkl(task_cfg.task_name)

    def reset_to_prefix():
        og.sim.viewer_camera.set_position_orientation(
            position=th.tensor(task_cfg.viewer_camera_pos, dtype=th.float32),
            orientation=th.tensor(task_cfg.viewer_camera_orn, dtype=th.float32),
        )
        for _attempt in range(1, MAX_RESET_RETRIES + 1):
            load_state_from_pkl(env, task_name=task_cfg.task_name, task_module=task_mod)
            og.sim.viewer_camera.set_position_orientation(
                position=th.tensor(task_cfg.viewer_camera_pos, dtype=th.float32),
                orientation=th.tensor(task_cfg.viewer_camera_orn, dtype=th.float32),
            )
            _maybe_ensure_gripper_closed(env, task_cfg.task_name)
            _maybe_ensure_gripper_persistent_closed(action_generator, task_cfg.task_name)
            _ensure_health_reset(the_base_env)
            env_health = the_base_env.get_env_health()
            all_clean = (
                all(h >= 100.0 for h in env_health.values())
                if env_health else True
            )
            if all_clean:
                if _attempt > 1:
                    print(f"[teleop] Reset clean after {_attempt} attempts.")
                if task_cfg.task_name == "pick_egg" and hasattr(
                    task_mod, "sync_teleop_gripper_after_env_reset"
                ):
                    task_mod.sync_teleop_gripper_after_env_reset(env, action_generator)
                return
            damaged = {k: v for k, v in (env_health or {}).items() if v < 100.0}
            print(f"[teleop] Reset attempt {_attempt}/{MAX_RESET_RETRIES}: "
                  f"health not clean: {damaged}. Retrying…")
        the_base_env._reset_damage_tracking()
        print(f"[teleop] WARNING: Max retries ({MAX_RESET_RETRIES}) reached. Forced health reset.")

    action_generator.register_custom_keymapping(
        key=lazy.carb.input.KeyboardInput.R,
        description="Reset to initial state from pickle",
        callback_fn=reset_to_prefix,
    )
    action_generator.register_custom_keymapping(
        key=lazy.carb.input.KeyboardInput.S,
        description="Save serialized state to init_states and breakpoint",
        callback_fn=save_state_and_break,
    )

    quit_requested = [False]
    episode_done = [False]
    discard_requested = [False]

    def on_esc():
        quit_requested[0] = True

    def on_tab():
        episode_done[0] = True

    def on_backspace():
        discard_requested[0] = True

    action_generator.register_custom_keymapping(
        key=lazy.carb.input.KeyboardInput.ESCAPE,
        description="Quit immediately and save video",
        callback_fn=on_esc,
    )
    action_generator.register_custom_keymapping(
        key=lazy.carb.input.KeyboardInput.TAB,
        description="End current episode (reset & start next)",
        callback_fn=on_tab,
    )
    action_generator.register_custom_keymapping(
        key=lazy.carb.input.KeyboardInput.BACKSPACE,
        description="Discard current trajectory and start over (no save)",
        callback_fn=on_backspace,
    )

    if task_cfg.task_name == "pick_egg" and hasattr(
        task_mod, "register_pick_egg_incremental_gripper_keys"
    ):
        task_mod.register_pick_egg_incremental_gripper_keys(env, action_generator, debug=True)

    if args.live_feedback:
        env.enable_health_visualization()

    # After pick_egg Z/X registration so custom keys appear in the banner.
    action_generator.print_keyboard_teleop_info()

    n_episodes = args.n_episodes
    overlay_links = getattr(args, "overlay_links", False)
    if overlay_links:
        target_objects_for_overlay = (
            getattr(task_cfg, "target_objects_health_with_links", None)
            or getattr(task_cfg, "target_objects_health", None)
            or []
        )
    else:
        target_objects_for_overlay = (
            getattr(task_cfg, "target_objects_health", None) or []
        )

    teleop_frames = []
    teleop_health_records = {}
    teleop_force_records = {}
    teleop_fps = 30
    target_objects_forces = getattr(task_cfg, "target_objects_forces", None) or []
    force_keys = getattr(task_cfg, "force_keys", None) or ["filtered_qs_forces"]

    print(f"[teleop] Running {n_episodes} episode(s). Press TAB to end an episode, ESC to quit, BACKSPACE to discard and restart.")
    if args.save_video:
        teleop_frames.append(capture_viewer_rgb())

    completed_episodes = 0
    while completed_episodes < n_episodes:
        print(f"[teleop] Episode {completed_episodes + 1}/{n_episodes} — teleoperating…")
        print("Press c to continue")
        # breakpoint()
        episode_done[0] = False
        discard_requested[0] = False
        start_frame_count = len(teleop_frames)
        debug_step_i = [0]
        prev_health = _snapshot_env_health(the_base_env)

        while True:
            action, key_str = action_generator.get_teleop_action()
            if task_cfg.task_name == "pick_egg" and hasattr(
                task_mod, "sync_pick_egg_close_level_from_persistent"
            ):
                task_mod.sync_pick_egg_close_level_from_persistent(env, action_generator)
            if quit_requested[0] or episode_done[0] or discard_requested[0]:
                break
            obs, reward, terminated, truncated, info = env.step(action)
            debug_step_i[0] += 1
            cur_health = _snapshot_env_health(the_base_env)
            _debug_print_health_drops(prev_health, cur_health, debug_step_i[0])
            prev_health = cur_health
            if task_cfg.task_name == "pick_egg" and debug_step_i[0] % 300 == 0:
                comp = getattr(action_generator, "binary_grippers", None)
                cl = getattr(env, "_pick_egg_gripper_close_level", None)
                pers = None
                if comp and len(comp) > 0:
                    pers = float(action_generator.persistent_gripper_action[comp[0]])
                print(
                    f"[teleop_b1k_debug] pick_egg tick {debug_step_i[0]}: "
                    f"close_level={cl} persistent={pers} last_key={key_str!r}"
                )
            if args.save_video:
                health_list_link_names = getattr(env, "health_list_link_names", None) or []
                health_arr = obs.get("health")
                _record_health_step(
                    health_arr, health_list_link_names, overlay_links,
                    target_objects_for_overlay, teleop_health_records,
                )
                damage_info = info.get("damage_info", {})
                _record_forces_step(
                    damage_info, target_objects_forces, force_keys, teleop_force_records,
                )
                frame = capture_viewer_rgb()
                teleop_frames.append(frame)

        if quit_requested[0]:
            break

        if discard_requested[0]:
            n_discard = len(teleop_frames) - start_frame_count
            teleop_frames = teleop_frames[:start_frame_count]
            for k in list(teleop_health_records.keys()):
                arr = teleop_health_records[k]
                if len(arr) > n_discard:
                    teleop_health_records[k] = arr[:-n_discard]
                else:
                    teleop_health_records[k] = []
            for obj_name in list(teleop_force_records.keys()):
                for fk in list(teleop_force_records[obj_name].keys()):
                    arr = teleop_force_records[obj_name][fk]
                    if len(arr) > n_discard:
                        teleop_force_records[obj_name][fk] = arr[:-n_discard]
                    else:
                        teleop_force_records[obj_name][fk] = []
            if collecting and hasattr(env, "current_traj_history"):
                env.current_traj_history = []
            print(f"[teleop] Discarded {n_discard} steps. Resetting and starting over…")
            reset_to_prefix()
            discard_requested[0] = False
            continue

        completed_episodes += 1
        print(f"[teleop] Episode {completed_episodes}/{n_episodes} finished "
              f"({len(teleop_frames)} frames total).")

        if quit_requested[0]:
            break

        # Flush the episode trajectory to HDF5 if collecting
        if collecting and hasattr(env, "flush_current_traj"):
            env.task._success = True
            env.flush_current_traj()

        # Reset for next episode (if more remain)
        if completed_episodes < n_episodes:
            reset_to_prefix()

    # Flush any remaining trajectory data
    if collecting and hasattr(env, "flush_current_traj") and hasattr(env, "current_traj_history"):
        if len(env.current_traj_history) > 0:
            env.task._success = True
            env.flush_current_traj()

    if args.save_video:
        _save_video(
            teleop_frames,
            teleop_health_records,
            target_objects_for_overlay,
            task_cfg,
            teleop_fps,
            overlay_position=args.overlay_position,
            overlay_layout=args.overlay_layout,
            teleop_force_records=teleop_force_records,
        )

    if collecting and hasattr(env, "save_data"):
        env.save_data()
        print(f"[teleop] HDF5 saved to {args.collect_hdf5_path}")

    if args.live_feedback:
        env.disable_health_visualization()

    og.shutdown()


if __name__ == "__main__":
    main()
