#!/usr/bin/env python3
"""
Unified playback & visualisation script for OmniGibson damage-tracking tasks.

Usage examples
--------------
# Playback shelve_item demos and save observations + health to a new HDF5
python scripts/playback.py --task_name shelve_item --playback

# Visualise health-overlay videos from an already-played-back HDF5
python scripts/playback.py --task_name shelve_item --visualize

# Compute per-object health metrics
python scripts/playback.py --task_name shelve_item --compute_metrics

# High-res playback with specific demo IDs
python scripts/playback.py --task_name shelve_item --playback --high_resolution --demo_ids 0 1 2

Supported task names
--------------------
shelve_item, add_firewood, pour_water   (add more in ``scripts/task_configs/``)
"""

from __future__ import annotations

import os
os.environ["CARB_LOG_CHANNELS"] = "omni.physx.plugin=off"
# os.environ.setdefault("CARB_LOG_CHANNELS", "omni.physx.plugin=off")

import argparse
import importlib
import json
import sys
from collections import defaultdict
from typing import Dict, List, Optional

import cv2
import h5py
import numpy as np

# ── Task-config registry ────────────────────────────────────────────────

# Maps CLI task_name → module path under ``scripts.task_configs``
TASK_REGISTRY: Dict[str, str] = {
    "shelve_item": "scripts.task_configs.shelve_item",
    "add_firewood": "scripts.task_configs.add_firewood",
    "firewood": "scripts.task_configs.add_firewood",  # alias
    "pour_water": "scripts.task_configs.pour_water",
}


def load_task_config(task_name: str):
    """Import the task config module and return its ``TaskConfig``."""
    if task_name not in TASK_REGISTRY:
        available = ", ".join(sorted(TASK_REGISTRY.keys()))
        raise ValueError(
            f"Unknown task '{task_name}'. Available tasks: {available}"
        )
    mod = importlib.import_module(TASK_REGISTRY[task_name])
    return mod.get_task_config()


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def build_external_sensors_config(
    task_cfg,
    robot_name: str,
    robot_type: str,
    image_height: int = 256,
    image_width: int = 256,
):
    """
    Build the list-of-dicts ``external_sensors_config`` expected by
    ``DamageableDataPlaybackWrapper.create_from_hdf5``.
    """
    import torch as th

    sensors = []
    for name, cam_cfg in task_cfg.external_camera_configs.items():
        idx = name.split("_")[-1]
        prim_path = (
            f"/controllable__damageable{robot_type.lower()}"
            f"__{robot_name}/base_link/external_sensor{idx}"
        )
        sensors.append(
            {
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
            }
        )
    return sensors


def extract_health_from_hdf5(
    f: h5py.File,
    demo_key: str,
    target_objects_health_with_links: List[str],
    target_objects_health: List[str],
):
    """
    Read per-link health arrays from the HDF5 and aggregate per-object
    (min across links).

    Returns:
        health: dict mapping ``obj@link`` *and* ``obj`` names → numpy arrays
    """
    all_obj_healths = np.array(f[f"data/{demo_key}/obs/health"])
    health_list_link_names = f[f"data/{demo_key}"].attrs["health_list_link_names"]
    health: Dict[str, Optional[np.ndarray]] = {}

    # Per-link health
    for obj_link_name in target_objects_health_with_links:
        idx = np.where(health_list_link_names == obj_link_name)[0]
        if len(idx) > 0:
            health[obj_link_name] = all_obj_healths[:, idx[0]][1:]  # skip t=0
        else:
            health[obj_link_name] = None

    # Aggregate per-object (min across links)
    for obj_name in target_objects_health:
        arrays = [
            v
            for k, v in health.items()
            if k.startswith(f"{obj_name}@") and v is not None
        ]
        health[obj_name] = np.minimum.reduce(arrays) if arrays else None

    return health


def overlay_health_on_frames(
    f: h5py.File,
    demo_key: str,
    camera_type: str,
    camera_name: str,
    target_objects_health: List[str],
    health: dict,
):
    """
    Read RGB + seg_instance frames from HDF5, tint damaged objects red,
    and return the resulting numpy array of RGB frames.
    """
    obs_info_list = []
    for i in range(len(f[f"data/{demo_key}/info/obs_info"])):
        obs_info = json.loads(
            f[f"data/{demo_key}/info/obs_info"][i].decode("utf-8")
        )
        obs_info_list.append(obs_info)

    imgs = np.array(f[f"data/{demo_key}/obs/{camera_type}::{camera_name}::rgb"])[1:]
    imgs_seg = np.array(
        f[f"data/{demo_key}/obs/{camera_type}::{camera_name}::seg_instance"]
    )[1:]

    new_imgs = []
    for i, img in enumerate(imgs):
        img = cv2.cvtColor(img[:, :, :3], cv2.COLOR_RGB2BGR)
        seg = imgs_seg[i]
        obs_info = obs_info_list[i]

        for obj_name in target_objects_health:
            if health.get(obj_name) is None:
                continue
            seg_info = obs_info[camera_type][camera_name]["seg_instance"]
            seg_key = int(
                next((k for k, v in seg_info.items() if v == obj_name), -1)
            )
            if seg_key == -1:
                continue
            h_val = health[obj_name][i]
            if h_val is not None and h_val < 100:
                mask = seg == seg_key
                alpha = 1.0 - h_val / 100.0
                overlay = np.array([0, 0, 255], dtype=np.uint8)
                img[mask] = (
                    (1 - alpha) * img[mask] + alpha * overlay
                ).astype(np.uint8)

        new_imgs.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    return np.array(new_imgs)


# ═══════════════════════════════════════════════════════════════════════
# PLAYBACK
# ═══════════════════════════════════════════════════════════════════════

def run_playback(args, task_cfg):
    """Create an OG env from HDF5 and replay demonstrations."""
    import torch as th
    import omnigibson as og
    from omnigibson.macros import gm

    from damagesim.omnigibson.damageable_env import (
        OGDamageableDataPlaybackWrapper,
    )

    gm.USE_GPU_DYNAMICS = task_cfg.use_gpu_dynamics
    gm.ENABLE_TRANSITION_RULES = task_cfg.enable_transition_rules

    robot_name = task_cfg.robot_name
    robot_type = task_cfg.robot_type

    if args.high_resolution:
        image_h, image_w = 1280, 1280
    else:
        image_h, image_w = 256, 256

    external_sensors_config = build_external_sensors_config(
        task_cfg, robot_name, robot_type, image_h, image_w,
    )

    robot_sensor_config = {
        "VisionSensor": {
            "modalities": ["rgb", "seg_instance"],
            "sensor_kwargs": {
                "image_height": image_h,
                "image_width": image_w,
            },
        },
    }

    # Allow task-specific playback wrapper (e.g. firewood overrides playback_episode)
    wrapper_cls = task_cfg.playback_wrapper_cls or OGDamageableDataPlaybackWrapper

    env = wrapper_cls.create_from_hdf5(
        input_path=args.collect_hdf5_path,
        output_path=args.playback_hdf5_path,
        robot_obs_modalities=["proprio", "rgb", "seg_instance"],
        robot_sensor_config=robot_sensor_config,
        external_sensors_config=external_sensors_config,
        n_render_iterations=1,
        only_successes=False,
    )

    # Viewer camera
    og.sim.viewer_camera.set_position_orientation(
        position=th.tensor(task_cfg.viewer_camera_pos, dtype=th.float32),
        orientation=th.tensor(task_cfg.viewer_camera_orn, dtype=th.float32),
    )
    for _ in range(10):
        og.sim.step()

    # Optional post-creation hook (e.g. _ensure_firewood_states)
    if task_cfg.post_playback_env_setup is not None:
        task_cfg.post_playback_env_setup(env)

    # Run playback
    demo_ids = args.demo_ids if args.demo_ids else None
    env.playback_dataset(record_data=True, demo_ids=demo_ids)
    env.save_data()

    og.shutdown()
    print(f"Playback complete.  Output → {args.playback_hdf5_path}")


# ═══════════════════════════════════════════════════════════════════════
# VISUALIZE
# ═══════════════════════════════════════════════════════════════════════

def run_visualize(args, task_cfg):
    """Read a played-back HDF5 and produce health-overlay videos."""
    from damagesim.utils.visualization import (  # noqa: F401 – lazy import
        save_rgb_camera_video,
        save_rgb_health_video_with_overlay,
    )

    f = h5py.File(args.playback_hdf5_path, "r")
    camera_type = "external"
    camera_name = args.camera_name

    output_dir = args.video_dir or task_cfg.default_video_dir
    os.makedirs(output_dir, exist_ok=True)

    for demo_key in sorted(f["data"].keys()):
        if not demo_key.startswith("demo_"):
            continue
        demo_idx = int(demo_key.split("_")[-1])
        print(f"Visualising episode {demo_idx} …")

        health = extract_health_from_hdf5(
            f,
            demo_key,
            task_cfg.target_objects_health_with_links,
            task_cfg.target_objects_health,
        )

        imgs = overlay_health_on_frames(
            f,
            demo_key,
            camera_type,
            camera_name,
            task_cfg.target_objects_health,
            health,
        )

        # Plain camera video (with health tint)
        cam_path = os.path.join(output_dir, f"demo_{demo_idx}_camera_video.mp4")
        save_rgb_camera_video(output_video_path=cam_path, imgs=imgs, fps=30)

        # Health overlay bars
        overlay_path = os.path.join(
            output_dir, f"demo_{demo_idx}_health_overlay_video.mp4"
        )
        save_rgb_health_video_with_overlay(
            output_video_path=overlay_path,
            imgs=imgs,
            target_objects=task_cfg.target_objects_health,
            health=health,
            position="bottom_center",
            n_columns=3,
            fps=30,
        )

    f.close()
    print(f"Videos saved to {output_dir}")


# ═══════════════════════════════════════════════════════════════════════
# COMPUTE METRICS
# ═══════════════════════════════════════════════════════════════════════

def run_compute_metrics(args, task_cfg):
    """Print per-object and per-episode health summaries."""
    f = h5py.File(args.playback_hdf5_path, "r")

    final_obj_healths: Dict[str, list] = defaultdict(list)
    final_env_healths: list = []

    for demo_key in sorted(f["data"].keys()):
        if not demo_key.startswith("demo_"):
            continue
        demo_idx = int(demo_key.split("_")[-1])

        health = extract_health_from_hdf5(
            f,
            demo_key,
            task_cfg.target_objects_health_with_links,
            task_cfg.target_objects_health,
        )

        print(f"\nEpisode {demo_idx}:")
        env_health = 0.0
        valid = 0
        for obj_name in task_cfg.target_objects_health:
            h = health.get(obj_name)
            if h is not None:
                final_val = float(h[-1])
                final_obj_healths[obj_name].append(final_val)
                env_health += final_val
                valid += 1
                print(f"  {obj_name:30s}  final health = {final_val:.2f}")
        if valid:
            final_env_healths.append(env_health / valid)

    # Summary
    print("\n" + "=" * 60)
    print("METRICS SUMMARY")
    print("=" * 60)
    for obj_name, vals in final_obj_healths.items():
        print(f"  {obj_name:30s}  avg = {np.mean(vals):.2f}")
    if final_env_healths:
        print(f"  {'Environment':30s}  avg = {np.mean(final_env_healths):.2f}")
    total_eps = len([k for k in f["data"].keys() if k.startswith("demo_")])
    zero_dmg = sum(
        1
        for eh in final_env_healths
        if eh >= 100.0
    )
    print(f"  Zero-damage episodes: {zero_dmg} / {total_eps}")
    print("=" * 60)

    f.close()


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified playback & visualisation for OG damage-tracking tasks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--task_name",
        type=str,
        required=True,
        help="Task name (e.g. shelve_item, add_firewood, pour_water).",
    )

    # Mode flags
    parser.add_argument("--playback", action="store_true", help="Replay recorded HDF5.")
    parser.add_argument("--visualize", action="store_true", help="Generate videos from played-back HDF5.")
    parser.add_argument("--compute_metrics", action="store_true", help="Compute per-object health metrics.")

    # Paths
    parser.add_argument("--collect_hdf5_path", type=str, default=None, help="Input (teleop) HDF5 path.")
    parser.add_argument("--playback_hdf5_path", type=str, default=None, help="Output (playback) HDF5 path.")
    parser.add_argument("--video_dir", type=str, default=None, help="Directory for saved videos.")

    # Playback options
    parser.add_argument("--demo_ids", nargs="*", type=int, default=None, help="Specific demo IDs to playback.")
    parser.add_argument("--high_resolution", action="store_true", help="Use 1280×1280 images.")
    parser.add_argument("--camera_name", type=str, default="external_sensor0", help="Camera for visualisation.")

    return parser.parse_args()


def main():
    args = parse_args()

    # Load per-task config
    task_cfg = load_task_config(args.task_name)

    # Fill in default paths from task config when not provided on CLI
    if args.collect_hdf5_path is None:
        args.collect_hdf5_path = task_cfg.default_collect_hdf5
    if args.playback_hdf5_path is None:
        args.playback_hdf5_path = task_cfg.default_playback_hdf5

    if not any([args.playback, args.visualize, args.compute_metrics]):
        print("Nothing to do. Specify at least one of: --playback  --visualize  --compute_metrics")
        sys.exit(0)

    if args.playback:
        run_playback(args, task_cfg)

    if args.visualize:
        run_visualize(args, task_cfg)

    if args.compute_metrics:
        run_compute_metrics(args, task_cfg)


if __name__ == "__main__":
    main()

