"""
Playback script for oopsieverse teleoperation data.

Replays recorded states and actions to compute damage observations, then
saves the enriched dataset (with rendered camera images and health data)
to a new HDF5 file.

Usage::

    python scripts/playback_robocasa.py --input <collected.hdf5> --output <playback.hdf5> --env ENV_NAME

Examples::

    python scripts/playback_robocasa.py --input resources/teleop_data/ENV_NAME.hdf5 --output resources/playback_data/ENV_NAME.hdf5 --env ENV_NAME
    python scripts/playback_robocasa.py --input resources/teleop_data/ENV_NAME.hdf5 --output resources/playback_data/ENV_NAME.hdf5 --env ENV_NAME --visualize
    python scripts/playback_robocasa.py --input resources/teleop_data/ENV_NAME.hdf5 --output resources/playback_data/ENV_NAME.hdf5 --env ENV_NAME --metrics
"""

import os
import cv2
import sys
import h5py
import json
import time
import numpy as np
import argparse
import matplotlib
matplotlib.use("Agg")  # non-GUI backend
from collections import defaultdict

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)
sys.path.insert(1, os.path.join(_project_root, "oopsiebench"))

from robosuite.controllers import load_composite_controller_config
from envs.registry import EnvironmentRegistry
from utils.misc_utils import (
    process_traj_to_hdf5,
    flush_current_file,
    save_rgb_camera_video,
    save_rgb_force_video,
    save_rgb_health_video,
)


# ═══════════════════════════════════════════════════════════════════════
# Visualization helpers
# ═══════════════════════════════════════════════════════════════════════


DEFAULT_FORCE_KEYS = ["filtered_qs_forces"]


def _to_str(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def derive_health_series(all_obj_healths, health_list_link_names):
    link_names = [_to_str(name) for name in health_list_link_names]
    health_by_link = {}
    for idx, link_name in enumerate(link_names):
        if idx < all_obj_healths.shape[1]:
            health_by_link[link_name] = all_obj_healths[:, idx]

    object_order = []
    health_by_object = {}
    for link_name, values in health_by_link.items():
        obj_name = link_name.split("@", 1)[0]
        if obj_name not in health_by_object:
            health_by_object[obj_name] = values
            object_order.append(obj_name)
        else:
            health_by_object[obj_name] = np.minimum(health_by_object[obj_name], values)

    return health_by_link, health_by_object, object_order


def resolve_force_keys(damage_info_entries, target_objects_forces):
    for damage_info in damage_info_entries:
        for obj_link_name in target_objects_forces:
            obj_name, link_name = obj_link_name.split("@", 1)
            mechanical = (
                damage_info.get(obj_name, {})
                .get(link_name, {})
                .get("mechanical", {})
            )
            if isinstance(mechanical, dict) and mechanical:
                if "filtered_qs_forces" in mechanical:
                    return ["filtered_qs_forces"]
                return [next(iter(mechanical.keys()))]
    return DEFAULT_FORCE_KEYS.copy()



# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════


def json_default(o):
    """Custom JSON encoder for numpy types."""
    if isinstance(o, (np.float32, np.float64, np.int32, np.int64, np.bool_)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, tuple):
        return list(o)
    if hasattr(o, "item"):
        try:
            return o.item()
        except Exception:
            pass
    raise TypeError(f"Object of type {type(o)} not JSON serializable")


def sync_damage_evaluator_velocities(env):
    """
    Sync damage evaluators' prev_linear_velocities with the current simulation state.
    Prevents spurious impact detection when restoring states during playback.
    """
    for obj in env.get_damageable_objects():
        for evaluator in obj.damage_evaluators:
            if hasattr(evaluator, 'prev_linear_velocities') and hasattr(evaluator, '_get_part_linear_velocity'):
                for part_name in evaluator._get_damageable_part_names():
                    evaluator.prev_linear_velocities[part_name] = evaluator._get_part_linear_velocity(part_name)


def flush_playback_traj(env, traj_grp_name, traj_data, playback_hdf5_file):
    traj_grp = process_traj_to_hdf5(
        env, traj_grp_name, traj_data,
        nested_keys=("obs", "info"), output_hdf5=playback_hdf5_file
    )
    health_list = []
    for obj in env.get_damageable_objects():
        for link_name in obj.link_healths:
            health_list.append(f"{obj.name}@{link_name}")
    traj_grp.attrs["health_list_link_names"] = health_list
    flush_current_file(playback_hdf5_file)



# ═══════════════════════════════════════════════════════════════════════
# Episode playback
# ═══════════════════════════════════════════════════════════════════════


def playback_episode(src_f, demo_name, env, playback_hdf5_file):
    """
    Replay a recorded demo using a hybrid restore-then-step approach:
      1. Restore the recorded state (prevents trajectory divergence)
      2. Step with the recorded action (computes contact forces and damage)
      3. Collect obs and info after each step

    Returns the trajectory data list.
    """
    states = src_f[f"data/{demo_name}/states"][:]
    # The first action in the collected dataset has no preceding recorded state,
    # so we skip it and align actions[1:] with states[0:].
    actions = src_f[f"data/{demo_name}/actions"][1:]

    num_actions = len(actions)
    num_states = len(states)
    assert num_states == num_actions + 1, (
        f"{demo_name}: expected {num_actions + 1} states, got {num_states}"
    )

    env.reset()

    # ── Restore exact model if stored ──
    demo_grp = src_f[f"data/{demo_name}"]
    if "model_file" in demo_grp.attrs:
        model_xml = demo_grp.attrs["model_file"]
        ep_meta = json.loads(demo_grp.attrs.get("ep_meta", "{}"))
        env.set_ep_meta(ep_meta)
        env.reset_from_xml_string(model_xml)
        env.sim.reset()

    # ── Hide teleop visualization markers ──
    for robot in env.robots:
        for arm_name in robot.arms:
            if robot.eef_site_id[arm_name] is not None:
                env.sim.model.site_rgba[robot.eef_site_id[arm_name]] = np.array([0., 0., 0., 0.])
            if robot.eef_cylinder_id[arm_name] is not None:
                env.sim.model.site_rgba[robot.eef_cylinder_id[arm_name]] = np.array([0., 0., 0., 0.])

    # ── Restore initial state and collect initial observation ──
    env.sim.set_state_from_flattened(states[0])
    env.sim.forward()
    sync_damage_evaluator_velocities(env)

    obs, info = env.get_observations()
    traj_data = [{"obs": obs, "info": info}]

    for i in range(num_actions):
        env.sim.set_state_from_flattened(states[i])
        env.sim.forward()
        sync_damage_evaluator_velocities(env)

        obs, reward, done, info = env.step(actions[i])
        traj_data.append({"obs": obs, "action": actions[i], "reward": reward, "info": info})

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{num_actions} steps")

    flush_playback_traj(env, demo_name, traj_data, playback_hdf5_file)
    return traj_data



# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════


def create_parser():
    parser = argparse.ArgumentParser(
        description="Playback teleoperation data and render observations for oopsieverse",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Playback only (no rendering)
  python scripts/playback_robocasa.py --input ENV_NAME.hdf5 --output ENV_NAME_rendered.hdf5 --env ENV_NAME

  # Playback + render all cameras
  python scripts/playback_robocasa.py --input ENV_NAME.hdf5 --output ENV_NAME_rendered.hdf5 --env ENV_NAME --camera all_cameras

  # Playback + visualize and compute metrics
  python scripts/playback_robocasa.py --input ENV_NAME.hdf5 --output ENV_NAME_rendered.hdf5 --env ENV_NAME --visualize --metrics
        """
    )
    parser.add_argument("--input", required=True, help="Path to collected (teleop) HDF5 file")
    parser.add_argument("--output", required=True, help="Path for playback (rendered) HDF5 output file")
    parser.add_argument("--env", required=True, help="Environment name (e.g. pastry_display)")
    parser.add_argument("--camera", default="all_cameras", help="Camera(s) to render (default: all_cameras)")
    parser.add_argument("--width", type=int, default=256, help="Frame width (default: 256)")
    parser.add_argument("--height", type=int, default=256, help="Frame height (default: 256)")
    parser.add_argument("--low-dim", action="store_true", help="Use low-dimensional observations")
    parser.add_argument("--visualize", action="store_true", help="Render and save videos after playback")
    parser.add_argument("--metrics", action="store_true", help="Compute and print health metrics after playback")
    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input HDF5 file not found: {args.input}")
        print("\nUsage: python scripts/playback_robocasa.py --input <file> --output <file> --env <env>")
        envs = EnvironmentRegistry.list_envs()
        print(f"Available environments: {', '.join(envs)}")
        return

    try:
        env_config = EnvironmentRegistry.get(args.env)
    except Exception:
        print(f"Error: Invalid environment name '{args.env}'")
        print(f"Available environments: {', '.join(EnvironmentRegistry.list_envs())}")
        return

    if args.camera == "all_cameras":
        camera_names = ["robot0_eye_in_hand", "robot0_agentview_left", "robot0_agentview_right"]
    else:
        camera_names = [args.camera]

    print(f"\n{'='*60}")
    print(f"oopsieverse Playback")
    print(f"{'='*60}")
    print(f"Input  : {args.input}")
    print(f"Output : {args.output}")
    print(f"Env    : {args.env}")
    print(f"Cameras: {', '.join(camera_names)}")
    print(f"Res    : {args.width}x{args.height}")
    print(f"{'='*60}\n")

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    src_f = h5py.File(args.input, "r")
    playback_hdf5_file = h5py.File(args.output, "w")

    if "data" not in src_f:
        print("No data found in input HDF5 file")
        return

    demos = list(src_f["data"].keys())
    print(f"Found demos: {demos}\n")

    env = env_config.damageable_class(
        robots=env_config.robot,
        controller_configs=load_composite_controller_config(robot=env_config.robot),
        translucent_robot=False,
        has_renderer=False,
        has_offscreen_renderer=True,
        ignore_done=True,
        use_camera_obs=True,
        camera_names=camera_names,
        camera_widths=args.width,
        camera_heights=args.height,
        camera_depths=False,
        low_dim=args.low_dim,
        control_freq=env_config.control_freq,
    )

    for demo_num, demo_name in enumerate(demos):
        if f"data/{demo_name}/states" not in src_f:
            print(f"Skipping {demo_name} — no states saved")
            continue
        if f"data/{demo_name}/actions" not in src_f:
            print(f"Skipping {demo_name} — no actions saved")
            continue

        num_states = src_f[f"data/{demo_name}/states"].shape[0]
        num_actions = src_f[f"data/{demo_name}/actions"].shape[0]
        # Teleop saves one (state, action) pair per step, so num_states == num_actions.
        # playback_episode handles this by skipping actions[0].
        if num_states != num_actions:
            print(f"Warning: {demo_name} has {num_states} states and {num_actions} actions (expected equal counts)")

        print(f"Playing back demo {demo_num + 1}/{len(demos)}: {demo_name}")
        start_time = time.time()
        playback_episode(src_f, demo_name, env, playback_hdf5_file)
        print(f"  Done in {time.time() - start_time:.1f}s")

    src_f.close()

    if args.visualize or args.metrics:
        f = h5py.File(args.output, "r")
        output_video_dir = f"resources/videos/{os.path.splitext(os.path.basename(args.output))[0]}"
        os.makedirs(output_video_dir, exist_ok=True)

        final_obj_healths = defaultdict(list)
        final_env_healths = []

        for demo_name in sorted(f["data"].keys()):
            print(f"Episode: {demo_name}")
            demo_group = f[f"data/{demo_name}"]

            obs_info_list = []
            for i in range(len(demo_group["info/obs_info"])):
                obs_info = json.loads(demo_group["info/obs_info"][i].decode("utf-8"))
                obs_info_list.append(obs_info)

            all_obj_healths = np.array(demo_group["obs/health"])
            health_list_link_names = demo_group.attrs["health_list_link_names"]
            health_by_link, health_by_object, target_objects_health = derive_health_series(
                all_obj_healths, health_list_link_names
            )
            target_objects_forces = list(health_by_link.keys())

            damage_info_entries = [
                json.loads(demo_group["info/damage_info"][i].decode("utf-8"))
                for i in range(len(demo_group["info/damage_info"]))
            ]
            force_keys = resolve_force_keys(damage_info_entries, target_objects_forces)

            if args.metrics:
                current_env_health = 0.0
                counted_objects = 0
                for obj_name in target_objects_health:
                    if obj_name in health_by_object:
                        final_obj_healths[obj_name].append(health_by_object[obj_name][-1])
                        print(f"  {obj_name} final health: {health_by_object[obj_name][-1]:.1f}%")
                        current_env_health += health_by_object[obj_name][-1]
                        counted_objects += 1
                if counted_objects:
                    final_env_healths.append(current_env_health / counted_objects)

            if args.visualize:
                obs_keys = list(demo_group["obs"].keys())
                image_keys = [k for k in obs_keys if k.endswith("_image")]
                if not image_keys:
                    print(f"Skipping visualization for {demo_name} — no image observations found")
                    continue
                preferred_image_key = f"{env_config.camera_name}_image"
                image_key = preferred_image_key if preferred_image_key in image_keys else image_keys[0]
                camera_name = image_key[:-6]
                segmentation_key = f"{camera_name}_segmentation_class"
                if segmentation_key not in obs_keys:
                    print(f"Skipping visualization for {demo_name} — missing {segmentation_key}")
                    continue

                imgs = demo_group[f"obs/{image_key}"]
                new_imgs = []
                imgs_seg = demo_group[f"obs/{segmentation_key}"]

                for i, img in enumerate(imgs):
                    img = cv2.cvtColor(img[:, :, :3], cv2.COLOR_RGB2BGR)
                    img_seg = imgs_seg[i]
                    obs_info = obs_info_list[i]
                    camera_type = "robot0"
                    for obj_name in target_objects_health:
                        if obj_name not in health_by_object or health_by_object[obj_name][i] >= 100:
                            continue
                        seg_instance_info = obs_info.get(camera_type, {}).get(camera_name, {}).get("seg_instance", {})
                        seg_key = next((k for k, v in seg_instance_info.items() if v == obj_name), None)
                        if seg_key is None:
                            continue
                        seg_instance_key = int(seg_key)
                        alpha = 1 - health_by_object[obj_name][i] / 100.0
                        overlay_color = np.array([0, 0, 255], dtype=np.uint8)
                        mask = img_seg == seg_instance_key
                        img[mask] = ((1 - alpha) * img[mask] + alpha * overlay_color).astype(np.uint8)
                    new_imgs.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

                imgs = np.array(new_imgs)

                # Save camera video
                camera_video_path = os.path.join(output_video_dir, f"{demo_name}_camera_video.mp4")
                save_rgb_camera_video(output_video_path=camera_video_path, imgs=imgs)

                # Save force video
                data = {}
                for obj_name in target_objects_forces:
                    data[obj_name] = {fk: [] for fk in force_keys}
                for damage_info in damage_info_entries:
                    for obj_name in target_objects_forces:
                        obj_part, link_part = obj_name.split("@")
                        for fk in force_keys:
                            value = (
                                damage_info.get(obj_part, {})
                                .get(link_part, {})
                                .get("mechanical", {})
                                .get(fk, 0.0)
                            )
                            data[obj_name][fk].append(value)

                forces_video_path = os.path.join(output_video_dir, f"{demo_name}_forces_video.mp4")
                save_rgb_force_video(output_video_path=forces_video_path, imgs=imgs, target_objects=target_objects_forces, data=data, forces_to_plot=force_keys)

                health_video_path = os.path.join(output_video_dir, f"{demo_name}_health_video.mp4")
                save_rgb_health_video(
                    output_video_path=health_video_path,
                    imgs=imgs,
                    target_objects=target_objects_health,
                    health=health_by_object,
                )

        f.close()

    if args.metrics:
        print(f"\n{'='*40}")
        print("Health Metrics Summary")
        print(f"{'='*40}")
        for obj_name in sorted(final_obj_healths.keys()):
            if final_obj_healths[obj_name]:
                print(f"  {obj_name}: avg final health = {np.mean(final_obj_healths[obj_name]):.1f}%")
        if final_env_healths:
            print(f"  Overall env: avg final health = {np.mean(final_env_healths):.1f}%")

    print("\nPlayback complete.")
    playback_hdf5_file.close()


if __name__ == "__main__":
    main()
