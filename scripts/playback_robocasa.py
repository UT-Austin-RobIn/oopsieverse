"""
Playback script for oopsieverse teleoperation data.

Replays recorded states and actions to compute damage observations, then
saves the enriched dataset (with rendered camera images and health data)
to a new HDF5 file.

Usage::

    python scripts/playback_robocasa.py --input <collected.hdf5> --output <playback.hdf5> --env ENV_NAME

Examples::

    python scripts/playback_robocasa.py --input resources/teleop_data/pick_egg.hdf5 --output resources/playback_data/pick_egg.hdf5 --env pick_egg
    python scripts/playback_robocasa.py --input resources/teleop_data/pick_egg.hdf5 --output resources/playback_data/pick_egg.hdf5 --env pick_egg --visualize
    python scripts/playback_robocasa.py --input resources/teleop_data/pick_egg.hdf5 --output resources/playback_data/pick_egg.hdf5 --env pick_egg --metrics
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
# Visualization config
# ═══════════════════════════════════════════════════════════════════════


def get_visualization_config(task_name, robot_name):
    if task_name == "pick_egg":
        return {
            "target_objects_health_with_links": [
                'egg@egg_main',
                f'{robot_name}@gripper0_right_right_gripper',
                f'{robot_name}@gripper0_right_eef',
                f'{robot_name}@gripper0_right_leftfinger',
                f'{robot_name}@gripper0_right_finger_joint1_tip',
                f'{robot_name}@gripper0_right_rightfinger',
                f'{robot_name}@gripper0_right_finger_joint2_tip',
            ],
            "target_objects_health": [f'{robot_name}', 'egg'],
            "target_objects_forces": [
                f'{robot_name}@gripper0_right_right_gripper',
                "egg@egg_main",
                f'{robot_name}@gripper0_right_eef',
                f'{robot_name}@gripper0_right_leftfinger',
                f'{robot_name}@gripper0_right_finger_joint1_tip',
                f'{robot_name}@gripper0_right_rightfinger',
                f'{robot_name}@gripper0_right_finger_joint2_tip',
            ],
            "force_keys": ["filtered_qs_forces"],
        }
    raise ValueError(f"No visualization config for task: {task_name}. "
                     f"Add an entry to get_visualization_config() in playback_robocasa.py.")



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
    Sync damage evaluators' prev_body_velocities with the current simulation state.
    Prevents spurious impact detection when restoring states during playback.
    """
    for obj in env.get_damageable_objects():
        for evaluator in obj.damage_evaluators:
            if hasattr(evaluator, 'prev_body_velocities') and hasattr(evaluator, '_get_body_velocity'):
                for body_name in evaluator._get_object_body_ids():
                    evaluator.prev_body_velocities[body_name] = evaluator._get_body_velocity(body_name)


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
  python scripts/playback_robocasa.py --input pick_egg.hdf5 --output pick_egg_rendered.hdf5 --env pick_egg

  # Playback + render all cameras
  python scripts/playback_robocasa.py --input pick_egg.hdf5 --output pick_egg_rendered.hdf5 --env pick_egg --camera all_cameras

  # Playback + visualize and compute metrics
  python scripts/playback_robocasa.py --input pick_egg.hdf5 --output pick_egg_rendered.hdf5 --env pick_egg --visualize --metrics
        """
    )
    parser.add_argument("--input", required=True, help="Path to collected (teleop) HDF5 file")
    parser.add_argument("--output", required=True, help="Path for playback (rendered) HDF5 output file")
    parser.add_argument("--env", required=True, help="Environment name (e.g. pick_egg)")
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
        robot_name = "PandaOmron"
        camera_name = "robot0_agentview_right"
        output_video_dir = f"resources/videos/{os.path.splitext(os.path.basename(args.output))[0]}"
        os.makedirs(output_video_dir, exist_ok=True)

        visualization_config = get_visualization_config(args.env, robot_name)
        target_objects_health_with_links = visualization_config["target_objects_health_with_links"]
        target_objects_health = visualization_config["target_objects_health"]
        target_objects_forces = visualization_config["target_objects_forces"]
        force_keys = visualization_config["force_keys"]

        final_obj_healths = defaultdict(list)
        final_env_healths = []

        for idx in range(len(f["data"])):
            print(f"Episode: {idx}")
            demo_idx = int(list(f["data"].keys())[idx].split("_")[-1])

            obs_info_list = []
            for i in range(len(f[f"data/demo_{demo_idx}/info/obs_info"])):
                obs_info = json.loads(f[f"data/demo_{demo_idx}/info/obs_info"][i].decode("utf-8"))
                obs_info_list.append(obs_info)

            all_obj_healths = np.array(f[f"data/demo_{demo_idx}/obs/health"])
            health_list_link_names = f[f"data/demo_{demo_idx}"].attrs["health_list_link_names"]
            health = {}
            for obj_name in target_objects_health_with_links:
                indices = np.where(health_list_link_names == obj_name)[0]
                if len(indices) > 0:
                    health[obj_name] = all_obj_healths[:, indices[0]]

            for obj_name in target_objects_health:
                arrays = [v for k, v in health.items() if k.startswith(f"{obj_name}@")]
                health[obj_name] = np.minimum.reduce(arrays) if arrays else None

            if args.metrics:
                current_env_health = 0.0
                for obj_name in target_objects_health:
                    if health.get(obj_name) is not None:
                        final_obj_healths[obj_name].append(health[obj_name][-1])
                        print(f"  {obj_name} final health: {health[obj_name][-1]:.1f}%")
                        current_env_health += health[obj_name][-1]
                final_env_healths.append(current_env_health / len(target_objects_health))

            if args.visualize:
                imgs = f[f"data/demo_{demo_idx}/obs/{camera_name}_image"]
                new_imgs = []
                imgs_seg = f[f"data/demo_{demo_idx}/obs/{camera_name}_segmentation_class"]

                for i, img in enumerate(imgs):
                    img = cv2.cvtColor(img[:, :, :3], cv2.COLOR_RGB2BGR)
                    img_seg = imgs_seg[i]
                    obs_info = obs_info_list[i]
                    camera_type = "robot0"
                    for obj_name in target_objects_health:
                        if health.get(obj_name) is None or health[obj_name][i] >= 100:
                            continue
                        seg_instance_info = obs_info.get(camera_type, {}).get(camera_name, {}).get("seg_instance", {})
                        seg_key = next((k for k, v in seg_instance_info.items() if v == obj_name), None)
                        if seg_key is None:
                            continue
                        seg_instance_key = int(seg_key)
                        alpha = 1 - health[obj_name][i] / 100.0
                        overlay_color = np.array([0, 0, 255], dtype=np.uint8)
                        mask = img_seg == seg_instance_key
                        img[mask] = ((1 - alpha) * img[mask] + alpha * overlay_color).astype(np.uint8)
                    new_imgs.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

                imgs = np.array(new_imgs)

                # Save camera video
                camera_video_path = os.path.join(output_video_dir, f"demo_{demo_idx}_camera_video.mp4")
                save_rgb_camera_video(output_video_path=camera_video_path, imgs=imgs)

                # Save force video
                data = {}
                for obj_name in target_objects_forces:
                    data[obj_name] = {fk: [] for fk in force_keys}
                for i in range(len(f[f"data/demo_{demo_idx}/info/damage_info"])):
                    damage_info = json.loads(f[f"data/demo_{demo_idx}/info/damage_info"][i].decode("utf-8"))
                    for obj_name in target_objects_forces:
                        obj_part, link_part = obj_name.split("@")
                        for fk in force_keys:
                            try:
                                data[obj_name][fk].append(damage_info[obj_part][link_part]["mechanical"][fk])
                            except (KeyError, TypeError):
                                data[obj_name][fk].append(0.0)

                forces_video_path = os.path.join(output_video_dir, f"demo_{demo_idx}_forces_video.mp4")
                save_rgb_force_video(output_video_path=forces_video_path, imgs=imgs, target_objects=target_objects_forces, data=data, forces_to_plot=force_keys)

                health_video_path = os.path.join(output_video_dir, f"demo_{demo_idx}_health_video.mp4")
                save_rgb_health_video(output_video_path=health_video_path, imgs=imgs, target_objects=target_objects_health, health=health)

        f.close()

    if args.metrics:
        print(f"\n{'='*40}")
        print("Health Metrics Summary")
        print(f"{'='*40}")
        for obj_name in target_objects_health:
            if final_obj_healths[obj_name]:
                print(f"  {obj_name}: avg final health = {np.mean(final_obj_healths[obj_name]):.1f}%")
        if final_env_healths:
            print(f"  Overall env: avg final health = {np.mean(final_env_healths):.1f}%")

    print("\nPlayback complete.")
    playback_hdf5_file.close()


if __name__ == "__main__":
    main()
