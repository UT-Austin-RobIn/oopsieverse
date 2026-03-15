"""
Action-only replay script to recollect trial HDF5 files with proper attributes.

Reads recorded actions from a trial HDF5 (which may be missing ep_meta/model_file
attributes or have a state size mismatch), resets a fresh env for each demo, steps
through the recorded actions, and writes a properly-formatted teleop HDF5 that
playback_robocasa.py can process.

Usage::

    python scripts/recollect_from_trial.py --input trial_1.hdf5 --output out.hdf5 --env pick_egg

Note: The kitchen layout and object positions will differ from the originals
(actions were teleoperated for a specific scene). Robot motions are preserved.
"""

import os
import sys
import json
import argparse

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)
sys.path.insert(1, os.path.join(_project_root, "oopsiebench"))

import h5py
import numpy as np
from robosuite.controllers import load_composite_controller_config

from envs.registry import EnvironmentRegistry
from utils.misc_utils import process_traj_to_hdf5, flush_current_file


def recollect_episode(src_f, demo_name, env, output_hdf5_file):
    """
    Replay recorded actions in a fresh env episode and save the result.

    Steps:
    1. Read all actions from the trial HDF5
    2. Reset env to get a fresh random scene
    3. Capture model_xml and ep_meta from the new scene
    4. Apply the teleop reset pattern (set_ep_meta → reset_from_xml_string → sim.reset → sim.forward)
    5. Hide teleop visualization markers
    6. Collect initial obs/info; build traj_data
    7. Step through each action; collect state, obs, reward, done, info
    8. Write to HDF5 via process_traj_to_hdf5
    9. Set model_file, ep_meta, health_list_link_names attributes
    10. Flush

    Args:
        src_f: Open h5py.File for the source trial HDF5
        demo_name: Name of the demo group (e.g. "demo_0")
        env: Initialized damageable env
        output_hdf5_file: Open h5py.File to write into

    Returns:
        Number of steps replayed, or None if skipped
    """
    demo_grp = src_f[f"data/{demo_name}"]
    if "actions" not in demo_grp:
        print(f"  Skipping {demo_name} — no actions key")
        return None

    actions = demo_grp["actions"][:]
    num_actions = len(actions)
    print(f"  {demo_name}: {num_actions} actions")

    # ── 1. Fresh env reset → new random scene ──
    env.reset()

    # ── 2. Capture model XML and ep_meta from the new scene ──
    model_xml = env.sim.model.get_xml()
    ep_meta = env.get_ep_meta()
    initial_state = env.sim.get_state().flatten()

    # ── 3. Apply teleop reset pattern (reloads XML so ep_meta is embedded) ──
    env.set_ep_meta(ep_meta)
    env.reset_from_xml_string(model_xml)
    env.sim.reset()
    env.sim.set_state_from_flattened(initial_state)
    env.sim.forward()

    # ── 4. Hide teleop visualization markers ──
    for robot in env.robots:
        for arm_name in robot.arms:
            if robot.eef_site_id[arm_name] is not None:
                env.sim.model.site_rgba[robot.eef_site_id[arm_name]] = np.array([0., 0., 0., 0.])
            if robot.eef_cylinder_id[arm_name] is not None:
                env.sim.model.site_rgba[robot.eef_cylinder_id[arm_name]] = np.array([0., 0., 0., 0.])

    # ── 5. Collect initial obs/info (before any action) ──
    obs, info = env.get_observations()
    traj_data = []

    # ── 6. Step through recorded actions ──
    for i, action in enumerate(actions):
        current_state = env.sim.get_state().flatten()
        obs, reward, done, info = env.step(action)

        traj_data.append({
            "obs": obs,
            "states": current_state,
            "actions": action,
            "rewards": reward,
            "dones": done,
            "info": info,
        })

        if (i + 1) % 100 == 0:
            print(f"    {i + 1}/{num_actions} steps")

    # ── 7. Write trajectory to HDF5 ──
    traj_grp = process_traj_to_hdf5(
        env, demo_name, traj_data,
        nested_keys=("obs", "info"),
        output_hdf5=output_hdf5_file,
    )

    # ── 8. Set required attributes ──
    traj_grp.attrs["model_file"] = model_xml
    traj_grp.attrs["ep_meta"] = json.dumps(ep_meta)

    health_list = []
    for obj in env.get_damageable_objects():
        for link_name in obj.link_healths:
            health_list.append(f"{obj.name}@{link_name}")
    traj_grp.attrs["health_list_link_names"] = health_list

    flush_current_file(output_hdf5_file)
    print(f"  {demo_name}: saved ({num_actions} steps)")
    return num_actions


def main():
    parser = argparse.ArgumentParser(
        description="Action-only replay to recollect trial HDF5 files with proper attributes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/recollect_from_trial.py \\
    --input trial_1.hdf5 \\
    --output resources/teleop_data/trial_1_recollected.hdf5 \\
    --env pick_egg

After recollecting, run playback:
  python scripts/playback_robocasa.py \\
    --input resources/teleop_data/trial_1_recollected.hdf5 \\
    --output resources/playback_data/trial_1_rendered.hdf5 \\
    --env pick_egg
        """
    )
    parser.add_argument("--input", required=True, help="Path to source trial HDF5 file")
    parser.add_argument("--output", required=True, help="Path for recollected output HDF5 file")
    parser.add_argument("--env", required=True, choices=EnvironmentRegistry.list_envs(), help="Environment name")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    env_config = EnvironmentRegistry.get(args.env)

    print(f"\n{'='*60}")
    print(f"oopsieverse Recollect from Trial")
    print(f"{'='*60}")
    print(f"Input  : {args.input}")
    print(f"Output : {args.output}")
    print(f"Env    : {args.env}")
    print(f"{'='*60}\n")

    # ── Build headless env (no rendering — output feeds into playback_robocasa.py) ──
    env = env_config.damageable_class(
        robots=env_config.robot,
        controller_configs=load_composite_controller_config(robot=env_config.robot),
        translucent_robot=False,
        has_renderer=False,
        has_offscreen_renderer=False,
        ignore_done=True,
        use_camera_obs=False,
        render_segmentation=False,
        control_freq=env_config.control_freq,
    )
    env.initialize_damageable_objects()

    src_f = h5py.File(args.input, "r")
    output_hdf5_file = h5py.File(args.output, "w")

    if "data" not in src_f:
        print("Error: No 'data' group found in input HDF5 file")
        src_f.close()
        output_hdf5_file.close()
        sys.exit(1)

    demos = list(src_f["data"].keys())
    print(f"Found demos: {demos}\n")

    total_steps = 0
    completed = 0
    for demo_num, demo_name in enumerate(demos):
        print(f"Demo {demo_num + 1}/{len(demos)}: {demo_name}")
        steps = recollect_episode(src_f, demo_name, env, output_hdf5_file)
        if steps is not None:
            total_steps += steps
            completed += 1

    src_f.close()
    output_hdf5_file.close()
    env.close()

    print(f"\n{'='*60}")
    print(f"Recollection complete!")
    print(f"Demos processed : {completed}/{len(demos)}")
    print(f"Total steps     : {total_steps}")
    print(f"Output          : {args.output}")
    print(f"{'='*60}")
    print(f"\nNext step — run playback to render observations:")
    print(f"  python scripts/playback_robocasa.py --input {args.output} --output <rendered.hdf5> --env {args.env}")
    print()


if __name__ == "__main__":
    main()
