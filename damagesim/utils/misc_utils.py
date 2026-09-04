"""
Shared utility functions for oopsieverse scripts.

Provides HDF5 trajectory processing, video saving, and live health
visualization helpers.
"""

import os
import cv2
import h5py
import json
import subprocess
import numpy as np
import matplotlib
import torch as th

# matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.animation as animation  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from collections import defaultdict
from typing import Dict, Optional


# ═══════════════════════════════════════════════════════════════════════
# JSON / Tensor helpers
# ═══════════════════════════════════════════════════════════════════════

def to_tensor(data):
    """Convert data to torch tensor if it's a numpy array or scalar."""
    if isinstance(data, th.Tensor):
        return data
    if isinstance(data, np.ndarray):
        if not data.flags["C_CONTIGUOUS"]:
            data = np.ascontiguousarray(data)
        return th.from_numpy(data)
    if isinstance(data, (int, float, bool)):
        return th.tensor(data)
    if isinstance(data, (list, tuple)):
        try:
            return th.tensor(data)
        except (ValueError, TypeError):
            return data
    return data


def json_default(o):
    """Custom JSON encoder for numpy/torch types."""
    if isinstance(o, (np.float32, np.float64, np.int32, np.int64, np.bool_)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, th.Tensor):
        return o.tolist()
    if isinstance(o, tuple):
        return list(o)
    if hasattr(o, "item"):
        try:
            return o.item()
        except Exception:
            pass
    raise TypeError(f"Object of type {type(o)} not JSON serializable")


# ═══════════════════════════════════════════════════════════════════════
# HDF5 trajectory processing
# ═══════════════════════════════════════════════════════════════════════

def process_traj_to_hdf5(
    env,
    traj_grp_name,
    traj_data,
    nested_keys=("obs", "info"),
    output_hdf5=None,
    compression=None,
):
    """
    Process trajectory data and store it in an HDF5 group.

    Args:
        env: The environment (unused, kept for API compatibility)
        traj_grp_name: Name of the trajectory group (e.g. "demo_0")
        traj_data: List of per-step dicts. Keys in ``nested_keys`` are
            themselves dicts of arrays; all other keys are flat arrays.
        nested_keys: Keys whose values are nested dicts (e.g. obs, info).
        output_hdf5: Open h5py.File to write into.
        compression: Optional dict of HDF5 compression kwargs.

    Returns:
        h5py.Group: The created trajectory group.
    """
    if compression is None:
        compression = {}

    nested_keys = set(nested_keys)
    data_grp = output_hdf5.require_group("data")
    traj_grp = data_grp.create_group(traj_grp_name)
    traj_grp.attrs["num_samples"] = len(traj_data)

    # ── Collect all data, converting arrays to tensors for uniform stacking ──
    data = defaultdict(list)
    for key in nested_keys:
        data[key] = defaultdict(list)

    for step_data in traj_data:
        for k, v in step_data.items():
            if k in nested_keys:
                for mod, step_mod_data in v.items():
                    data[k][mod].append(to_tensor(step_mod_data))
            else:
                data[k].append(to_tensor(v))

    # ── Serialize dicts and objects to JSON strings for HDF5 storage ──
    for k, v in data.items():
        if k == "info":
            for mod, traj_mod_data in v.items():
                data[k][mod] = [json.dumps(item, default=json_default) for item in traj_mod_data]
        elif k == "obs":
            for mod, traj_mod_data in v.items():
                if traj_mod_data and isinstance(traj_mod_data[0], dict):
                    data[k][mod] = [json.dumps(item, default=json_default) for item in traj_mod_data]

    # ── Write to HDF5 ──
    for k, dat in data.items():
        if not dat:
            continue
        if k in nested_keys:
            obs_grp = traj_grp.create_group(k)
            for mod, traj_mod_data in dat.items():
                try:
                    if traj_mod_data and isinstance(traj_mod_data[0], str):
                        dt = h5py.string_dtype(encoding="utf-8")
                        dset = obs_grp.create_dataset(mod, shape=(len(traj_mod_data),), dtype=dt)
                        dset[...] = traj_mod_data
                    else:
                        obs_grp.create_dataset(
                            mod,
                            data=th.stack(traj_mod_data, dim=0).cpu(),
                            **compression,
                        )
                except Exception as e:
                    print(f"Warning: could not save obs/{mod}: {e}")
        else:
            flat = th.stack(dat, dim=0) if isinstance(dat[0], th.Tensor) else th.tensor(dat)
            traj_grp.create_dataset(k, data=flat, **compression)

    return traj_grp


def flush_current_file(output_hdf5_file):
    """Flush HDF5 file to disk."""
    output_hdf5_file.flush()
    fd = output_hdf5_file.id.get_vfd_handle()
    os.fsync(fd)


def evaluate_task_completion(check_fn, env) -> bool:
    """Return ``bool(check_fn(env))``, or False if ``check_fn`` is missing/errors."""
    if check_fn is None:
        return False
    try:
        return bool(check_fn(env))
    except Exception as e:
        print(f"Warning: task_completion check failed: {e}")
        return False


def robocasa_check_success(env) -> bool:
    """Unwrap Robocasa wrappers and call ``_check_success`` if present.

    Robocasa ``_check_success`` is a bound method (takes only ``self``), unlike
    Behavior-1K ``task_completion_fn`` callables which take ``env``.
    """
    base = env
    while hasattr(base, "env"):
        base = base.env
    check = getattr(base, "_check_success", None)
    if check is None:
        return False
    try:
        return bool(check())
    except Exception as e:
        print(f"Warning: task_completion check failed: {e}")
        return False


def write_task_completion(traj_grp, values):
    """Write per-step bool ``task_completion`` dataset onto an open demo group."""
    arr = np.asarray(values, dtype=np.bool_)
    if "task_completion" in traj_grp:
        del traj_grp["task_completion"]
    traj_grp.create_dataset("task_completion", data=arr)


def write_reward_from_task_completion(traj_grp, values):
    """Overwrite per-step ``reward`` with 1.0 where ``task_completion`` is True."""
    arr = np.asarray(values, dtype=np.bool_).astype(np.float32)
    if "reward" in traj_grp:
        del traj_grp["reward"]
    traj_grp.create_dataset("reward", data=arr)


SAFE_ENV_HEALTH_THRESHOLD = 95.0


def read_final_task_completion(demo_grp) -> bool:
    """Return last-step ``task_completion`` from a demo group, or False if missing."""
    if "task_completion" not in demo_grp:
        return False
    arr = np.asarray(demo_grp["task_completion"])
    if arr.size == 0:
        return False
    return bool(arr.reshape(-1)[-1])


def is_safe_task_completion(task_completed: bool, env_health: float,
                            threshold: float = SAFE_ENV_HEALTH_THRESHOLD) -> bool:
    """True when the task succeeded and final env health is at/above ``threshold``."""
    return bool(task_completed) and float(env_health) >= float(threshold)


# ═══════════════════════════════════════════════════════════════════════
# Video saving
# ═══════════════════════════════════════════════════════════════════════

def save_rgb_camera_video(output_video_path, imgs, fps=30):
    """
    Save an array of RGB images as an MP4 video.

    Args:
        output_video_path: Path without extension (or with .mp4).
        imgs: (T, H, W, 3) uint8 RGB array.
        fps: Frames per second.
    """
    if len(imgs) == 0:
        return
    base = output_video_path.replace(".mp4", "").replace(".avi", "")
    avi_path = base + ".avi"
    mp4_path = base + ".mp4"

    h, w = imgs[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(avi_path, fourcc, fps, (w, h))
    for img in imgs:
        writer.write(cv2.cvtColor(img[:, :, :3], cv2.COLOR_RGB2BGR))
    writer.release()

    subprocess.run(
        [
            "ffmpeg", "-y", "-i", avi_path,
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-loglevel", "error", "-hide_banner", "-nostats",
            mp4_path,
        ],
        check=True,
    )
    os.remove(avi_path)


def save_rgb_force_video(
    output_video_path,
    imgs,
    target_objects,
    data,
    forces_to_plot=("dynamic_forces", "static_forces", "raw_forces_from_sim"),
    fps=30,
):
    """Save video with RGB frames alongside a force plot."""
    T = len(data[target_objects[0]][forces_to_plot[0]])

    fig = plt.figure(figsize=(14, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.2])

    ax_video = fig.add_subplot(gs[0, 0])
    ax_video.axis("off")
    video_im = ax_video.imshow(imgs[0][:, :, :3])

    ax_force = fig.add_subplot(gs[0, 1])
    ax_force.set_title("Force History")
    ax_force.set_xlabel("Time (s)")
    ax_force.set_ylabel("Force (N)")
    ax_force.set_xlim(0, T / fps)
    ax_force.set_ylim(0, 500.0)
    ax_force.grid(True)

    force_lines = {}
    for obj_name in target_objects:
        for force_key in forces_to_plot:
            force_lines[f"{obj_name}_{force_key}"], = ax_force.plot(
                [], [], lw=2, label=f"{obj_name} {force_key}"
            )
    ax_force.legend(loc="upper right", fontsize=9)
    fig.subplots_adjust(left=0.05, right=0.97, wspace=0.25)

    time_axis = [i / fps for i in range(T)]

    def init():
        video_im.set_data(imgs[0][:, :, :3])
        for line in force_lines.values():
            line.set_data([], [])
        return [video_im] + list(force_lines.values())

    def animate(i):
        video_im.set_data(imgs[i][:, :, :3])
        for obj_name in target_objects:
            for force_key in forces_to_plot:
                force_lines[f"{obj_name}_{force_key}"].set_data(
                    time_axis[: i + 1], data[obj_name][force_key][: i + 1]
                )
        return [video_im] + list(force_lines.values())

    ani = animation.FuncAnimation(
        fig, animate, init_func=init, frames=T, interval=1000 / fps, blit=True
    )
    writer = animation.FFMpegWriter(
        fps=fps,
        codec="libx264",
        extra_args=[
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            "-loglevel", "error", "-hide_banner", "-nostats",
        ],
    )
    ani.save(output_video_path, writer=writer)
    plt.close(fig)


def save_rgb_health_video(
    output_video_path,
    imgs,
    target_objects,
    health,
    fps=30,
):
    """Save video with RGB frames alongside a health-over-time plot."""
    T = len(health[target_objects[0]])

    fig = plt.figure(figsize=(14, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.2])

    ax_video = fig.add_subplot(gs[0, 0])
    ax_video.axis("off")
    video_im = ax_video.imshow(imgs[0][:, :, :3])

    ax_health = fig.add_subplot(gs[0, 1])
    ax_health.set_title("Health Over Time")
    ax_health.set_xlabel("Time (s)")
    ax_health.set_ylabel("Health")
    ax_health.set_xlim(0, T / fps)
    ax_health.set_ylim(-5.0, 105.0)
    ax_health.grid(True)

    health_lines = {}
    for obj_name in target_objects:
        health_lines[obj_name], = ax_health.plot([], [], lw=2, label=f"{obj_name} Health")
    ax_health.legend(loc="upper right", fontsize=9)
    fig.subplots_adjust(left=0.05, right=0.97, wspace=0.25)

    time_axis = [i / fps for i in range(T)]

    def init():
        video_im.set_data(imgs[0][:, :, :3])
        for line in health_lines.values():
            line.set_data([], [])
        return [video_im] + list(health_lines.values())

    def animate(i):
        video_im.set_data(imgs[i][:, :, :3])
        for obj_name in target_objects:
            health_lines[obj_name].set_data(time_axis[: i + 1], health[obj_name][: i + 1])
        return [video_im] + list(health_lines.values())

    ani = animation.FuncAnimation(
        fig, animate, init_func=init, frames=T, interval=1000 / fps, blit=True
    )
    writer = animation.FFMpegWriter(
        fps=fps,
        codec="libx264",
        extra_args=[
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            "-loglevel", "error", "-hide_banner", "-nostats",
        ],
    )
    ani.save(output_video_path, writer=writer)
    plt.close(fig)


def _load_og_ui_modules():
    """
    Lazy-import OmniGibson UI dependencies to avoid import-time overhead
    for callers that do not need viewport utilities.
    """
    import omnigibson as og
    import omnigibson.lazy as lazy
    from omnigibson.utils.ui_utils import dock_window

    return og, lazy, dock_window

def setup_viewport_layout(
):

    og, lazy, _ = _load_og_ui_modules()
    import omni.ui as ui
    from omni.kit.viewport.window import ViewportWindow
    viewports = [w for w in ui.Workspace.get_windows() if isinstance(w, ViewportWindow)]
    for w in ui.Workspace.get_windows():
        if isinstance(w, ViewportWindow):
            w.visible = True
        else:
            w.visible = False

    vp1 = ui.Workspace.get_window("Viewport 1")
    vp1.visible = False
    vp = ui.Workspace.get_window("Viewport")
    vp.height = 890
    vp.width = 1430
    for _ in range(10): og.sim.render()


def create_panda_eef_cylinders(
    robot,
    scene,
    width=0.01,
    lengths=(0.4, 0.4, 0.8),
    proportion_offsets=(0.0, 0.0, 0.5),
    colors=(
        (1.0, 0.0, 0.0),  # X-axis
        (0.0, 1.0, 0.0),  # Y-axis
        (0.0, 0.0, 1.0),  # Z-axis
    ),
    quat_offsets=None,
):
    """
    Create end-effector cylinder visualizers (X / Y / Z) for a Panda robot.

    Returns:
        dict: arm_name -> list[VisualGeomPrim] for the three axis cylinders.
    """
    _load_og_ui_modules()
    from omnigibson.prims import VisualGeomPrim
    from omnigibson.prims.material_prim import OmniPBRMaterialPrim
    from omnigibson.utils import transform_utils as T
    from omnigibson.utils.usd_utils import (
        create_primitive_mesh,
        absolute_prim_path_to_scene_relative,
    )

    if quat_offsets is None:
        quat_offsets = (
            T.euler2quat(th.tensor([0.0, th.pi / 2, 0.0])),
            T.euler2quat(th.tensor([-th.pi / 2, 0.0, 0.0])),
            T.euler2quat(th.tensor([0.0, 0.0, 0.0])),
        )

    color_tensors = tuple(th.as_tensor(c, dtype=th.float32) for c in colors)
    vis_geoms = {}

    arm_names = getattr(robot, "arm_names", ["arm"])
    for arm in arm_names:
        if arm not in robot.eef_links:
            continue
        hand_link = robot.eef_links[arm]
        arm_geoms = []
        for axis, length, color, prop_offset, quat_offset in zip(
            ("x", "y", "z"),
            lengths,
            color_tensors,
            proportion_offsets,
            quat_offsets,
        ):
            mat_prim_path = f"{robot.prim_path}/Looks/panda_eef_vis_{arm}_{axis}_mat"
            mat = OmniPBRMaterialPrim(
                relative_prim_path=absolute_prim_path_to_scene_relative(
                    scene, mat_prim_path
                ),
                name=f"{robot.name}:panda_eef_vis_{arm}_{axis}_mat",
            )
            mat.load(scene)
            mat.diffuse_color_constant = color

            vis_prim_path = f"{hand_link.prim_path}/panda_eef_vis_{axis}"
            create_primitive_mesh(
                vis_prim_path,
                "Cylinder",
                extents=1.0,
            )
            vis_geom = VisualGeomPrim(
                relative_prim_path=absolute_prim_path_to_scene_relative(
                    scene, vis_prim_path
                ),
                name=f"{robot.name}:arm_{arm}:panda_eef_vis_{axis}",
            )
            vis_geom.load(scene)
            vis_geom.material = mat
            vis_geom.scale = th.tensor([width, width, length], dtype=th.float32)
            vis_geom.set_position_orientation(
                position=th.tensor(
                    [0.0, 0.0, length * prop_offset], dtype=th.float32
                ),
                orientation=quat_offset,
                frame="parent",
            )
            arm_geoms.append(vis_geom)

        vis_geoms[arm] = arm_geoms

    return vis_geoms


def setup_robot_eef_visualization(robot, scene, arms=None, **cylinder_kwargs):
    """
    Attach RGB XYZ axis cylinders to each arm's EEF (Panda ``eef_link``, Tiago ``*_eef_link``, …).

    Tiago marks EEF links ``visible=False`` in ``_post_load``; this helper re-enables them so the
    cylinders show in the viewport (same idea as joylo ``setup_robot_visualizers``).
    """
    og, _, _ = _load_og_ui_modules()
    eef_vis = create_panda_eef_cylinders(robot, scene, **cylinder_kwargs)

    eef_links = getattr(robot, "eef_links", {})
    for arm, hand_link in eef_links.items():
        if arms is not None and arm not in arms:
            continue
        hand_link.visible = True
        hand_link.prim.GetAttribute("visibility").Set("inherited")

    # Franka / single-EEF robots that expose ``eef_link`` only via ``robot.links``
    if "eef_link" in robot.links and "eef_link" not in {lnk.prim_path for lnk in eef_links.values()}:
        robot.links["eef_link"].visible = True
        robot.links["eef_link"].prim.GetAttribute("visibility").Set("inherited")

    for geom_list in eef_vis.values():
        for geom in geom_list:
            geom.prim.GetAttribute("visibility").Set("inherited")
    for _ in range(10):
        og.sim.render()
    return eef_vis

