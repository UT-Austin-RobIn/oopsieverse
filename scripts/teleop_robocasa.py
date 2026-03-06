"""
Teleoperation script for oopsieverse environments.

Supports all registered environments with keyboard or SpaceMouse control
and damage data collection.

Usage::

    python scripts/teleop_robocasa.py --env ENV_NAME [--device DEVICE] [--output PATH]

Examples::

    python scripts/teleop_robocasa.py --env pick_egg --device keyboard
    python scripts/teleop_robocasa.py --env pick_egg --device spacemouse
    python scripts/teleop_robocasa.py --env pick_egg --device keyboard --output my_data.hdf5
"""

import os
import sys
import platform
import argparse
import time
import random
import json

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)
sys.path.insert(1, os.path.join(_project_root, "oopsiebench"))

import cv2
import h5py
import mujoco
import numpy as np
from robosuite.controllers import load_composite_controller_config
from robosuite.wrappers import DataCollectionWrapper
from robosuite.devices import Keyboard, SpaceMouse

from envs.registry import EnvironmentRegistry
from utils.misc_utils import process_traj_to_hdf5, flush_current_file
from damagesim.utils.visualization import render_health_bar_overlay, OBJ_NAME_DISPLAY_NAME_MAPPING


# ═══════════════════════════════════════════════════════════════════════
# Data Collection Wrapper
# ═══════════════════════════════════════════════════════════════════════

class DamageDataCollectionWrapper:
    """Wrapper that collects damage data during teleoperation.

    Args:
        env: The environment to wrap
        output_path: Path to HDF5 output file
        output_frequency: Show periodic output every N steps (default: 50, 0 to disable)
    """

    def __init__(self, env, output_path, output_frequency=50):
        self.env = env
        self.output_path = output_path
        self.output_frequency = output_frequency
        self.episode_count = 0
        self._reset_episode_buffer()

        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        self.output_hdf5_file = h5py.File(output_path, "w")

    def _reset_episode_buffer(self):
        self.episode_data = []

    def reset(self):
        self._reset_episode_buffer()
        result = self.env.reset()
        if isinstance(result, tuple):
            return result[0]
        return result

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        step_data = {
            "obs": obs,
            "states": self.env.sim.get_state().flatten(),
            "actions": action,
            "rewards": reward,
            "dones": done,
            "info": info,
        }
        self.episode_data.append(step_data)
        return obs, reward, done, info

    def flush_current_traj(self, nested_keys=("obs", "info")):
        traj_grp_name = f"demo_{self.episode_count}"
        traj_grp = process_traj_to_hdf5(
            self.env, traj_grp_name, self.episode_data,
            nested_keys=nested_keys, output_hdf5=self.output_hdf5_file
        )

        health_list = []
        for obj in self.env.get_damageable_objects():
            for link_name in obj.link_healths:
                health_list.append(f"{obj.name}@{link_name}")
        traj_grp.attrs["health_list_link_names"] = health_list

        flush_current_file(self.output_hdf5_file)
        print(f"Episode {self.episode_count} saved ({len(self.episode_data)} steps)")
        self.episode_count += 1

    def shutdown(self):
        self.env.close()

    def __getattr__(self, name):
        return getattr(self.env, name)


# ═══════════════════════════════════════════════════════════════════════
# Damage Visualization Components
# ═══════════════════════════════════════════════════════════════════════

class DamageColorManager:
    """
    Manages visual color feedback for damageable objects.
    Objects transition from their original color to bright red as health decreases.
    """

    def __init__(self, env):
        self.env = env
        self._original_colors = {}
        self._previous_healths = {}
        self._destroyed_objects = set()

    def _get_object_geoms(self, obj_name):
        """Get all visual geom IDs for an object."""
        geom_ids = []
        prefixes_to_check = [f"{obj_name}_"]

        for obj in self.env.get_damageable_objects():
            if hasattr(obj, 'name') and obj.name == obj_name:
                if hasattr(obj, 'naming_prefix') and obj.naming_prefix:
                    prefixes_to_check.append(obj.naming_prefix)
                if hasattr(obj, 'root_body') and obj.root_body:
                    root_prefix = obj.root_body.rsplit('_', 1)[0] + "_"
                    if root_prefix not in prefixes_to_check:
                        prefixes_to_check.append(root_prefix)
                break

        for i in range(self.env.sim.model.ngeom):
            geom_name = self.env.sim.model.geom_id2name(i)
            if geom_name:
                for prefix in prefixes_to_check:
                    if geom_name.startswith(prefix):
                        if self.env.sim.model.geom_group[i] == 1:
                            mat_id = self.env.sim.model.geom_matid[i]
                            orig_alpha = self.env.sim.model.geom_rgba[i][3]
                            if mat_id < 0 and orig_alpha < 0.01:
                                break
                            geom_ids.append(i)
                        break
        return geom_ids

    def initialize_colors(self):
        """Store original colors for all tracked damageable objects."""
        if self._original_colors:
            return

        for obj in self.env.get_damageable_objects():
            obj_name = getattr(obj, 'name', None)
            if not obj_name:
                continue
            geom_ids = self._get_object_geoms(obj_name)
            self._original_colors[obj_name] = {}
            for geom_id in geom_ids:
                colors = {'geom_rgba': self.env.sim.model.geom_rgba[geom_id].copy()}
                mat_id = self.env.sim.model.geom_matid[geom_id]
                if mat_id >= 0:
                    colors['mat_rgba'] = self.env.sim.model.mat_rgba[mat_id].copy()
                self._original_colors[obj_name][geom_id] = colors
            self._previous_healths[obj_name] = 100.0

    def reset(self):
        self._destroyed_objects.clear()
        self._original_colors.clear()
        self._previous_healths.clear()

    def _apply_damage_color(self, obj_name, alpha):
        if obj_name not in self._original_colors:
            return
        red_color = np.array([1.0, 0.0, 0.0, 1.0])
        for geom_id, colors in self._original_colors[obj_name].items():
            original_rgba = colors['geom_rgba']
            self.env.sim.model.geom_rgba[geom_id] = (1.0 - alpha) * original_rgba + alpha * red_color
            if 'mat_rgba' in colors:
                mat_id = self.env.sim.model.geom_matid[geom_id]
                self.env.sim.model.mat_rgba[mat_id] = (
                    (1.0 - alpha) * colors['mat_rgba'] + alpha * red_color
                )

    def update(self, health_states: dict):
        """
        Update visual feedback based on current health states.

        Args:
            health_states: {obj_name: health_pct (0.0–100.0)} from env.get_env_health()
        """
        if not self._original_colors:
            self.initialize_colors()

        events = {'damage_taken': {}, 'destroyed': []}

        for obj_name, health_pct in health_states.items():
            previous_health = self._previous_healths.get(obj_name, 100.0)
            alpha = 1.0 - (max(0.0, health_pct) / 100.0)
            self._apply_damage_color(obj_name, alpha)

            if health_pct <= 0 and obj_name not in self._destroyed_objects:
                self._destroyed_objects.add(obj_name)
                events['destroyed'].append(obj_name)
            elif health_pct < previous_health:
                events['damage_taken'][obj_name] = {
                    'amount': previous_health - health_pct,
                    'new_health': health_pct,
                }

            self._previous_healths[obj_name] = health_pct

        return events


class ConsoleHealthDisplay:
    """Displays health information in the console with visual health bars."""

    def __init__(self, bar_length=20):
        self.bar_length = bar_length

    def print_status(self, health_states: dict, step_num: int):
        """Print health status for all tracked damageable objects.

        Args:
            health_states: {obj_name: health_pct (0.0–100.0)} from env.get_env_health()
        """
        for obj_name, health_pct in health_states.items():
            filled = int((health_pct / 100.0) * self.bar_length)
            bar = "█" * filled + "░" * (self.bar_length - filled)
            print(f"[Step {step_num:5d}] {obj_name}: [{bar}] {health_pct:5.1f}%")

    def print_damage_event(self, obj_name, damage_amount, new_health):
        print(f"\nDAMAGE! {obj_name}: -{damage_amount:.1f} HP (Health: {new_health:.1f}%)")

    def print_destroyed(self, obj_name):
        print(f"\n{'='*50}")
        print(f" {obj_name.upper()} DESTROYED!")
        print(f"{'='*50}\n")



class LiveHUDRenderer:
    """
    Real-time HUD renderer that displays health bars in an OpenCV window
    during teleoperation, using MuJoCo's offscreen renderer.
    """

    DEFAULT_CAMERAS = [
        "robot0_agentview_right",
        "robot0_agentview_left",
        "robot0_agentview_center",
        "robot0_eye_in_hand",
        "frontview",
        "birdview",
        "sideview",
    ]

    def __init__(
        self,
        env,
        camera_name="robot0_agentview_right",
        width=1280,
        height=720,
        window_name="RoboCasa Teleop HUD",
        video_path=None,
        video_fps=30,
    ):
        self.env = env
        self.camera_name = camera_name
        self.width = width
        self.height = height
        self.window_name = window_name

        self._renderer = None
        self._scene_option = None
        self._window_created = False
        self._quit_requested = False
        self._available_cameras = []
        self._camera_index = 0

        self.video_fps = video_fps
        self._video_writer = None
        self._episode_frame_count = 0

    def _initialize_cameras(self):
        if self._available_cameras:
            return
        self._available_cameras = []
        model = self.env.sim.model
        for i in range(model.ncam):
            cam_name = model.camera_id2name(i)
            if cam_name:
                self._available_cameras.append(cam_name)
        if not self._available_cameras:
            self._available_cameras = list(self.DEFAULT_CAMERAS)
        if self.camera_name in self._available_cameras:
            self._camera_index = self._available_cameras.index(self.camera_name)
        else:
            self._available_cameras.insert(0, self.camera_name)
            self._camera_index = 0

    def switch_camera(self, direction):
        if not self._available_cameras:
            return
        self._camera_index = (self._camera_index + direction) % len(self._available_cameras)
        self.camera_name = self._available_cameras[self._camera_index]

    def initialize(self):
        if self.env.sim is not None:
            model = self.env.sim.model._model
            model.vis.global_.offwidth = max(model.vis.global_.offwidth, self.width)
            model.vis.global_.offheight = max(model.vis.global_.offheight, self.height)
            self._renderer = mujoco.Renderer(model, height=self.height, width=self.width)
            self._scene_option = mujoco.MjvOption()
            self._scene_option.geomgroup[0] = 0
            self._scene_option.geomgroup[1] = 1
            return True

    def start_video(self, video_path):
        if self._video_writer is not None:
            self.stop_video()
        video_dir = os.path.dirname(video_path)
        if video_dir:
            os.makedirs(video_dir, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self._video_writer = cv2.VideoWriter(video_path, fourcc, self.video_fps, (self.width, self.height))
        self._episode_frame_count = 0
        print(f"Started video recording: {video_path}")

    def stop_video(self):
        if self._video_writer is not None:
            self._video_writer.release()
            self._video_writer = None
            print(f"Video saved ({self._episode_frame_count} frames)")
            self._episode_frame_count = 0

    def render(self, health_states=None):
        """
        Render the scene with HUD overlay and display in OpenCV window.

        Args:
            health_states: {obj_name: health_pct} from env.get_env_health(), or None

        Returns:
            False if user pressed 'q' to quit, True otherwise.
        """
        if self._quit_requested:
            return False
        if not self._available_cameras:
            self._initialize_cameras()
        if self._renderer is None:
            if not self.initialize():
                return True

        if self._scene_option is not None:
            self._renderer.update_scene(self.env.sim.data._data, camera=self.camera_name, scene_option=self._scene_option)
        else:
            self._renderer.update_scene(self.env.sim.data._data, camera=self.camera_name)
        scene_rgb = self._renderer.render()
        scene_bgr = cv2.cvtColor(scene_rgb, cv2.COLOR_RGB2BGR)

        if health_states:
            scene_with_hud = render_health_bar_overlay(
                cv2.cvtColor(scene_bgr, cv2.COLOR_BGR2RGB),
                sorted(health_states.keys()),
                health_states,
                obj_display_names=OBJ_NAME_DISPLAY_NAME_MAPPING,
            )
            final_frame = cv2.cvtColor(scene_with_hud, cv2.COLOR_RGB2BGR)
        else:
            final_frame = scene_bgr

        if self._video_writer is not None:
            self._video_writer.write(final_frame)
            self._episode_frame_count += 1

        cv2.imshow(self.window_name, final_frame)
        if not self._window_created:
            cv2.setWindowProperty(self.window_name, cv2.WND_PROP_TOPMOST, 1)
            self._window_created = True

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self._quit_requested = True
            return False
        elif key == ord('['):
            self.switch_camera(-1)
        elif key == ord(']'):
            self.switch_camera(1)

        return True

    def is_quit_requested(self):
        return self._quit_requested

    def reset(self):
        self._renderer = None
        self._quit_requested = False
        self._available_cameras = []

    def close(self):
        self.stop_video()
        if self._window_created:
            cv2.destroyWindow(self.window_name)
            self._window_created = False
        self._renderer = None


# ═══════════════════════════════════════════════════════════════════════
# Manual Recording Wrapper
# ═══════════════════════════════════════════════════════════════════════

class ManualRecordingWrapper:
    """Wraps a device to add K-key episode-end and =-key pause via pynput."""

    def __init__(self, device):
        self.device = device
        self.end_recording_requested = False
        self.paused = False
        self.capture_camera_requested = False
        self.listener = None

        try:
            from pynput import keyboard
            self.listener = keyboard.Listener(on_press=self._on_press)
            self.listener.daemon = True
            self.listener.start()
        except ImportError:
            print("Warning: pynput not installed. 'K' key support disabled. Run: pip install pynput")
        except Exception as e:
            print(f"Warning: Could not start keyboard listener: {e}")
            print("'K' key support disabled. Use Ctrl+Q or Ctrl+C to quit.")

    def _on_press(self, key):
        try:
            if hasattr(key, "char") and key.char:
                if key.char.lower() == "k":
                    self.end_recording_requested = True
                    print("\n[K pressed - ending episode...]")
                elif key.char == "=":
                    if not self.paused:
                        self.paused = True
                        print("\n[= pressed - simulation PAUSED (press Esc to resume)]")
            else:
                from pynput.keyboard import Key
                if key == Key.esc and self.paused:
                    self.paused = False
                    print("\n[Esc pressed - simulation RESUMED]")
        except Exception:
            pass

    def cleanup(self):
        if self.listener:
            try:
                self.listener.stop()
                self.listener = None
            except Exception:
                pass

    def __getattr__(self, name):
        return getattr(self.device, name)


# ═══════════════════════════════════════════════════════════════════════
# Camera helpers
# ═══════════════════════════════════════════════════════════════════════

def save_camera_pose(env, env_name, output_dir="resources/camera_states"):
    """Save the current free camera pose to a JSON file."""
    if not hasattr(env, 'viewer') or env.viewer is None:
        print("Warning: No viewer available. Cannot save camera pose.")
        return None

    if not hasattr(env.viewer, 'viewer') or env.viewer.viewer is None:
        try:
            if hasattr(env, 'initialize_viewer'):
                env.initialize_viewer()
        except Exception:
            pass

    if not hasattr(env.viewer, 'viewer') or env.viewer.viewer is None:
        print("Warning: Viewer not initialized. Cannot save camera pose.")
        return None

    viewer_cam = env.viewer.viewer.cam
    if viewer_cam.type != 0:
        print("Warning: Camera is not in free mode. Switch to free camera (ESC) before saving.")
        return None

    os.makedirs(output_dir, exist_ok=True)
    camera_data = {
        "lookat": viewer_cam.lookat.tolist(),
        "distance": float(viewer_cam.distance),
        "azimuth": float(viewer_cam.azimuth),
        "elevation": float(viewer_cam.elevation),
    }
    filepath = os.path.join(output_dir, f"{env_name}.json")
    with open(filepath, 'w') as f:
        json.dump(camera_data, f, indent=2)

    print(f"[P pressed - saved free camera pose: {filepath}]")
    print(f"  distance={viewer_cam.distance:.2f}, azimuth={viewer_cam.azimuth:.1f}°, elevation={viewer_cam.elevation:.1f}°")
    return filepath


def load_camera_pose(env, env_name, camera_states_dir="resources/camera_states"):
    """Load a saved free camera pose and apply it to the viewer."""
    if not hasattr(env, 'viewer') or env.viewer is None:
        return False

    if not hasattr(env.viewer, 'viewer') or env.viewer.viewer is None:
        try:
            if hasattr(env, 'initialize_viewer'):
                env.initialize_viewer()
        except Exception:
            pass

    if not hasattr(env.viewer, 'viewer') or env.viewer.viewer is None:
        return False

    filepath = os.path.join(camera_states_dir, f"{env_name}.json")
    if not os.path.exists(filepath):
        return False

    with open(filepath, 'r') as f:
        camera_data = json.load(f)

    viewer_cam = env.viewer.viewer.cam
    viewer_cam.type = 0
    viewer_cam.lookat = np.array(camera_data["lookat"])
    viewer_cam.distance = camera_data["distance"]
    viewer_cam.azimuth = camera_data["azimuth"]
    viewer_cam.elevation = camera_data["elevation"]
    env.viewer.viewer.sync()

    print(f"Loaded camera pose from: {filepath}")
    return True


# ═══════════════════════════════════════════════════════════════════════
# Device creation
# ═══════════════════════════════════════════════════════════════════════

def create_device(device_type, env):
    """Create and wrap the appropriate device with ManualRecordingWrapper."""
    if device_type == "keyboard":
        device = Keyboard(env=env, pos_sensitivity=4.0, rot_sensitivity=4.0)
    elif device_type == "spacemouse":
        device = SpaceMouse(env=env, pos_sensitivity=2.0, rot_sensitivity=2.0)
    else:
        raise ValueError(f"Unknown device type: {device_type}")
    return ManualRecordingWrapper(device)


# ═══════════════════════════════════════════════════════════════════════
# Main teleoperation loop
# ═══════════════════════════════════════════════════════════════════════

def collect_human_trajectory(
    env,
    device_wrapper,
    arm,
    env_configuration,
    mirror_actions,
    render=True,
    max_fr=None,
    color_manager=None,
    console_display=None,
    console_freq=30,
    live_hud_renderer=None,
    env_name=None,
):
    """
    Collect a demonstration using the device (keyboard or SpaceMouse).

    Press 'K' to end the current recording and save.
    Press '=' to pause (free-camera). Press Esc to resume.
    Press Ctrl+Q to discard and quit.

    Returns:
        (ep_directory, discard_traj, task_success)
    """
    device_wrapper.end_recording_requested = False

    # ── Reset gripper state to fully open for new episode ──
    underlying_device = device_wrapper.device if hasattr(device_wrapper, 'device') else device_wrapper
    if hasattr(underlying_device, '_reset_gripper_positions'):
        underlying_device._reset_gripper_positions()
    else:
        if hasattr(underlying_device, 'grasp_states'):
            for robot_idx in range(len(underlying_device.grasp_states)):
                for arm_idx in range(len(underlying_device.grasp_states[robot_idx])):
                    underlying_device.grasp_states[robot_idx][arm_idx] = False
        if hasattr(underlying_device, 'single_click_and_hold'):
            underlying_device.single_click_and_hold = False

    env.reset()

    if render:
        env.render()

    if env_name is not None:
        load_camera_pose(env, env_name)

    all_prev_gripper_actions = [
        {
            f"{robot_arm}_gripper": np.repeat([0], robot.gripper[robot_arm].dof)
            for robot_arm in robot.arms
            if robot.gripper[robot_arm].dof > 0
        }
        for robot in env.robots
    ]

    # ── Initialize environment with a zero action ──
    zero_action = np.zeros(env.action_dim)
    if isinstance(env, (DataCollectionWrapper, DamageDataCollectionWrapper)):
        env.env.step(zero_action)
    else:
        env.step(zero_action)

    discard_traj = False
    task_success = False
    nonzero_ac_seen = False
    step_count = 0
    health_states = {}  # cached health for pause rendering

    if color_manager:
        color_manager.reset()
    if live_hud_renderer:
        live_hud_renderer.reset()

    # ── Hide teleop visualization markers ──
    for robot in env.robots:
        for arm_name in robot.arms:
            if robot.eef_site_id[arm_name] is not None:
                env.sim.model.site_rgba[robot.eef_site_id[arm_name]] = np.array([0., 0., 0., 0.])
            if robot.eef_cylinder_id[arm_name] is not None:
                env.sim.model.site_rgba[robot.eef_cylinder_id[arm_name]] = np.array([0., 0., 0., 0.])

    print("Ready for teleoperation...")
    while True:
        start = time.time()

        # ── Handle pause state — keep rendering, skip action processing ──
        if device_wrapper.paused:
            if render:
                env.render()
            if live_hud_renderer:
                if not live_hud_renderer.render(health_states):
                    return None, True, False
            time.sleep(0.05)
            continue

        active_robot_idx = device_wrapper.active_robot
        active_robot = env.robots[active_robot_idx]
        input_ac_dict = device_wrapper.input2action(mirror_actions=mirror_actions)

        if input_ac_dict is None:
            return None, True, False  # Ctrl+Q: discard immediately

        if device_wrapper.end_recording_requested:
            device_wrapper.end_recording_requested = False
            break

        if device_wrapper.capture_camera_requested:
            device_wrapper.capture_camera_requested = False
            if env_name is not None:
                save_camera_pose(env, env_name)

        action_dict = dict(input_ac_dict)
        for arm_name in active_robot.arms:
            controller_input_type = active_robot.part_controllers[arm_name].input_type
            if controller_input_type == "delta":
                action_dict[arm_name] = input_ac_dict[f"{arm_name}_delta"]
            elif controller_input_type == "absolute":
                action_dict[arm_name] = input_ac_dict[f"{arm_name}_abs"]
            else:
                raise ValueError(f"Unknown controller input type: {controller_input_type}")

        if not nonzero_ac_seen:
            if "right_delta" in action_dict and not np.all(action_dict["right_delta"] == 0):
                nonzero_ac_seen = True

        env_action = [
            robot.create_action_vector(all_prev_gripper_actions[i])
            for i, robot in enumerate(env.robots)
        ]
        env_action[active_robot_idx] = active_robot.create_action_vector(action_dict)
        env_action = np.concatenate(env_action)

        obs, _, _, _ = env.step(env_action)
        step_count += 1

        # ── Update damage state ──
        if color_manager or console_display or live_hud_renderer:
            health_states = env.get_env_health()

        if color_manager:
            events = color_manager.update(health_states)
            if console_display:
                for obj_name in events.get('destroyed', []):
                    console_display.print_destroyed(obj_name)
                for obj_name, damage_info in events.get('damage_taken', {}).items():
                    console_display.print_damage_event(obj_name, damage_info['amount'], damage_info['new_health'])

        if console_display and step_count % console_freq == 0:
            console_display.print_status(health_states, step_count)

        # ── Check for task success ──
        try:
            base_env = env
            while hasattr(base_env, 'env'):
                base_env = base_env.env
            if hasattr(base_env, '_check_success') and base_env._check_success():
                task_success = True
                print("\n[TASK SUCCESS DETECTED - Episode complete!]")
                break
        except Exception:
            pass

        if render:
            env.render()

        if live_hud_renderer:
            if not live_hud_renderer.render(health_states):
                return None, True, False

        if max_fr is not None:
            elapsed = time.time() - start
            diff = 1 / max_fr - elapsed
            if diff > 0:
                time.sleep(diff)

    ep_directory = None
    if nonzero_ac_seen and hasattr(env, "ep_directory"):
        ep_directory = env.ep_directory

    return ep_directory, discard_traj, task_success


def main():
    parser = argparse.ArgumentParser(
        description="Teleoperation for oopsieverse environments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  python scripts/teleop_robocasa.py --env pick_egg --device keyboard
  python scripts/teleop_robocasa.py --env pick_egg --device spacemouse
  python scripts/teleop_robocasa.py --env pick_egg --output my_data.hdf5

After collecting data, run playback:
  python scripts/playback_robocasa.py --input my_data.hdf5 --output my_data_rendered.hdf5 --env pick_egg

Available environments: {', '.join(EnvironmentRegistry.list_envs())}
        """
    )

    parser.add_argument("--env", required=True, choices=EnvironmentRegistry.list_envs(), help="Environment to run")
    parser.add_argument("--device", default="keyboard", choices=["keyboard", "spacemouse"], help="Input device (default: keyboard)")
    parser.add_argument("--output", help="Output HDF5 file path (default: resources/teleop_data/ENV_NAME.hdf5)")
    parser.add_argument("--n-episodes", type=int, default=15, help="Number of episodes to collect (default: 15)")
    parser.add_argument("--health-console", action="store_true", help="Enable console health status printing")
    parser.add_argument("--health-color", action="store_true", help="Enable damage color feedback (objects turn red when damaged)")
    parser.add_argument("--health-console-freq", type=int, default=30, help="Print health status every N steps (default: 30)")
    parser.add_argument("--health-hud", action="store_true", help="Enable live health bar HUD (OpenCV window)")
    parser.add_argument("--video", action="store_true", help="Record video (automatically enables --health-hud)")
    parser.add_argument("--video-fps", type=int, default=30, help="Video recording FPS (default: 30)")

    args = parser.parse_args()

    if args.video:
        args.health_hud = True

    # ── Check cv2.imshow availability ──
    # opencv-python-headless can overtake opencv-python
    # if running locally, remove GUI support.
    if args.health_hud:
        try:
            test_img = np.zeros((1, 1, 3), dtype=np.uint8)
            cv2.imshow("_probe", test_img)
            cv2.destroyWindow("_probe")
            cv2.waitKey(1)
        except cv2.error:
            print("Error: cv2.imshow is not available — your OpenCV build has no GUI support.")
            print("  This usually happens when opencv-python-headless shadows opencv-python.")
            print("  Fix with:")
            print("    pip uninstall -y opencv-python-headless opencv-python")
            print("    pip install opencv-python")
            sys.exit(1)

    # ── macOS GUI requirements ──
    # --health-hud / --video: cv2.imshow needs the main thread (no mjpython).
    # All other modes: MuJoCo viewer requires mjpython for OpenGL/GLFW.
    on_macos = platform.system() == "Darwin"
    if on_macos:
        using_mjpython = "MJPYTHON_BIN" in os.environ
        if args.health_hud:
            if using_mjpython:
                print("Error: --health-hud (and --video) require regular python on macOS, not mjpython.")
                print("  cv2.imshow needs the main thread, which mjpython gives to its run loop.")
                print(f"  python scripts/teleop_robocasa.py {' '.join(sys.argv[1:])}")
                sys.exit(1)
        else:
            if not using_mjpython:
                print("Error: On macOS this script must be run with mjpython, not python.")
                print(f"  mjpython scripts/teleop_robocasa.py {' '.join(sys.argv[1:])}")
                sys.exit(1)

    seed = random.randint(0, 1000000)
    print(f"Seed: {seed}")
    random.seed(seed)
    np.random.seed(seed)

    env_config = EnvironmentRegistry.get(args.env)
    output_path = args.output or f"resources/teleop_data/{args.env}.hdf5"

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"oopsieverse Teleoperation")
    print(f"{'='*60}")
    print(f"Environment : {args.env}")
    print(f"Device      : {args.device}")
    print(f"Robot       : {env_config.robot}")
    print(f"Output      : {output_path}")
    print(f"Display     : {'OpenCV HUD (with health bars)' if args.health_hud else 'MuJoCo Viewer'}")
    if args.video:
        video_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "resources", "videos")
        print(f"Video       : {video_dir}/")
    print(f"{'='*60}\n")

    env_kwargs = dict(
        robots=env_config.robot,
        controller_configs=load_composite_controller_config(robot=env_config.robot),
        translucent_robot=False,
        has_renderer=not args.health_hud,
        has_offscreen_renderer=not on_macos,
        render_camera=env_config.camera_name,
        ignore_done=True,
        use_camera_obs=False,
        render_segmentation=False,
        control_freq=env_config.control_freq,
    )
    if not args.health_hud:
        env_kwargs["renderer"] = "mjviewer"
    env = env_config.damageable_class(**env_kwargs)
    env.initialize_damageable_objects()

    color_manager = DamageColorManager(env) if (args.health_color or args.health_hud) else None
    console_display = ConsoleHealthDisplay() if args.health_console else None
    live_hud_renderer = None

    video_dir = None
    if args.video:
        video_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "resources", "videos")
        os.makedirs(video_dir, exist_ok=True)

    if args.health_hud:
        live_hud_renderer = LiveHUDRenderer(
            env=env,
            camera_name=env_config.camera_name,
            width=1280,
            height=720,
            window_name="oopsieverse Teleop HUD",
            video_fps=args.video_fps,
        )

    env = DamageDataCollectionWrapper(env=env, output_path=output_path)
    env.reset()  # Must call reset() before device.start_control() so robots are initialized
    device = create_device(args.device, env)
    device.start_control()

    console_freq = args.health_console_freq

    print(f"Starting teleoperation with {args.device}")
    print("\nHealth Tracking:")
    if args.health_hud:
        print("  ✓ Live health bar HUD (OpenCV window)")
    if args.health_color:
        print("  ✓ Damage color feedback (objects turn red when damaged)")
    if args.health_console:
        print(f"  ✓ Console health display (every {console_freq} steps)")
    if args.video:
        print(f"  ✓ Video recording (FPS: {args.video_fps})")
    if not args.health_color and not args.health_console and not args.health_hud:
        print("  (none — use --health-hud, --health-color, or --health-console)")
    print("\nControls:")
    print("  K          — end episode and save/discard prompt")
    print(f"  P          — save free camera pose (resources/camera_states/{args.env}.json)")
    if args.health_hud:
        print("  Q          — quit (in HUD window)")
        print("  [ / ]      — switch camera (in HUD window)")
    else:
        print("  Ctrl+Q     — quit teleoperation")
    print("  =          — pause simulation (free camera while paused)")
    print("  Esc        — resume simulation")
    print("  Ctrl+C     — force quit")
    print("\nEpisodes also end automatically on task success.\n")

    n_episodes = args.n_episodes
    completed_episodes = 0
    try:
        while completed_episodes < n_episodes:
            print(f"Episode {completed_episodes} starts (target: {n_episodes})")

            current_video_path = None
            if args.video and live_hud_renderer and video_dir is not None:
                current_video_path = os.path.join(video_dir, f"{args.env}.mp4")
                live_hud_renderer.start_video(current_video_path)

            try:
                ep_directory, discard_traj, task_success = collect_human_trajectory(
                    env, device, "right", "single-arm-opposed",
                    mirror_actions=True, render=not args.health_hud, max_fr=30,
                    color_manager=color_manager,
                    console_display=console_display,
                    console_freq=console_freq,
                    live_hud_renderer=live_hud_renderer,
                    env_name=args.env,
                )

                if args.video and live_hud_renderer:
                    live_hud_renderer.stop_video()

                if discard_traj:
                    print("\nQuitting teleoperation...")
                    if current_video_path and os.path.exists(current_video_path):
                        os.remove(current_video_path)
                    break

                print(f"\n{'='*60}")
                print(f"Episode {completed_episodes} ended")
                if task_success:
                    print("Task completed successfully!")
                print(f"{'='*60}")

                while True:
                    save_response = input("Save this episode to HDF5? (y/n): ").strip().lower()
                    if save_response in ('y', 'yes'):
                        env.flush_current_traj()
                        completed_episodes += 1
                        if current_video_path:
                            print(f"Video saved: {current_video_path}")
                        break
                    elif save_response in ('n', 'no'):
                        print("Episode discarded")
                        if current_video_path and os.path.exists(current_video_path):
                            os.remove(current_video_path)
                            print(f"Video deleted: {current_video_path}")
                        break
                    else:
                        print("Please enter 'y' or 'n'")

                print(f"\nStarting next episode (total saved: {completed_episodes})")

            except KeyboardInterrupt:
                if args.video and live_hud_renderer:
                    live_hud_renderer.stop_video()
                print("\nForce quitting teleoperation...")
                break

    finally:
        device.cleanup()
        if live_hud_renderer:
            live_hud_renderer.close()
        env.shutdown()

    print(f"\n{'='*60}")
    print(f"Teleoperation complete!")
    print(f"Total demonstrations recorded: {completed_episodes}")
    print(f"Data saved to: {output_path}")
    print(f"{'='*60}")
    print(f"\nNext step — run playback to render observations:")
    print(f"  mjpython scripts/playback_robocasa.py --input {output_path} --output <output_path> --env {args.env}")
    print()


if __name__ == "__main__":
    main()
