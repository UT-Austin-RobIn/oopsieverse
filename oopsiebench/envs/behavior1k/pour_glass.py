from ast import Pass
import os
import sys
os.environ["CARB_LOG_CHANNELS"] = "omni.physx.plugin=off"
import argparse
import yaml
import imageio
import json
import h5py
import pickle
import torch as th
import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import omnigibson as og
from omnigibson import object_states
from omnigibson.object_states import Filled
# from omnigibson.object_states import IsGrasped
from omnigibson.systems import FluidSystem
from omnigibson.macros import gm

# Commented out - not needed for keyboard teleoperation (causes mediapipe import error)
# from telemoma.configs.base_config import teleop_config
# from omnigibson.utils.teleop_utils import TeleopSystem
from omnigibson.utils.ui_utils import KeyboardRobotController
from omnigibson.envs import DataCollectionWrapper, DataPlaybackWrapper
import omnigibson.lazy as lazy
from omnigibson.controllers.controller_base import IsGraspingState
from omnigibson.utils.vision_utils import segmentation_to_rgb
import omnigibson.utils.transform_utils as T
import cv2
import subprocess

from damagesim.omnigibson.damageable_env import (
    OGDamageableEnvironment,
    OGDamageableDataCollectionWrapper,
    OGDamageableDataPlaybackWrapper,
)
from damagesim.utils.visualization import (
    save_rgb_health_video_with_overlay as save_rgb_health_video,
    save_rgb_camera_video,
)
from collections import defaultdict
import math


def create_panda_eef_cylinders(robot, scene):
    """No-op for compatibility; EEF cylinders not used in this codebase."""
    return {}


def save_rgb_water_contacts_video(output_video_path, imgs, target_objects, water_contacts, fps=30):
    """Save RGB video (water contacts overlay not implemented in this codebase)."""
    save_rgb_camera_video(output_video_path, imgs, fps=fps)


gm.USE_GPU_DYNAMICS = True
gm.ENABLE_TRANSITION_RULES = False


def save_camera_video(hdf5_file, output_video_path, robot_name, camera_type, camera_name, target_objects, obs_info_list, health, demo_idx=0):
    """Save RGB camera frames from HDF5 to video file."""
    imgs = hdf5_file[f"data/demo_{demo_idx}/obs/{camera_type}::{camera_name}::rgb"]
    imgs = np.array(imgs[1:])  # Skip first frame, convert to numpy
    
    if len(imgs) == 0:
        print(f"No images found for demo_{demo_idx}")
        return
    
    mp4_video = output_video_path + ".mp4"
    imageio.mimsave(mp4_video, imgs)
    print(f"Saved video: {mp4_video} ({len(imgs)} frames)")


def set_laptop_pose(env, target_deg: float = 130.0):
    """Open the laptop to a specified angle."""
    laptop = env.scene.object_registry("name", "laptop")
    if laptop is None:
        return
    target_rad = math.radians(float(target_deg))
    if hasattr(laptop, "joints"):
        for joint in laptop.joints.values():
            lo = joint.lower_limit
            hi = joint.upper_limit
            target = max(lo, min(hi, target_rad))
            joint.set_pos(target)
            joint.friction = 50000000.0
            if hasattr(joint, "keep_still"):
                joint.keep_still()
    if hasattr(laptop, "keep_still"):
        laptop.keep_still()


LAPTOP_INIT_POS = [6.3, 0.2, 1.3]
LAPTOP_INIT_ORI = [0.0, 0.0, 0.0, 1.0]
COFFEE_CUP_1_INIT_POS = [6.3, 0.0, 1.3]
WATER_GLASS_INIT_POS = [6.3, 0.5, 1.3]

# Task objects are located in BEHAVIOR-1k/datasets/objects/*
TASK_OBJECTS = {
    "laptop": {
        "type": "DatasetObject",
        "name": "laptop",
        "category": "laptop",
        "model": "nvulcs",
        "position": LAPTOP_INIT_POS,
        "orientation": LAPTOP_INIT_ORI,
        "scale": [1.0, 1.0, 1.0],
    },
    "coffee_cup_1": {
        "type": "DatasetObject",
        "name": "coffee_cup_1",
        "category": "coffee_cup",
        "model": "ckkwmj",
        "position": COFFEE_CUP_1_INIT_POS,
        "orientation": [0.0, 0.0, 0.0, 1.0],
        "scale": [1.2, 1.2, 1.2],
    },
    "water_glass": {
        "type": "DatasetObject",
        "name": "water_glass",
        "category": "water_glass",
        "model": "ewgotr",
        "position": WATER_GLASS_INIT_POS,
        "orientation": [0.0, 0.0, 0.0, 1.0],
        "scale": [0.9, 0.9, 0.9],
    },
}


def reset_env(env):
    env.reset()
    # load state
    with open(os.path.join(_REPO_ROOT, "oopsiebench", "envs", "behavior1k", "init_states", "pour_water.pkl"), "rb") as f:
        state_flat_array = pickle.load(f)
    og.sim.load_state(state_flat_array, serialized=True)
    
    # CRITICAL: Get all original positions IMMEDIATELY after loading state, before any sim steps
    # This prevents NaN quaternion issues from simulation instability
    robot = env.robots[0]
    robot_pos, robot_orn = robot.get_position_orientation()
    robot_joint_positions = robot.get_joint_positions()  # Save joint positions to restore later
    
    water_glass = env.scene.object_registry("name", "water_glass")
    if water_glass is not None:
        water_glass_pos, water_glass_orn = water_glass.get_position_orientation()
    else:
        water_glass_pos, water_glass_orn = None, None
    
    laptop = env.scene.object_registry("name", "laptop")
    coffee_cup = env.scene.object_registry("name", "coffee_cup_1")
    
    # Save original positions from the freshly loaded state (before any sim steps)
    laptop_orig_pos, laptop_orig_orn = None, None
    cup_orig_pos, cup_orig_orn = None, None
    if laptop is not None:
        laptop_orig_pos, laptop_orig_orn = laptop.get_position_orientation()
    if coffee_cup is not None:
        cup_orig_pos, cup_orig_orn = coffee_cup.get_position_orientation()
    
    # Now sync robot and run sim steps
    robot.keep_still()
    for _ in range(10):
        robot.keep_still()
        og.sim.step()
    
    # Open the laptop
    set_laptop_pose(env, target_deg=130.0)
    
    # Randomize positions of laptop and coffee_cup_1 using absolute position limits
    min_distance = 0.15  # Minimum distance to prevent collision
    
    # Absolute position limits for each object (x_min, x_max, y_min, y_max)
    # Based on initial positions: laptop [6.3, 0.2], coffee_cup [6.3, 0.0]
    laptop_limits = {"x": (6.25, 6.35), "y": (0.1, 0.2)}
    cup_limits = {"x": (6.25, 6.35), "y": (-0.1, 0.0)}
    
    # Retry loop: randomize both, check distance, retry if too close
    max_trials = 50
    for trial in range(max_trials):
        # Sample absolute positions within limits
        if laptop is not None:
            laptop_x = np.random.uniform(laptop_limits["x"][0], laptop_limits["x"][1])
            laptop_y = np.random.uniform(laptop_limits["y"][0], laptop_limits["y"][1])
            laptop_new_pos = laptop_orig_pos.clone()
            laptop_new_pos[0] = laptop_x
            laptop_new_pos[1] = laptop_y
        
        if coffee_cup is not None:
            cup_x = np.random.uniform(cup_limits["x"][0], cup_limits["x"][1])
            cup_y = np.random.uniform(cup_limits["y"][0], cup_limits["y"][1])
            cup_new_pos = cup_orig_pos.clone()
            cup_new_pos[0] = cup_x
            cup_new_pos[1] = cup_y
        
        # Check distance between laptop and coffee cup
        final_distance = None
        if laptop is not None and coffee_cup is not None:
            laptop_xy = laptop_new_pos[:2].cpu().numpy()
            cup_xy = cup_new_pos[:2].cpu().numpy()
            distance = np.linalg.norm(laptop_xy - cup_xy)
            
            if distance >= min_distance:
                # Distance OK, apply positions and break
                laptop.set_position_orientation(laptop_new_pos, laptop_orig_orn)
                coffee_cup.set_position_orientation(cup_new_pos, cup_orig_orn)
                final_distance = distance
                print(f"Randomization trial {trial}: distance={distance:.3f} >= {min_distance} (OK)")
                print(f"  laptop: pos=({laptop_x:.3f}, {laptop_y:.3f})")
                print(f"  coffee_cup: pos=({cup_x:.3f}, {cup_y:.3f})")
                break
            else:
                if trial < max_trials - 1:
                    print(f"Randomization trial {trial}: distance={distance:.3f} < {min_distance} (retrying...)")
        else:
            # Only one object exists, just apply and break
            if laptop is not None:
                laptop.set_position_orientation(laptop_new_pos, laptop_orig_orn)
                print(f"Randomizing laptop: pos=({laptop_x:.3f}, {laptop_y:.3f})")
            if coffee_cup is not None:
                coffee_cup.set_position_orientation(cup_new_pos, cup_orig_orn)
                print(f"Randomizing coffee_cup: pos=({cup_x:.3f}, {cup_y:.3f})")
            # If only one object, use a default distance for scale calculation
            final_distance = min_distance
            break
    else:
        # Max trials reached, use last positions anyway
        print(f"Warning: Could not satisfy distance constraint after {max_trials} trials, using last positions")
        if laptop is not None:
            laptop.set_position_orientation(laptop_new_pos, laptop_orig_orn)
        if coffee_cup is not None:
            coffee_cup.set_position_orientation(cup_new_pos, cup_orig_orn)
        # Calculate distance for scale even if constraint wasn't satisfied
        if laptop is not None and coffee_cup is not None:
            laptop_xy = laptop_new_pos[:2].cpu().numpy()
            cup_xy = cup_new_pos[:2].cpu().numpy()
            final_distance = np.linalg.norm(laptop_xy - cup_xy)
        else:
            final_distance = min_distance
    
    # Randomize laptop scale based on distance between objects
    # Further apart = larger scale allowed (up to 1.2), closer = smaller scale (min 0.9)
    if laptop is not None:
        old_scale = laptop.scale.tolist()
        # Dump state AFTER position randomization to preserve new positions
        temp_state = og.sim.dump_state(serialized=False)
        og.sim.stop()
        
        # Calculate max scale based on distance
        # Estimate max possible distance: sqrt((x_range)^2 + (y_range)^2)
        # x_range = 6.35 - 6.25 = 0.1, y_range = 0.25 - (-0.05) = 0.3
        max_possible_distance = np.sqrt(0.1**2 + 0.3**2)  # ~0.316
        # Map distance from [min_distance, max_possible_distance] to scale max [0.9, 1.2]
        if final_distance is not None:
            # Clamp distance to reasonable range
            distance_clamped = np.clip(final_distance, min_distance, max_possible_distance)
            # Linear interpolation: distance -> max_scale
            max_scale = 0.9 + (distance_clamped - min_distance) / (max_possible_distance - min_distance) * (1.1 - 0.9)
            max_scale = np.clip(max_scale, 0.9, 1.1)  # Ensure within bounds
        else:
            max_scale = 1.2  # Default if distance not calculated
        
        x_scale_mult = np.random.uniform(0.9, max_scale)
        y_scale_mult = np.random.uniform(0.9, max_scale)
        z_scale_mult = np.random.uniform(0.9, max_scale)
        new_scale = [old_scale[0] * x_scale_mult, old_scale[1] * y_scale_mult, old_scale[2] * z_scale_mult]
        laptop.scale = th.tensor(new_scale)
        print(f"Randomizing laptop scale: distance={final_distance:.3f}, max_scale={max_scale:.3f}, old={old_scale}, multipliers=[{x_scale_mult:.3f}, {y_scale_mult:.3f}, {z_scale_mult:.3f}], new={new_scale}")
        og.sim.play()
        og.sim.load_state(temp_state)
        # CRITICAL: Sync robot IK controller immediately after loading state
        robot.keep_still()

    for _ in range(10):
        robot.keep_still()
        og.sim.step()
    
    # Sync robot controller state after loading - prevents random movement
    robot.keep_still()
    og.sim.step()
    
    # Ensure gripper is closed to match saved grasping state
    # The saved state should have the grasping constraint, but we need to close the gripper controller
    # Create action to keep gripper closed and arm still
    keep_gripper_closed_action = th.zeros(robot.action_dim)
    keep_gripper_closed_action[robot.gripper_action_idx[robot.default_arm]] = -1.0  # Close gripper
    for _ in range(10):
        robot.set_joint_positions(robot_joint_positions)
        robot.set_joint_velocities(th.zeros(robot.n_dof))
        robot.keep_still()
        env.step(keep_gripper_closed_action)
        # Force restore after step to prevent any drift
        robot.set_joint_positions(robot_joint_positions)
        robot.set_joint_velocities(th.zeros(robot.n_dof))
        robot.keep_still()
    print("Gripper closed to match saved grasping state")
    
    # Fill water glass with water
    water_glass = env.scene.object_registry("name", "water_glass")
    if water_glass is not None:
        water_system = env.scene.get_system("water", force_init=True)
        # Set the Filled state to True
        if Filled in water_glass.states:
            water_glass.states[Filled].set_value(water_system, True)
        # Generate water particles in batches with env.step() for proper simulation
        glass_pos, _ = water_glass.get_position_orientation()
        z_offset = 0.05
        for _ in range(130):
            if isinstance(glass_pos, th.Tensor):
                drop_pos = (glass_pos + th.tensor([0.0, 0.0, z_offset], dtype=th.float32)).tolist()
            else:
                drop_pos = [glass_pos[0], glass_pos[1], glass_pos[2] + z_offset]
            water_system.generate_particles(positions=[drop_pos])
            robot.set_joint_positions(robot_joint_positions)
            robot.set_joint_velocities(th.zeros(robot.n_dof))
            robot.keep_still()
            env.step(keep_gripper_closed_action)
            # Force restore after EVERY step to guarantee no movement
            robot.set_joint_positions(robot_joint_positions)
            robot.set_joint_velocities(th.zeros(robot.n_dof))
            robot.keep_still()
        print(f"Generated 130 water particles in water glass. Total particles: {water_system.n_particles}")
    
    # Let simulation settle with env.step() for proper physics
    for _ in range(30):
        robot.set_joint_positions(robot_joint_positions)
        robot.set_joint_velocities(th.zeros(robot.n_dof))
        robot.keep_still()
        env.step(keep_gripper_closed_action)
        robot.set_joint_positions(robot_joint_positions)
        robot.set_joint_velocities(th.zeros(robot.n_dof))
        robot.keep_still()
    
    # Restore robot and water_glass positions to ensure they stay fixed across episodes
    robot.set_position_orientation(robot_pos, robot_orn)
    robot.set_joint_positions(robot_joint_positions)
    # CRITICAL: Zero out all joint velocities to prevent residual movement
    robot.set_joint_velocities(th.zeros(robot.n_dof))
    if water_glass is not None and water_glass_pos is not None:
        water_glass.set_position_orientation(water_glass_pos, water_glass_orn)
    
    # Reset the arm controller's internal state
    arm_controller = robot.controllers.get(f"arm_{robot.default_arm}")
    if arm_controller is not None:
        arm_controller.reset()
    
    # Final sync after position restoration - critical for IK controller
    robot.keep_still()
    for _ in range(10):
        robot.set_joint_positions(robot_joint_positions)
        robot.set_joint_velocities(th.zeros(robot.n_dof))
        robot.keep_still()
        og.sim.step()
    
    # One more keep_still to ensure controller is synced before teleoperation starts
    robot.keep_still()


def save_camera_images(env, output_dir=None):
    """
    Directly capture and save RGB + segmentation images from all cameras.
    """
    if output_dir is None:
        output_dir = os.path.join(_REPO_ROOT, "resources", "debug_images")
    os.makedirs(output_dir, exist_ok=True)
    robot = env.robots[0]
    
    # Get robot camera observations
    robot_obs, _ = robot.get_obs()
    for sensor_name, sensor_data in robot_obs.items():
        if isinstance(sensor_data, dict) and "rgb" in sensor_data and "seg_instance" in sensor_data:
            rgb = sensor_data["rgb"]
            seg = sensor_data["seg_instance"]
            
            # Convert to numpy
            rgb_np = rgb.cpu().numpy() if isinstance(rgb, th.Tensor) else np.array(rgb)
            if rgb_np.shape[-1] == 4:
                rgb_np = rgb_np[:, :, :3]
            if rgb_np.dtype != np.uint8:
                rgb_np = (rgb_np * 255).astype(np.uint8) if rgb_np.max() <= 1.0 else rgb_np.astype(np.uint8)
            
            # Convert seg to RGB visualization
            seg_tensor = seg if isinstance(seg, th.Tensor) else th.tensor(seg)
            seg_rgb = segmentation_to_rgb(seg_tensor, N=256)
            seg_np = seg_rgb.cpu().numpy() if isinstance(seg_rgb, th.Tensor) else np.array(seg_rgb)
            if seg_np.dtype != np.uint8:
                seg_np = (seg_np * 255).astype(np.uint8) if seg_np.max() <= 1.0 else seg_np.astype(np.uint8)
            
            # Resize if needed and combine side by side
            if rgb_np.shape[:2] != seg_np.shape[:2]:
                seg_np = cv2.resize(seg_np, (rgb_np.shape[1], rgb_np.shape[0]))
            combined = np.concatenate([rgb_np, seg_np], axis=1)
            
            # Save
            safe_name = sensor_name.replace(":", "_")
            filepath = os.path.join(output_dir, f"robot_{safe_name}_rgb_seg.png")
            cv2.imwrite(filepath, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
            print(f"Saved: {filepath}")
    
    # Get external sensor observations
    if hasattr(env, "external_sensors") and env.external_sensors:
        for sensor_name, sensor in env.external_sensors.items():
            sensor_obs, _ = sensor.get_obs()
            if "rgb" in sensor_obs and "seg_instance" in sensor_obs:
                rgb = sensor_obs["rgb"]
                seg = sensor_obs["seg_instance"]
                
                # Convert to numpy
                rgb_np = rgb.cpu().numpy() if isinstance(rgb, th.Tensor) else np.array(rgb)
                if rgb_np.shape[-1] == 4:
                    rgb_np = rgb_np[:, :, :3]
                if rgb_np.dtype != np.uint8:
                    rgb_np = (rgb_np * 255).astype(np.uint8) if rgb_np.max() <= 1.0 else rgb_np.astype(np.uint8)
                
                # Convert seg to RGB visualization
                seg_tensor = seg if isinstance(seg, th.Tensor) else th.tensor(seg)
                seg_rgb = segmentation_to_rgb(seg_tensor, N=256)
                seg_np = seg_rgb.cpu().numpy() if isinstance(seg_rgb, th.Tensor) else np.array(seg_rgb)
                if seg_np.dtype != np.uint8:
                    seg_np = (seg_np * 255).astype(np.uint8) if seg_np.max() <= 1.0 else seg_np.astype(np.uint8)
                
                # Resize if needed and combine side by side
                if rgb_np.shape[:2] != seg_np.shape[:2]:
                    seg_np = cv2.resize(seg_np, (rgb_np.shape[1], rgb_np.shape[0]))
                combined = np.concatenate([rgb_np, seg_np], axis=1)
                
                # Save
                filepath = os.path.join(output_dir, f"external_{sensor_name}_rgb_seg.png")
                cv2.imwrite(filepath, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
                print(f"Saved: {filepath}")
    
    print(f"Images saved to {output_dir}")


def __main__():
    np.random.seed(0)
    th.manual_seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument('--collect_hdf5_path', type=str, help='Target hdf5 path', default="resources/teleop_data/pour_glass.hdf5")
    parser.add_argument('--playback_hdf5_path', type=str, help='Output hdf5 path', default="resources/playback_data/pour_glass_playback.hdf5")
    parser.add_argument('--n_episodes', type=int, help='Number of episodes', default=1)
    parser.add_argument('--teleop', action='store_true', help='Teleoperate the robot')
    parser.add_argument('--playback', action='store_true', help='Playback the data')
    parser.add_argument('--visualize', action='store_true', help='Visualize the data')
    parser.add_argument('--seg', action='store_true', help='Run one teleop episode and save RGB + segmentation side-by-side video')
    parser.add_argument('--seg_output', type=str, default='resources/videos/pour_glass_seg.mp4', help='Output path for segmentation video')
    parser.add_argument('--health_graph', action='store_true', help='Show live health graph window during teleop (use with --teleop)')
    parser.add_argument('--health_video', action='store_true', help='Run one teleop episode and save health/force visualization videos')
    parser.add_argument('--health_video_output_dir', type=str, default='resources/videos/pour_glass_health', help='Output directory for health videos')
    args = parser.parse_args()

    if args.teleop:
        # TODO: Set this
        if not os.path.exists(args.collect_hdf5_path):
            os.makedirs(os.path.dirname(args.collect_hdf5_path), exist_ok=True)

        # Load the pre-selected configuration and set the online_sampling flag
        config_filename = os.path.join(og.example_config_path, "tiago_primitives.yaml")
        cfg = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

        # Overwrite any configs here
        cfg["scene"]["scene_model"] = "house_single_floor"
        cfg["scene"]["not_load_object_categories"] = ["ottoman"]
        cfg["scene"]["load_room_instances"] = ["kitchen_0", "dining_room_0", "entryway_0", "living_room_0"]
        
        ############### Franka robot ###############
        # Completely replace robot config to avoid Tiago-specific settings carrying over
        cfg["robots"][0] = {
            "type": "FrankaPanda",
            "name": "franka0",
            "position": [6.8, 0.2, 1.0],  # Match Tiago base position
            "orientation": [0.0, 0.0, 1.0, 0.0],
            "grasping_mode": "assisted",
            "obs_modalities": ["rgb", "depth", "seg_instance"],
            "action_normalize": False,
            "self_collisions": True,
            # Franka has single arm (arm_0, gripper_0) instead of left/right
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
        }

        # Generate external sensors config automatically
        # Get robot name and type from config to construct correct prim path
        robot_name = cfg["robots"][0].get("name", "franka0")
        robot_type = cfg["robots"][0].get("type", "FrankaPanda").lower()

        # Set external cameras for data collection
        EXTERNAL_CAMERA_CONFIGS = {
            # Side camera (fixed to base_link frame)
            "external_sensor_0": {
                "position": [7.3920, -0.6436, 1.7519],
                "orientation": [0.5273, 0.2970, 0.3907, 0.6936],
                "horizontal_aperture": 15.0,
                "relative_prim_path": f"/controllable__damageable{robot_type}__{robot_name}/base_link/external_sensor0",
            },
            # Left Shoulder (fixed to base_link frame)
            "external_sensor_1": {
                "position": [7.1264, 1.1205, 2.0117],
                "orientation": [0.2131, 0.4377, 0.7853, 0.3824],
                "horizontal_aperture": 15.0,
                "relative_prim_path": f"/controllable__damageable{robot_type}__{robot_name}/base_link/external_sensor1",
            },
        }
        external_sensors_config = []
        for name, camera_cfg in EXTERNAL_CAMERA_CONFIGS.items():
            i = name.split("_")[-1]
            position = camera_cfg["position"]
            orientation = camera_cfg["orientation"]
            external_sensors_config.append({
                "sensor_type": "VisionSensor",
                "name": f"external_sensor{i}",
                "relative_prim_path": camera_cfg["relative_prim_path"],
                "modalities": ["rgb", "seg_instance"],
                "sensor_kwargs": {
                    "image_height": 256,
                    "image_width": 256,
                    "horizontal_aperture": camera_cfg["horizontal_aperture"],
                },
                "position": th.tensor(position, dtype=th.float32),
                "orientation": th.tensor(orientation, dtype=th.float32),
                "pose_frame": "world",
            })
        cfg["env"]["external_sensors"] = external_sensors_config

        # Add objects here
        cfg["objects"] = [TASK_OBJECTS[obj] for obj in TASK_OBJECTS]

        env = OGDamageableEnvironment(configs=cfg)
        env = OGDamageableDataCollectionWrapper(
            env=env,
            output_path=args.collect_hdf5_path,
            only_successes=False,
            enable_dump_filters=False,
        )

        robot = env.robots[0]
        # set viewer camera
        og.sim.viewer_camera.set_position_orientation(
            position=th.tensor([ 7.0659, -0.7141,  1.9185]),
            orientation=th.tensor([0.4850, 0.1528, 0.2586, 0.8213]),
        )
        for _ in range(10): og.sim.step()

        # trying
        eef_vis = create_panda_eef_cylinders(robot, env.scene)
        robot.links["eef_link"].prim.GetAttribute("visibility").Set("inherited")
        for geom_list in eef_vis.values():
            for geom in geom_list:
                geom.prim.GetAttribute("visibility").Set("inherited")
        for _ in range(10): og.sim.render()

        # Keyboard Teleop only (telemoma commented out due to mediapipe issues)
        action_generator = KeyboardRobotController(robot=robot)
        action_generator.register_custom_keymapping(
            key=lazy.carb.input.KeyboardInput.R,
            description="Reset the robot",
            callback_fn=lambda: env.reset(),
        )
        
        # Initialize gripper to CLOSED state (-1.0) instead of default open (1.0)
        # This ensures gripper stays closed until user presses T to toggle
        for gripper_name in action_generator.binary_grippers:
            action_generator.gripper_direction[gripper_name] = -1.0
            action_generator.persistent_gripper_action[gripper_name] = -1.0
        
        action_generator.print_keyboard_teleop_info()

        # ======================== Health visualization setup (if enabled) ========================
        enable_health_graph = args.health_graph
        target_objects_water_contacts = ["laptop"]
        tracked_objects = {}
        water_contacts_data = defaultdict(list)  # object_name -> list of total particle counts
        
        if enable_health_graph:
            # Enable health visualization using the general environment method
            env.enable_health_visualization()
            
            # Get references to objects we want to track electrical damage for (for water contacts)
            for obj_name in target_objects_water_contacts:
                obj = env.scene.object_registry("name", obj_name)
                if obj is not None:
                    tracked_objects[obj_name] = obj
                    print(f"Found object '{obj_name}' for water contact tracking")
                else:
                    print(f"Warning: Object '{obj_name}' not found in scene")

        # ======================== Data collection ========================
        n_episodes = args.n_episodes
        completed_episodes = 0
        while completed_episodes < n_episodes:
            print(f"Episode {completed_episodes} starts (target: {n_episodes})")
            reset_env(env)
            
            # CRITICAL: Sync robot one more time after reset before teleop starts
            # Zero velocities and reset controller to prevent any residual movement
            robot.set_joint_velocities(th.zeros(robot.n_dof))
            arm_controller = robot.controllers.get(f"arm_{robot.default_arm}")
            if arm_controller is not None:
                arm_controller.reset()
            robot.keep_still()
            og.sim.step()
            robot.set_joint_velocities(th.zeros(robot.n_dof))
            robot.keep_still()
            
            # If the robot is grasping, ensure persistent gripper action is closed
            if robot.is_grasping().value == IsGraspingState.TRUE:
                action_generator.persistent_gripper_action[action_generator.binary_grippers[0]] = -1.0
                action_generator.gripper_direction[action_generator.binary_grippers[0]] = -1.0
            
            print("Ready for teleoperation. Press TAB to end episode, BACKSPACE/DELETE to discard and reset.")
            
            # Reset health tracking data for new episode
            if enable_health_graph:
                water_contacts_data.clear()
                # Health visualization is automatically reset in env.reset()
            
            discard_episode = False
            while True:
                ret = action_generator.get_teleop_action()
                if isinstance(ret, tuple) and len(ret) == 2:
                    action, keypress_str = ret
                else:
                    action = ret
                    keypress_str = None
                
                # TAB: end episode and save
                if keypress_str and keypress_str.upper() == "TAB":
                    print("TAB pressed - ending episode")
                    break
                
                # BACKSPACE/DELETE: discard current trajectory and reset
                if keypress_str and keypress_str.upper() in ("BACKSPACE", "DEL", "DELETE"):
                    print("BACKSPACE/DELETE pressed - discarding current trajectory and resetting...")
                    # Clear the current trajectory data without saving
                    steps_to_remove = len(env.current_traj_history)
                    env.current_traj_history = []
                    env.step_count -= steps_to_remove
                    print(f"Discarded {steps_to_remove} steps from current trajectory")
                    discard_episode = True
                    # Reset health tracking data
                    if enable_health_graph:
                        water_contacts_data.clear()
                        # Health visualization will be reset on next env.reset()
                    break
                
                obs, reward, terminated, truncated, info = env.step(action)
                
                # Health visualization updates automatically in env.step() if enabled
                # Track water particle contacts via electrical damage evaluators (if needed)
                if enable_health_graph:
                    for obj_name, obj in tracked_objects.items():
                        total_contacts = 0
                        # Find electrical damage evaluator for this object
                        for evaluator in getattr(obj, 'damage_evaluators', []):
                            if evaluator.name == "electrical":
                                # Get contact summary from the evaluator
                                contact_summary = evaluator.get_contact_summary()
                                total_contacts = contact_summary.get("total_contact", 0)
                                break
                        water_contacts_data[obj_name].append(total_contacts)
            
            # Only count completed episodes (not discarded ones)
            if not discard_episode:
                completed_episodes += 1
                print(f"Episode completed ({completed_episodes}/{n_episodes})")
            else:
                print(f"Episode discarded, redoing...")
        
        env.save_data()
        print("Data saved")
        
        # Close live health visualization if it was enabled
        if enable_health_graph:
            env.disable_health_visualization()
        
        og.shutdown()

    if args.playback:
        robot_name = "franka0"
        robot_type = "FrankaPanda"
        image_height = 256
        image_width = 256
        # Set external cameras for videos
        EXTERNAL_CAMERA_CONFIGS = {
            # Side camera (fixed to base_link frame)
            "external_sensor_0": {
                "position": [7.3920, -0.6436, 1.7519],
                "orientation": [0.5273, 0.2970, 0.3907, 0.6936],
                "horizontal_aperture": 15.0,
                "relative_prim_path": f"/controllable__damageable{robot_type.lower()}__{robot_name}/base_link/external_sensor0",
            },
            # Left Shoulder (fixed to base_link frame)
            "external_sensor_1": {
                "position": [7.1264, 1.1205, 2.0117],
                "orientation": [0.2131, 0.4377, 0.7853, 0.3824],
                "horizontal_aperture": 15.0,
                "relative_prim_path": f"/controllable__damageable{robot_type.lower()}__{robot_name}/base_link/external_sensor1",
            },
        }
        external_sensors_config = []
        for name, camera_cfg in EXTERNAL_CAMERA_CONFIGS.items():
            i = name.split("_")[-1]
            position = camera_cfg["position"]
            orientation = camera_cfg["orientation"]
            external_sensors_config.append({
                "sensor_type": "VisionSensor",
                "name": f"external_sensor{i}",
                "relative_prim_path": camera_cfg["relative_prim_path"],
                "modalities": ["rgb", "seg_instance"],
                "sensor_kwargs": {
                    "image_height": image_height,
                    "image_width": image_width,
                    "horizontal_aperture": camera_cfg["horizontal_aperture"],
                },
                "position": th.tensor(position, dtype=th.float32),
                "orientation": th.tensor(orientation, dtype=th.float32),
                "pose_frame": "world",
            })

        # In case we want to modify the robot sensors that were used during data collection
        robot_sensor_config = {
            "VisionSensor": {
                "modalities": ["rgb", "seg_instance"],
                "sensor_kwargs": {
                    "image_height": image_height,
                    "image_width": image_width,
                },
            },
        }
        
        env = OGDamageableDataPlaybackWrapper.create_from_hdf5(
            input_path=args.collect_hdf5_path,
            output_path=args.playback_hdf5_path,
            robot_obs_modalities=["proprio", "rgb", "seg_instance"],
            robot_sensor_config=robot_sensor_config,
            external_sensors_config=external_sensors_config,
            n_render_iterations=1,
            only_successes=False,
        )

        # set viewer camera
        og.sim.viewer_camera.set_position_orientation(
            position=th.tensor([ 7.0659, -0.7141,  1.9185]),
            orientation=th.tensor([0.4850, 0.1528, 0.2586, 0.8213]),
        )
        for _ in range(10): og.sim.step()

        # Playback the dataset
        env.playback_dataset(record_data=True)    
        env.save_data()
        og.shutdown()


    if args.visualize:
        f = h5py.File(args.playback_hdf5_path, "r")
        f_name = "pour_glass"
        scene_file = json.loads(f["data"].attrs["scene_file"])
        robot_name = "franka0"       
        camera_type = "external"
        camera_name = "external_sensor0"

        output_video_dir = "resources/videos"
        os.makedirs(output_video_dir, exist_ok=True)
        
        # Get all demo keys
        demo_keys = sorted([k for k in f["data"].keys() if k.startswith("demo_")])
        print(f"Found {len(demo_keys)} demos to visualize")
        
        # Save video for each demo
        for demo_idx, demo_key in enumerate(demo_keys):
            print(f"Processing {demo_key} ({demo_idx + 1}/{len(demo_keys)})")
            
            # Parse info to obtain relevant information for visualization
            obs_info_list = []
            obs_info_data = f[f"data/{demo_key}/info/obs_info"]
            for i in range(len(obs_info_data)):            
                obs_info = json.loads(obs_info_data[i].decode("utf-8"))
                obs_info_list.append(obs_info)

            health = []
            target_objects_health = []
            output_video_path = f"{output_video_dir}/{f_name}_demo_{demo_idx}_camera_video"
            save_camera_video(hdf5_file=f, 
                        output_video_path=output_video_path,
                        robot_name=robot_name, 
                        camera_type=camera_type, 
                        camera_name=camera_name,
                        target_objects=target_objects_health,
                        obs_info_list=obs_info_list,
                        health=health,
                        demo_idx=demo_idx)
        
        print(f"Saved {len(demo_keys)} videos to {output_video_dir}")

    if args.seg:
        # Load the pre-selected configuration
        config_filename = os.path.join(og.example_config_path, "tiago_primitives.yaml")
        cfg = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

        # Overwrite configs
        cfg["scene"]["scene_model"] = "house_single_floor"
        cfg["scene"]["not_load_object_categories"] = ["ottoman"]
        cfg["scene"]["load_room_instances"] = ["kitchen_0", "dining_room_0", "entryway_0", "living_room_0"]
        
        # Franka robot config
        cfg["robots"][0] = {
            "type": "FrankaPanda",
            "name": "franka0",
            "position": [6.8, 0.2, 1.0],
            "orientation": [0.0, 0.0, 1.0, 0.0],
            "grasping_mode": "assisted",
            "obs_modalities": ["rgb", "depth", "seg_instance"],
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
        }

        robot_name = cfg["robots"][0].get("name", "franka0")
        robot_type = cfg["robots"][0].get("type", "FrankaPanda").lower()

        # External camera for capturing video
        EXTERNAL_CAMERA_CONFIGS = {
            "external_sensor_0": {
                "position": [7.3920, -0.6436, 1.7519],
                "orientation": [0.5273, 0.2970, 0.3907, 0.6936],
                "horizontal_aperture": 15.0,
                "relative_prim_path": f"/controllable__damageable{robot_type}__{robot_name}/base_link/external_sensor0",
            },
        }
        external_sensors_config = []
        for name, camera_cfg in EXTERNAL_CAMERA_CONFIGS.items():
            i = name.split("_")[-1]
            external_sensors_config.append({
                "sensor_type": "VisionSensor",
                "name": f"external_sensor{i}",
                "relative_prim_path": camera_cfg["relative_prim_path"],
                "modalities": ["rgb", "seg_instance"],
                "sensor_kwargs": {
                    "image_height": 256,
                    "image_width": 256,
                    "horizontal_aperture": camera_cfg["horizontal_aperture"],
                },
                "position": th.tensor(camera_cfg["position"], dtype=th.float32),
                "orientation": th.tensor(camera_cfg["orientation"], dtype=th.float32),
                "pose_frame": "world",
            })
        cfg["env"]["external_sensors"] = external_sensors_config

        # Add objects
        cfg["objects"] = [TASK_OBJECTS[obj] for obj in TASK_OBJECTS]

        env = OGDamageableEnvironment(configs=cfg)
        robot = env.robots[0]
        
        # Set viewer camera
        og.sim.viewer_camera.set_position_orientation(
            position=th.tensor([7.0659, -0.7141, 1.9185]),
            orientation=th.tensor([0.4850, 0.1528, 0.2586, 0.8213]),
        )
        for _ in range(10): og.sim.step()

        # Create eef visualization
        eef_vis = create_panda_eef_cylinders(robot, env.scene)
        robot.links["eef_link"].prim.GetAttribute("visibility").Set("inherited")
        for geom_list in eef_vis.values():
            for geom in geom_list:
                geom.prim.GetAttribute("visibility").Set("inherited")
        for _ in range(10): og.sim.render()

        # Keyboard Teleop
        action_generator = KeyboardRobotController(robot=robot)
        action_generator.register_custom_keymapping(
            key=lazy.carb.input.KeyboardInput.R,
            description="Reset the robot",
            callback_fn=lambda: env.reset(),
        )
        
        # Initialize gripper to CLOSED state
        for gripper_name in action_generator.binary_grippers:
            action_generator.gripper_direction[gripper_name] = -1.0
            action_generator.persistent_gripper_action[gripper_name] = -1.0
        
        action_generator.print_keyboard_teleop_info()

        # Reset environment
        print("Resetting environment...")
        reset_env(env)
        
        # Sync robot after reset
        robot.set_joint_velocities(th.zeros(robot.n_dof))
        arm_controller = robot.controllers.get(f"arm_{robot.default_arm}")
        if arm_controller is not None:
            arm_controller.reset()
        robot.keep_still()
        og.sim.step()
        robot.set_joint_velocities(th.zeros(robot.n_dof))
        robot.keep_still()
        
        if robot.is_grasping().value == IsGraspingState.TRUE:
            action_generator.persistent_gripper_action[action_generator.binary_grippers[0]] = -1.0
            action_generator.gripper_direction[action_generator.binary_grippers[0]] = -1.0
        
        print("Reset complete. Starting frame capture.")
        print("Press TAB to end episode and save video, BACKSPACE to discard.")
        
        # Frame storage
        frames = []
        
        # Get external sensor for capturing
        external_sensor = None
        if hasattr(env, "external_sensors") and env.external_sensors:
            external_sensor = list(env.external_sensors.values())[0]
        
        # Teleop loop with frame capture
        while True:
            ret = action_generator.get_teleop_action()
            if isinstance(ret, tuple) and len(ret) == 2:
                action, keypress_str = ret
            else:
                action = ret
                keypress_str = None
            
            # TAB: end episode and save video
            if keypress_str and keypress_str.upper() == "TAB":
                print("TAB pressed - ending episode and saving video")
                break
            
            # BACKSPACE/DELETE: discard
            if keypress_str and keypress_str.upper() in ("BACKSPACE", "DEL", "DELETE"):
                print("BACKSPACE/DELETE pressed - discarding...")
                frames = []
                break
            
            # Step environment
            env.step(action)
            
            # Capture frame from external sensor
            if external_sensor is not None:
                sensor_obs, _ = external_sensor.get_obs()
                if "rgb" in sensor_obs and "seg_instance" in sensor_obs:
                    rgb = sensor_obs["rgb"]
                    seg = sensor_obs["seg_instance"]
                    
                    # Convert RGB to numpy
                    rgb_np = rgb.cpu().numpy() if isinstance(rgb, th.Tensor) else np.array(rgb)
                    if rgb_np.shape[-1] == 4:
                        rgb_np = rgb_np[:, :, :3]
                    if rgb_np.dtype != np.uint8:
                        rgb_np = (rgb_np * 255).astype(np.uint8) if rgb_np.max() <= 1.0 else rgb_np.astype(np.uint8)
                    
                    # Convert segmentation to RGB visualization
                    seg_tensor = seg if isinstance(seg, th.Tensor) else th.tensor(seg)
                    seg_rgb = segmentation_to_rgb(seg_tensor, N=256)
                    seg_np = seg_rgb.cpu().numpy() if isinstance(seg_rgb, th.Tensor) else np.array(seg_rgb)
                    if seg_np.dtype != np.uint8:
                        seg_np = (seg_np * 255).astype(np.uint8) if seg_np.max() <= 1.0 else seg_np.astype(np.uint8)
                    
                    # Resize seg if needed
                    if rgb_np.shape[:2] != seg_np.shape[:2]:
                        seg_np = cv2.resize(seg_np, (rgb_np.shape[1], rgb_np.shape[0]))
                    
                    # Combine side by side
                    combined = np.concatenate([rgb_np, seg_np], axis=1)
                    frames.append(combined)
            
            if len(frames) % 100 == 0 and len(frames) > 0:
                print(f"Captured {len(frames)} frames...")
        
        # Save video
        if len(frames) > 0:
            output_path = args.seg_output
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            imageio.mimsave(output_path, frames, fps=30)
            print(f"Saved video with {len(frames)} frames to: {output_path}")
        else:
            print("No frames captured, video not saved.")
        
        og.shutdown()

    if args.health_video:
        # Configuration for objects to track
        robot_name = "franka0"
        # Objects to track health for
        target_objects_health = ["laptop", "coffee_cup_1", "water_glass"]
        # Objects to track water particle contacts for (electrical damage)
        target_objects_water_contacts = ["laptop"]
        
        # Load configuration
        config_filename = os.path.join(og.example_config_path, "tiago_primitives.yaml")
        cfg = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

        cfg["scene"]["scene_model"] = "house_single_floor"
        cfg["scene"]["not_load_object_categories"] = ["ottoman"]
        cfg["scene"]["load_room_instances"] = ["kitchen_0", "dining_room_0", "entryway_0", "living_room_0"]
        
        # Franka robot config
        cfg["robots"][0] = {
            "type": "FrankaPanda",
            "name": "franka0",
            "position": [6.8, 0.2, 1.0],
            "orientation": [0.0, 0.0, 1.0, 0.0],
            "grasping_mode": "assisted",
            "obs_modalities": ["rgb", "depth", "seg_instance"],
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
        }

        robot_type = cfg["robots"][0].get("type", "FrankaPanda").lower()

        # External camera for capturing video
        EXTERNAL_CAMERA_CONFIGS = {
            "external_sensor_0": {
                "position": [7.3920, -0.6436, 1.7519],
                "orientation": [0.5273, 0.2970, 0.3907, 0.6936],
                "horizontal_aperture": 15.0,
                "relative_prim_path": f"/controllable__damageable{robot_type}__{robot_name}/base_link/external_sensor0",
            },
        }
        external_sensors_config = []
        for name, camera_cfg in EXTERNAL_CAMERA_CONFIGS.items():
            i = name.split("_")[-1]
            external_sensors_config.append({
                "sensor_type": "VisionSensor",
                "name": f"external_sensor{i}",
                "relative_prim_path": camera_cfg["relative_prim_path"],
                "modalities": ["rgb", "seg_instance"],
                "sensor_kwargs": {
                    "image_height": 256,
                    "image_width": 256,
                    "horizontal_aperture": camera_cfg["horizontal_aperture"],
                },
                "position": th.tensor(camera_cfg["position"], dtype=th.float32),
                "orientation": th.tensor(camera_cfg["orientation"], dtype=th.float32),
                "pose_frame": "world",
            })
        cfg["env"]["external_sensors"] = external_sensors_config

        # Add objects
        cfg["objects"] = [TASK_OBJECTS[obj] for obj in TASK_OBJECTS]

        # Create environment (not using DataCollectionWrapper)
        env = OGDamageableEnvironment(configs=cfg)
        robot = env.robots[0]
        
        # Set viewer camera
        og.sim.viewer_camera.set_position_orientation(
            position=th.tensor([7.0659, -0.7141, 1.9185]),
            orientation=th.tensor([0.4850, 0.1528, 0.2586, 0.8213]),
        )
        for _ in range(10): og.sim.step()

        # Create eef visualization
        eef_vis = create_panda_eef_cylinders(robot, env.scene)
        robot.links["eef_link"].prim.GetAttribute("visibility").Set("inherited")
        for geom_list in eef_vis.values():
            for geom in geom_list:
                geom.prim.GetAttribute("visibility").Set("inherited")
        for _ in range(10): og.sim.render()

        # Keyboard Teleop
        action_generator = KeyboardRobotController(robot=robot)
        action_generator.register_custom_keymapping(
            key=lazy.carb.input.KeyboardInput.R,
            description="Reset the robot",
            callback_fn=lambda: env.reset(),
        )
        
        # Initialize gripper to CLOSED state
        for gripper_name in action_generator.binary_grippers:
            action_generator.gripper_direction[gripper_name] = -1.0
            action_generator.persistent_gripper_action[gripper_name] = -1.0
        
        action_generator.print_keyboard_teleop_info()

        # Reset environment
        print("Resetting environment...")
        reset_env(env)
        
        # Sync robot after reset
        robot.set_joint_velocities(th.zeros(robot.n_dof))
        arm_controller = robot.controllers.get(f"arm_{robot.default_arm}")
        if arm_controller is not None:
            arm_controller.reset()
        robot.keep_still()
        og.sim.step()
        robot.set_joint_velocities(th.zeros(robot.n_dof))
        robot.keep_still()
        
        if robot.is_grasping().value == IsGraspingState.TRUE:
            action_generator.persistent_gripper_action[action_generator.binary_grippers[0]] = -1.0
            action_generator.gripper_direction[action_generator.binary_grippers[0]] = -1.0
        
        # Get references to objects we want to track electrical damage for
        # We'll access their electrical damage evaluators directly for water contact counts
        tracked_objects = {}
        for obj_name in target_objects_water_contacts:
            obj = env.scene.object_registry("name", obj_name)
            if obj is not None:
                tracked_objects[obj_name] = obj
                print(f"Found object '{obj_name}' for water contact tracking")
            else:
                print(f"Warning: Object '{obj_name}' not found in scene")
        
        print("Reset complete. Starting frame capture with health/water contact tracking.")
        print("Press TAB to end episode and save videos, BACKSPACE to discard.")
        
        # Enable health visualization using the general environment method
        env.enable_health_visualization()
        
        # Data storage
        frames = []
        health_data = defaultdict(list)  # object_name -> list of health values (for video generation)
        water_contacts_data = defaultdict(list)  # object_name -> list of total particle counts
        
        # Get external sensor for capturing
        external_sensor = None
        if hasattr(env, "external_sensors") and env.external_sensors:
            external_sensor = list(env.external_sensors.values())[0]
        
        # Build mapping from health index to object@link name (for video generation)
        health_link_names = env.health_list_link_names  # List of "object@link" strings
        
        # Teleop loop with health/water contact tracking
        while True:
            ret = action_generator.get_teleop_action()
            if isinstance(ret, tuple) and len(ret) == 2:
                action, keypress_str = ret
            else:
                action = ret
                keypress_str = None
            
            # TAB: end episode and save videos
            if keypress_str and keypress_str.upper() == "TAB":
                print("TAB pressed - ending episode and saving videos")
                break
            
            # BACKSPACE/DELETE: discard
            if keypress_str and keypress_str.upper() in ("BACKSPACE", "DEL", "DELETE"):
                print("BACKSPACE/DELETE pressed - discarding...")
                frames = []
                health_data.clear()
                water_contacts_data.clear()
                # Health visualization will be reset on next env.reset()
                break
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Capture frame from external sensor
            if external_sensor is not None:
                sensor_obs, _ = external_sensor.get_obs()
                if "rgb" in sensor_obs:
                    rgb = sensor_obs["rgb"]
                    rgb_np = rgb.cpu().numpy() if isinstance(rgb, th.Tensor) else np.array(rgb)
                    if rgb_np.shape[-1] == 4:
                        rgb_np = rgb_np[:, :, :3]
                    if rgb_np.dtype != np.uint8:
                        rgb_np = (rgb_np * 255).astype(np.uint8) if rgb_np.max() <= 1.0 else rgb_np.astype(np.uint8)
                    frames.append(rgb_np)
            
            # Track health per link (for video generation)
            # Health visualization updates automatically in env.step()
            health_array = obs.get("health", None)
            if health_array is not None:
                health_array = health_array.cpu().numpy() if isinstance(health_array, th.Tensor) else np.array(health_array)
                # Map health values to object@link names
                link_healths = {}
                for idx, link_name in enumerate(health_link_names):
                    if idx < len(health_array):
                        link_healths[link_name] = health_array[idx]
                
                # Aggregate health per object (min across links) for video generation
                for obj_name in target_objects_health:
                    obj_link_healths = [v for k, v in link_healths.items() if k.startswith(f"{obj_name}@")]
                    if obj_link_healths:
                        current_health = min(obj_link_healths)
                    else:
                        current_health = 100.0  # Default full health
                    health_data[obj_name].append(current_health)
            
            # Track water particle contacts via electrical damage evaluators
            for obj_name, obj in tracked_objects.items():
                total_contacts = 0
                # Find electrical damage evaluator for this object
                for evaluator in getattr(obj, 'damage_evaluators', []):
                    if evaluator.name == "electrical":
                        # Get contact summary from the evaluator
                        contact_summary = evaluator.get_contact_summary()
                        total_contacts = contact_summary.get("total_contact", 0)
                        break
                water_contacts_data[obj_name].append(total_contacts)
            
            if len(frames) % 100 == 0 and len(frames) > 0:
                # Print current water contact status
                contact_status = ", ".join([f"{k}: {v[-1]}" for k, v in water_contacts_data.items() if v])
                print(f"Captured {len(frames)} frames... Water contacts: {contact_status}")
        
        # Save videos
        if len(frames) > 0:
            output_dir = args.health_video_output_dir
            os.makedirs(output_dir, exist_ok=True)
            
            imgs = np.array(frames)
            
            # Save health video
            health_video_path = os.path.join(output_dir, "health_video.mp4")
            # Convert health_data from defaultdict to regular dict for the function
            health_dict = {k: np.array(v) for k, v in health_data.items()}
            print(f"Saving health video with {len(target_objects_health)} objects...")
            save_rgb_health_video(
                output_video_path=health_video_path,
                imgs=imgs,
                target_objects=target_objects_health,
                health=health_dict,
                fps=30
            )
            print(f"Saved health video to: {health_video_path}")
            
            # Save water contacts video
            if water_contacts_data:
                water_video_path = os.path.join(output_dir, "water_contacts_video.mp4")
                # Convert water_contacts_data to regular dict with lists
                water_dict = {k: list(v) for k, v in water_contacts_data.items()}
                print(f"Saving water contacts video with {len(target_objects_water_contacts)} objects...")
                save_rgb_water_contacts_video(
                    output_video_path=water_video_path,
                    imgs=imgs,
                    target_objects=target_objects_water_contacts,
                    water_contacts=water_dict,
                    fps=30
                )
                print(f"Saved water contacts video to: {water_video_path}")
            
            # Also save raw camera video
            raw_video_path = os.path.join(output_dir, "camera_video.mp4")
            imageio.mimsave(raw_video_path, imgs, fps=30)
            print(f"Saved raw camera video to: {raw_video_path}")
            
            print(f"\nAll videos saved to: {output_dir}")
        else:
            print("No frames captured, videos not saved.")
        
        # Close live health visualization
        env.disable_health_visualization()
        
        og.shutdown()


if __name__ == "__main__":
    __main__()