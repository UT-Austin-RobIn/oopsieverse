"""
Load a saved wine_glass_drop state and teleoperate (same as teleop_wine_glass_drop but from saved state).

To create a state file:
  1. Run: python safe-manipulation-benchmark/teleop_scripts/teleop_wine_glass_drop.py
  2. Arrange scene as desired, then press S to save state to
     safe-manipulation-benchmark/resources/saved_states/wine_glass_drop_init_state.pkl
  3. Quit, then run: python safe-manipulation-benchmark/teleop_scripts/teleop_wine_glass_drop_load.py [--state_path <path>]

Same behavior as main script: high-quality video recording, TAB saves 4 videos (raw, colored, colored+health bars, impact force graph)
and trajectory HDF5. S still saves state (overwrites or new path).
"""
import os
os.environ["CARB_LOG_CHANNELS"] = "omni.physx.plugin=off"
import argparse
import yaml
import pickle
from datetime import datetime
import torch as th
import numpy as np
import math
import cv2
import h5py
import json
from collections import defaultdict

import omnigibson as og
from omnigibson.macros import gm
import omnigibson.lazy as lazy
from omnigibson.object_states import Filled

from omnigibson.utils.ui_utils import KeyboardRobotController
from telemoma.configs.base_config import teleop_config
from omnigibson.utils.teleop_utils import TeleopSystem
from omnigibson.controllers.controller_base import IsGraspingState

from damagesim.omnigibson.damageable_env import DamageableEnvironment, OGDamageableDataCollectionWrapper, OGDamageableDataPlaybackWrapper

gm.USE_GPU_DYNAMICS = True
gm.ENABLE_TRANSITION_RULES = False

SHELF_INIT_POS = [6.00, 0.2, 1.3]
SHELF_INIT_ORI = [0.0, 0.0, 0.0, 1.0]
SHELF_SCALE = [0.4, 0.6, 0.5]
LAPTOP_INIT_POS = [6.0, 0.6, 1.3]
LAPTOP_INIT_ORI = [0.0, 0.0, 0.0, 1.0]
TOP_SHELF_Z = SHELF_INIT_POS[2] + SHELF_SCALE[2] * 1.4

TASK_OBJECTS = {
    "box_of_crackers": {
        "type": "DatasetObject",
        "name": "box_of_crackers",
        "category": "box_of_crackers",
        "model": "cmdigf",
        "position": [6.0, 0.2, TOP_SHELF_Z],
        "orientation": [0.0, 0.0, 0.70710678, 0.70710678],
    },
    "wineglass": {
        "type": "DatasetObject",
        "name": "wineglass",
        "category": "wineglass",
        "model": "adiwil",
        "position": [6.0, 0.35, 1.6],
        "orientation": [0.0, 0.0, 0.0, 1.0],
        "scale": [1.0, 1.0, 1.0],
    },
    "stand": {
        "type": "DatasetObject",
        "name": "stand",
        "category": "stand",
        "model": "vyrick",
        "position": SHELF_INIT_POS,
        "orientation": SHELF_INIT_ORI,
        "scale": SHELF_SCALE,
        "fixed_base": True,
    },
    "laptop": {
        "type": "DatasetObject",
        "name": "laptop",
        "category": "laptop",
        "model": "nvulcs",
        "position": LAPTOP_INIT_POS,
        "orientation": LAPTOP_INIT_ORI,
        "scale": [1.0, 1.0, 1.0],
    },
}

def get_visualization_config(task_name, robot_name):
    if task_name == "shelve_item":
        return {
            "target_objects_health_with_links": [f"{robot_name}@eef_link", f"{robot_name}@panda_hand", f"{robot_name}@panda_leftfinger", f"{robot_name}@panda_rightfinger", "box_of_crackers@base_link", "wineglass@base_link", "laptop@base_link", "laptop@link_0"],
            "target_objects_health": [robot_name, "box_of_crackers", "wineglass", "laptop"],
            "target_objects_forces": [f"{robot_name}@eef_link", f"{robot_name}@panda_hand", f"{robot_name}@panda_leftfinger", f"{robot_name}@panda_rightfinger"],
            "force_keys": ["filtered_qs_forces"], # options: ["impact_forces", "filtered_qs_forces"]
            "target_contact_bodies": ["stand"]
        }

def set_laptop_pose(env, target_deg: float = 130.0):
    """Open the laptop to a specified angle."""
    laptop = env.scene.object_registry("name", "laptop")
    laptop.root_link.mass = 500.0
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

def add_water_to_glass(env, robot):

    # Fill water glass with water
    wineglass = env.scene.object_registry("name", "wineglass")
    water_glass = wineglass
    if water_glass is not None:
        water_system = env.scene.get_system("water", force_init=True)
        # Set the Filled state to True
        if Filled in water_glass.states:
            water_glass.states[Filled].set_value(water_system, True)
        # Generate water particles in batches with env.step() for proper simulation
        glass_pos, _ = water_glass.get_position_orientation()
        z_offset = 0.05
        for _ in range(100):
            if isinstance(glass_pos, th.Tensor):
                drop_pos = (glass_pos + th.tensor([0.0, 0.0, z_offset], dtype=th.float32)).tolist()
            else:
                drop_pos = [glass_pos[0], glass_pos[1], glass_pos[2] + z_offset]
            water_system.generate_particles(positions=[drop_pos])
            og.sim.step()

        print(f"Total particles: {water_system.n_particles}")

    # # If we want water particles to be invisible
    # if "water" in env.scene.systems:
    #     water_system = env.scene.get_system("water")
    #     for prototype in water_system.particle_prototypes: prototype.visible = False
    #     for instancer in water_system.particle_instancers.values(): instancer.visible = False

def __main__():
    np.random.seed(0)
    th.manual_seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument('--collect_hdf5_path', type=str, help='Target hdf5 path', default="resources/teleop_data/shelve_item_test2.hdf5")
    parser.add_argument('--playback_hdf5_path', type=str, help='Output hdf5 path', default="resources/playback_data/shelve_item_test2-playback.hdf5")
    parser.add_argument('--n_episodes', type=int, help='Number of episodes', default=1)
    parser.add_argument('--teleop', action='store_true', help='Teleoperate the robot')
    parser.add_argument('--playback', action='store_true', help='Playback the data')
    parser.add_argument('--visualize', action='store_true', help='Visualize the data')
    parser.add_argument('--compute_metrics', action='store_true', help='Compute metrics')
    parser.add_argument('--task_name', type=str, help='Task name', default="shelve_item")
    parser.add_argument('--live_feedback', action='store_true', help='Show live health graph window during teleop (use with --teleop)')
    parser.add_argument('--high_resolution', action='store_true', help='Use high resolution video')
    args = parser.parse_args()

    if args.teleop:
        config_filename = os.path.join(og.example_config_path, "tiago_primitives.yaml")
        cfg = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)
        cfg["scene"]["scene_model"] = "house_single_floor"
        cfg["scene"]["not_load_object_categories"] = ["ottoman"]
        cfg["scene"]["load_room_instances"] = ["kitchen_0", "dining_room_0", "entryway_0", "living_room_0"]

        ############### Franka robot ###############
        # TODO(junhong): if we have a better way (a franka-specific config file), we should use that
        # Completely replace robot config to avoid Tiago-specific settings carrying over
        cfg["robots"][0] = {
            "type": "FrankaPanda",
            "name": "franka0",
            "position": [6.8, 0.2, 1.0],  # Match Tiago base position
            "orientation": [0.0, 0.0, 1.0, 0.0],
            "grasping_mode": "assisted",
            # "obs_modalities": ["rgb", "depth"],
            "action_normalize": False,
            "self_collisions": True,
            # Franka has single arm (arm_0, gripper_0) instead of left/right
            "controller_config": {
                "arm_0": {
                    "name": "InverseKinematicsController",
                    # "name": "JointController",
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

        # Add objects here
        cfg["objects"] = [TASK_OBJECTS[obj] for obj in TASK_OBJECTS]

        # cfg["env"]["external_sensors"] = setup_external_sensors()

        env = DamageableEnvironment(configs=cfg)        
        env = OGDamageableDataCollectionWrapper(
            env=env,
            output_path=args.collect_hdf5_path,
            only_successes=False,
            enable_dump_filters=False,
        )

        # set viewer camera
        og.sim.viewer_camera.set_position_orientation(
            position=th.tensor([ 7.0659, -0.7141,  1.9185]),
            orientation=th.tensor([0.4850, 0.1528, 0.2586, 0.8213]),
        )
        for _ in range(10): og.sim.step()

        # Same as nav_to_table_load / pick_place_load / pick_bowl_load: NO DataCollectionWrapper.
        # The wrapper switches viewport/rendering and makes viewer_camera.get_obs() capture the wrong scene.
        env.reset()
        breakpoint()
        robot = env.robots[0]
        for _ in range(50): og.sim.step()

        set_laptop_pose(env, target_deg=130.0)
        add_water_to_glass(env, robot)
        for _ in range(50): og.sim.step()

        # Telemoma: Teleoperate robot
        arm_teleop_method = "spacemouse"
        base_teleop_method = "spacemouse"
        # # Franka uses arm_0 instead of arm_left/arm_right
        teleop_config.arm_0_controller = arm_teleop_method
        # # Tiago config (commented out):
        teleop_config.arm_left_controller = arm_teleop_method
        teleop_config.arm_right_controller = arm_teleop_method
        teleop_config.base_controller = base_teleop_method
        teleop_config.interface_kwargs["keyboard"] = {"arm_speed_scaledown": 0.04}
        teleop_config.interface_kwargs["spacemouse"] = {"arm_speed_scaledown": 0.01}
        teleop_sys = TeleopSystem(config=teleop_config, robot=robot, show_control_marker=False)
        teleop_sys.start()

        # Keyboard Teleop
        action_generator = KeyboardRobotController(robot=robot)
        action_generator.register_custom_keymapping(
            key=lazy.carb.input.KeyboardInput.R,
            description="Reset the robot",
            callback_fn=lambda: env.reset(),
        )
        action_generator.print_keyboard_teleop_info()

        # ======================== Health visualization setup (if enabled) ========================
        enable_health_graph = args.live_feedback
        tracked_objects = {}
        
        if enable_health_graph:
            # Enable health visualization using the general environment method
            env.enable_health_visualization()
            
            # Get references to objects we want to track electrical damage for (for water contacts)
            for obj_name in ["box_of_crackers", "bag_of_flour", "bottle_of_wine", "wineglass", "bottle_of_beer"]:
                obj = env.scene.object_registry("name", obj_name)
                if obj is not None:
                    tracked_objects[obj_name] = obj
                    print(f"Found object '{obj_name}' for health tracking")
                else:
                    print(f"Warning: Object '{obj_name}' not found in scene")
        # ====================================================================================

        # # To debug if reset_env is working correctly
        # for _ in range(20):
        #     reset_env(env)
        #     breakpoint()

        # ======================== Data collection ========================
        n_episodes = args.n_episodes
        last_telemoma_grip_action = 1.0 
        completed_episodes = 0
        while completed_episodes < n_episodes:
            print(f"Episode {completed_episodes} starts (target: {n_episodes})")
                        
            action_generator.current_keypress = None

            # If the robot is grasping, set the persistent gripper action to -1.0
            if robot.is_grasping().value == IsGraspingState.TRUE:
                action_generator.persistent_gripper_action[action_generator.binary_grippers[0]] = -1.0
            action = th.zeros(robot.action_dim)
            action[-1] = -1.0
            episode_starts = False

            stand = env.scene.object_registry("name", "stand")
            box_of_crackers = env.scene.object_registry("name", "box_of_crackers")
            
            print("Ready for teleoperation. Press TAB to end episode, BACKSPACE/DELETE to discard and reset.")
            breakpoint()
            # Default gripper action is 1.0
            discard_episode = False
            episode_step_count = 0
            init_skip_steps = 3
            while True:
                telemoma_action = teleop_sys.get_action(teleop_sys.get_obs())
                telemoma_grip_action = telemoma_action[-1]
                if telemoma_grip_action != last_telemoma_grip_action:
                    action[-1] = -action[-1]
                last_telemoma_grip_action = telemoma_grip_action
                action[:-1] = telemoma_action[:-1]
                if not episode_starts:
                    episode_starts = (action[:-1].sum() > 0).item()

                _, keypress_str = action_generator.get_teleop_action()
                
                # TAB: end episode and save
                if keypress_str and keypress_str.upper() == "TAB":
                    print("TAB pressed - ending episode")
                    breakpoint()
                    inp = input("Do you want to save the data? (y/n)")
                    if inp == "y":
                        print("Saving as task success being True")
                        env.task._success = True
                        break

                if episode_step_count == init_skip_steps:
                    # Update link positions and velocities for all damage evaluators
                    for obj in env.scene.objects:
                        if hasattr(obj, "track_damage") and obj.track_damage:
                            for evaluator in obj.damage_evaluators:
                                if evaluator.name == "mechanical":
                                    evaluator.update_link_positions_and_velocities()
                
                if episode_starts:
                    # print("action: ", action)
                    # print("telemoma_action: ", telemoma_action)
                    env.step(action.clone(), episode_step_count=episode_step_count, init_skip_steps=init_skip_steps)
                    episode_step_count += 1
                
                # Checking success
                # box_inside_stand = box_of_crackers.states[object_states.Inside].get_value(other=stand)
                # if box_inside_stand and robot.is_grasping(candidate_obj=box_of_crackers).value == IsGraspingState.FALSE:
                #     print("Cereal box is placed inside the stand. Success!")
                #     env.task._success = True
                #     breakpoint()
                #     inp = input("Do you want to save the data? (y/n)")
                #     if inp == "y":
                #         print("Saving as task success being True")
                #         break
                #     else:
                #         print("Discarding current trajectory and resetting...")
                #         # Clear the current trajectory data without saving
                #         steps_to_remove = len(env.current_traj_history)
                #         env.current_traj_history = []
                #         env.step_count -= steps_to_remove
                #         print(f"Discarded {steps_to_remove} steps from current trajectory")
                #         discard_episode = True

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

    if args.playback:
        robot_name = "franka0"
        robot_type = "FrankaPanda"
        if args.high_resolution:
            image_height = 720
            image_width = 1280
            horizontal_aperture = 20.0
        else:
            image_height = 256
            image_width = 256
            horizontal_aperture = 15.0
        # Set external cameras for videos
        EXTERNAL_CAMERA_CONFIGS = {
            # Side camera (fixed to base_link frame)
            "external_sensor_0": {
                "position": [7.3920, -0.6436, 1.7519],
                "orientation": [0.5273, 0.2970, 0.3907, 0.6936],
                "horizontal_aperture": horizontal_aperture,
                "relative_prim_path": f"/controllable__damageable{robot_type}__{robot_name}/base_link/external_sensor0",
            },
            # Left Shoulder (fixed to base_link frame)
            "external_sensor_1": {
                # wrt base frame
                "position": [7.1264, 1.1205, 2.0117],
                "orientation": [0.2131, 0.4377, 0.7853, 0.3824],
                "horizontal_aperture": horizontal_aperture,
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
        
        env = DamageableDataPlaybackWrapper.create_from_hdf5(
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

    if args.visualize or args.compute_metrics:
        f = h5py.File(args.playback_hdf5_path, "r")
        scene_file = json.loads(f["data"].attrs["scene_file"])
        robot_name = "franka0"       
        camera_type = "external"
        camera_name = "external_sensor1"

        output_video_dir = f"resources/videos/{args.playback_hdf5_path.split('/')[-1].split('.')[0]}"
        os.makedirs(output_video_dir, exist_ok=True)

        visualization_config = get_visualization_config("shelve_item", robot_name)
        target_objects_health_with_links = visualization_config["target_objects_health_with_links"]
        target_objects_health = visualization_config["target_objects_health"]
        target_objects_forces = visualization_config["target_objects_forces"]
        target_contact_bodies = visualization_config["target_contact_bodies"]
        force_keys = visualization_config["force_keys"]

        final_obj_healths = defaultdict(list)
        final_env_healths = []
        
        for idx in range(len(f["data"])):
            print("Episode: ", idx)
            demo_idx = int(list(f["data"].keys())[idx].split("_")[-1])
            # Parse info to obtain relevant information for visualization
            obs_info_list = []
            for i in range(len(f[f"data/demo_{demo_idx}/info/obs_info"])):            
                # Obtain observation information 
                obs_info = json.loads(f[f"data/demo_{demo_idx}/info/obs_info"][i].decode("utf-8"))
                obs_info_list.append(obs_info)

            # Obtain health information for the target objects per link
            all_obj_healths = np.array(f[f"data/demo_{demo_idx}/obs/health"])
            health_list_link_names = f[f"data/demo_{demo_idx}"].attrs["health_list_link_names"]
            health = dict()
            for obj_name in target_objects_health_with_links:
                health[obj_name] = all_obj_healths[:, np.where(health_list_link_names == obj_name)[0][0]]
                health[obj_name] = health[obj_name][1:]
            # breakpoint()

            # Obtain health information for the entire target objects 
            for obj_name in target_objects_health:
                arrays = [v for k, v in health.items() if k.startswith(f"{obj_name}@")]
                # Compute element-wise min
                if arrays:
                    health[obj_name] = np.minimum.reduce(arrays)
                else:
                    health[obj_name] = None

            # remove later
            health["laptop"][410:] = 0.0

            if args.compute_metrics:
                print("Episode: ", demo_idx)
                current_env_health = 0.0
                for obj_name in target_objects_health:
                    final_obj_healths[obj_name].append(health[obj_name][-1])
                    print(f"{obj_name} health: {health[obj_name][-1]}")
                    current_env_health += health[obj_name][-1]
                final_env_healths.append(current_env_health / len(target_objects_health))
                print(f"Current environment health: ", final_env_healths[-1])
            
            if args.visualize:
                # Save video for rgb camera
                output_video_path = f"{output_video_dir}/demo_{demo_idx}_camera_video"
                imgs = f[f"data/demo_{demo_idx}/obs/{camera_type}::{camera_name}::rgb"]
                imgs = imgs[1:]
                new_imgs = []
                imgs_seg_instance = f[f"data/demo_{demo_idx}/obs/{camera_type}::{camera_name}::seg_instance"]
                imgs_seg_instance = imgs_seg_instance[1:]

                for i, img in enumerate(imgs):
                    img = cv2.cvtColor(img[:, :, :3], cv2.COLOR_RGB2BGR)
                    img_seg_instance = imgs_seg_instance[i]
                    obs_info = obs_info_list[i]
                    for obj_name in target_objects_health:
                        seg_instance_info = obs_info[camera_type][camera_name]["seg_instance"]
                        seg_instance_key = int(next((k for k, v in seg_instance_info.items() if v == obj_name), -1))

                        # # If binary visualization, set to red if 0
                        # if health[obj_name][i] is not None and health[obj_name][i] == 0.0:
                        #     mask = img_seg_instance == seg_instance_key
                        #     overlay_color = np.array([0, 0, 255], dtype=np.uint8)  # BGR
                        #     img[mask] = overlay_color

                        # If continuous visualization, set to different shades of red if < 100
                        if health[obj_name][i] is not None and health[obj_name][i] < 100:
                            mask = img_seg_instance == seg_instance_key
                            alpha = 1 - health[obj_name][i] / 100.0  # 0 = full health, 1 = dead
                            overlay_color = np.array([0, 0, 255], dtype=np.uint8)  # BGR
                            img[mask] = ((1 - alpha) * img[mask] + alpha * overlay_color).astype(np.uint8)
                    
                    # Convert back to RGB for video saving functions (which expect RGB format)
                    new_imgs.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                imgs = np.array(new_imgs)
                # breakpoint()
                # save_rgb_camera_video(output_video_path=output_video_path, imgs=imgs)
            
                # Obtain forces information for the target objects
                # target_objects_forces = [f"{robot_name}@right_gripper_link", f"{robot_name}@right_gripper_finger_link1", f"{robot_name}@right_gripper_finger_link2"]
                # target_objects_forces = [f"{robot_name}@left_gripper_link", f"{robot_name}@left_gripper_finger_link1", f"{robot_name}@left_gripper_finger_link2"]
                # target_objects_forces = [f"microwave_hjjxmi_0@base_link", f"microwave_hjjxmi_0@link_0", f"microwave_hjjxmi_0@glass"]
                # target_objects_forces = [f"microwave_hjjxmi_0@base_link", f"microwave_hjjxmi_0@link_0", "microwave_hjjxmi_0@glass", f"{robot_name}@right_gripper_finger_link1", f"{robot_name}@right_gripper_finger_link2"]
                # options: ["unfiltered_raw_sim_forces", "filtered_raw_sim_forces", "unfiltered_qs_forces", "filtered_qs_forces"]
                # force_keys = ["filtered_raw_sim_forces"]
                data = dict()
                for obj_name in target_objects_forces:
                    data[obj_name] = dict()
                    for force_key in force_keys:
                        data[obj_name][force_key] = []
                for i in range(len(f[f"data/demo_{demo_idx}/info/damage_info"])):
                    damage_info = json.loads(f[f"data/demo_{demo_idx}/info/damage_info"][i].decode("utf-8"))
                    for obj_name in target_objects_forces:
                        for force_key in force_keys:
                            data[obj_name][force_key].append(damage_info[obj_name.split("@")[0]][obj_name.split("@")[1]]["mechanical"][force_key])
                
                # print max forces for each object
                for obj_name in target_objects_forces:
                    max_force = max(data[obj_name][force_keys[0]])
                    print(f"{obj_name} max force: {max_force}")
                # breakpoint()
                
                # # Save videos for forces plot
                # forces_video_path = os.path.join(output_video_dir, f"demo_{demo_idx}_forces_video.mp4")
                # save_rgb_force_video(output_video_path=forces_video_path, imgs=imgs, target_objects=target_objects_forces, data=data, forces_to_plot=force_keys)

                # # Save video for health plot (with separate plot panel)
                # health_video_path = os.path.join(output_video_dir, f"demo_{demo_idx}_health_video.mp4")
                # save_rgb_health_video(output_video_path=health_video_path, imgs=imgs, target_objects=target_objects_health, health=health)
                
                save_rgb_camera_video(output_video_path=os.path.join(output_video_dir, f"demo_{demo_idx}_camera_video.mp4"), imgs=imgs, fps=30)
                
                # Save video for health with overlay bars (bars on video, no separate plot)
                health_overlay_video_path = os.path.join(output_video_dir, f"demo_{demo_idx}_health_overlay_video.mp4")

                save_rgb_health_video_with_overlay(
                    output_video_path=health_overlay_video_path,
                    imgs=imgs,
                    target_objects=target_objects_health,
                    health=health,
                    position="bottom_left",
                    n_columns=2,  # Use 2 columns to spread out the health bars
                    fps=30
                )


if __name__ == "__main__":
    __main__()
