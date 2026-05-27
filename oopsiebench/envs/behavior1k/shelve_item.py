"""
Task configuration for **shelve_item**.

Scene : house_single_floor
Robot : FrankaPanda (franka0)
Damage: mechanical only (impact + quasistatic)
"""
import pickle
import numpy as np
import torch as th
import omnigibson as og
from omnigibson.utils import transform_utils as T
from scipy.spatial.transform import Rotation as R
from omnigibson import object_states
from omnigibson.controllers.controller_base import IsGraspingState

from oopsiebench.envs.behavior1k.base import TaskConfig
from oopsiebench.envs.behavior1k.spatial_checks import gripper_far_from_object

ROBOT_NAME = "franka0"
ROBOT_TYPE = "FrankaPanda"

# ── Task objects ────────────────────────────────────────────────────────

TASK_OBJECTS = {
    "box_of_crackers": {
        "type": "DatasetObject",
        "name": "box_of_crackers",
        "category": "box_of_crackers",
        "model": "cmdigf",
        "position": [6.0, 0.2, 2.0],
        "orientation": [0.0, 0.0, 0.70710678, 0.70710678],
    },
    "bag_of_flour": {
        "type": "DatasetObject",
        "name": "book",
        "category": "bag_of_flour",
        "model": "rlejxx",
        "position": [6.00, 0.35, 1.35],
        "orientation": [0.0, 0.0, 0.0, 1.0],
        "scale": [1.0, 1.0, 0.9],
    },
    "bottle_of_wine": {
        "type": "DatasetObject",
        "name": "bottle_of_wine",
        "category": "bottle_of_wine",
        "model": "hnkiog",
        "position": [6.00, 0.2, 1.35],
        "orientation": [0.0, 0.0, 0.0, 1.0],
        "scale": [1.0, 1.0, 1.0],
    },
    "wineglass": {
        "type": "DatasetObject",
        "name": "wineglass",
        "category": "wineglass",
        "model": "adiwil",
        "position": [6.00, 0.12, 1.35],
        "orientation": [0.0, 0.0, 0.0, 1.0],
        "scale": [1.0, 1.0, 1.0],
    },
    "bottle_of_beer": {
        "type": "DatasetObject",
        "name": "bottle_of_beer",
        "category": "bottle_of_beer",
        "model": "dqfsgv",
        "position": [6.00, 0.08, 1.35],
        "orientation": [0.0, 0.0, 0.0, 1.0],
        "scale": [1.0, 1.0, 1.0],
    },
    "stand": {
        "type": "DatasetObject",
        "name": "stand",
        "category": "stand",
        "model": "vyrick",
        "position": [6.00, 0.2, 1.35],
        "orientation": [0.0, 0.0, 0.0, 1.0],
        "scale": [0.3, 0.7, 0.5],
        "fixed_base": True,
    },
}

# ── Cameras ─────────────────────────────────────────────────────────────

VIEWER_CAMERA_POS = [7.0659, -0.7141, 1.9185]
VIEWER_CAMERA_ORN = [0.4850, 0.1528, 0.2586, 0.8213]

EXTERNAL_CAMERA_CONFIGS = {
    "external_sensor_0": {
        "position": [7.3920, -0.6436, 1.7519],
        "orientation": [0.5273, 0.2970, 0.3907, 0.6936],
        "horizontal_aperture": 15.0,
    },
    "external_sensor_1": {
        "position": [7.1264, 1.1205, 2.0117],
        "orientation": [0.2131, 0.4377, 0.7853, 0.3824],
        "horizontal_aperture": 15.0,
    },
}

INIT_STATE_PATH = "resources/init_states/shelve_item.pkl"

# ── Public entry point ──────────────────────────────────────────────────

def get_task_config() -> TaskConfig:
    return TaskConfig(
        task_name="shelve_item",

        # OG macros
        use_gpu_dynamics=False,
        enable_transition_rules=False,

        # Scene
        scene_config={
            "scene_model": "house_single_floor",
            "not_load_object_categories": ["ottoman"],
            "load_room_instances": [
                "kitchen_0", "dining_room_0", "entryway_0", "living_room_0",
            ],
        },

        # Robot
        robot_name=ROBOT_NAME,
        robot_type=ROBOT_TYPE,
        robot_config={
            "type": ROBOT_TYPE,
            "name": ROBOT_NAME,
            "position": [6.8, 0.2, 1.0],
            "orientation": [0.0, 0.0, 1.0, 0.0],
            "grasping_mode": "assisted",
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
        },

        # Objects
        task_objects=TASK_OBJECTS,

        # Cameras
        viewer_camera_pos=VIEWER_CAMERA_POS,
        viewer_camera_orn=VIEWER_CAMERA_ORN,
        external_camera_configs=EXTERNAL_CAMERA_CONFIGS,

        # Visualization
        target_objects_health_with_links=[
            f"{ROBOT_NAME}@eef_link",
            f"{ROBOT_NAME}@panda_hand",
            f"{ROBOT_NAME}@panda_leftfinger",
            f"{ROBOT_NAME}@panda_rightfinger",
            "box_of_crackers@base_link",
            "book@base_link",
            "bottle_of_wine@base_link",
            "wineglass@base_link",
            "bottle_of_beer@base_link",
        ],
        target_objects_health=[
            ROBOT_NAME,
            "box_of_crackers",
            "book",
            "bottle_of_wine",
            "wineglass",
            "bottle_of_beer",
        ],
        target_objects_forces=[
            f"{ROBOT_NAME}@eef_link",
            f"{ROBOT_NAME}@panda_hand",
            f"{ROBOT_NAME}@panda_leftfinger",
            f"{ROBOT_NAME}@panda_rightfinger",
            "box_of_crackers@base_link",
            "book@base_link",
            "bottle_of_wine@base_link",
            "wineglass@base_link",
            "bottle_of_beer@base_link",
        ],
        force_keys=["filtered_qs_forces", "impact_forces"],
        target_contact_bodies=["stand"],

        # Default paths
        default_collect_hdf5="demos/behavior1k/teleop_data/shelve_item.hdf5",
        default_playback_hdf5="demos/behavior1k/playback_data/shelve_item_playback.hdf5",
        default_video_dir="demos/behavior1k/playback_videos/shelve_item",
    )

def check_object_upright(obj):
    q = obj.get_position_orientation()[1]
    r = R.from_quat(q)

    # Rotate the up vector
    up_rotated = r.apply([0, 0, 1])
    z_alignment = up_rotated[2]  # should be close to 1 if not toppled

    threshold = 0.995  # cos(small angle) ~1
    upright = z_alignment > threshold
    
    return upright

_U_XY = 0.03
_U_YAW = 0.12
_U_ARM = 0.07

def reset(env):
    env.reset()

    flour = env.scene.object_registry("name", "book")
    wineglass = env.scene.object_registry("name", "wineglass")
    winebottle = env.scene.object_registry("name", "bottle_of_wine")
    beerbottle = env.scene.object_registry("name", "bottle_of_beer")
    stand = env.scene.object_registry("name", "stand")

    objects = [flour, wineglass, winebottle, beerbottle]
    trial_number = 0
    while True:
        print("Reset trial number: ", trial_number)

        # Load initial state
        with open(INIT_STATE_PATH, "rb") as f: state_flat_array = pickle.load(f)
        og.sim.load_state(state_flat_array, serialized=True)

        # Reset the robot's position and orientation
        if not getattr(env, "robots", None):
            return
        robot = env.robots[0]
        pos, orn = robot.get_position_orientation()
        pos = pos.clone()
        pos[0] += float(np.random.uniform(-_U_XY, _U_XY))
        pos[1] += float(np.random.uniform(-_U_XY, _U_XY))
        euler = T.quat2euler(orn).clone()
        euler[2] = euler[2] + float(np.random.uniform(-_U_YAW, _U_YAW))
        orn = T.euler2quat(euler)
        robot.set_position_orientation(pos, orn)

        # Reset the robot's arm joints
        q = robot.get_joint_positions().clone()
        for arm_name in robot.arm_control_idx:
            idx = robot.arm_control_idx[arm_name]
            u = (th.rand(len(idx), device=q.device, dtype=q.dtype) * 2 - 1) * _U_ARM
            q[idx] = q[idx] + u
        robot.set_joint_positions(q)
        robot.set_joint_velocities(th.zeros(robot.n_dof, device=q.device, dtype=q.dtype))
        robot.keep_still()
        for _ in range(8):
            og.sim.step()

        # Randomize object positions
        for obj in objects:
            pos, orn = obj.get_position_orientation()
            pos_magnitude = [-0.05, 0.05] 
            rot_magnitude = np.pi / 12 # 15 degrees
            pos_diff_xy = np.random.uniform(pos_magnitude[0], pos_magnitude[1], size=2)
            pos_diff = th.from_numpy(np.concatenate([pos_diff_xy, np.zeros(1)])).float()
            new_pos = pos + pos_diff
            orn_diff = th.from_numpy(np.array([0.0, 0.0, np.random.uniform(-rot_magnitude, rot_magnitude)]))
            new_orn = T.mat2quat(T.euler2mat(orn_diff) @ T.quat2mat(orn))
            obj.set_position_orientation(new_pos, new_orn)

        # Randomize object scales
        temp_state = og.sim.dump_state(serialized=False)
        object_scales = {obj.name: obj.scale.tolist() for obj in objects}
        og.sim.stop()
        for obj in objects:
            x_scale_magnitude = np.random.uniform(0.9, 1.1)
            y_scale_magnitude = np.random.uniform(0.9, 1.1)
            z_scale_magnitude = np.random.uniform(0.9, 1.1)
            # obtain obj original scales
            original_scale = object_scales[obj.name]
            new_scale = [original_scale[0] * x_scale_magnitude, original_scale[1] * y_scale_magnitude, original_scale[2] * z_scale_magnitude]
            obj.scale = th.tensor(new_scale)

        # # Randomize stand scale
        # y_scale_magnitude = np.random.uniform(0.9, 1.0)
        # stand_scale = stand.scale.tolist()
        # new_scale = [stand_scale[0], stand_scale[1] * y_scale_magnitude, stand_scale[2]]
        # stand.scale = th.tensor(new_scale)

        og.sim.play()
        og.sim.load_state(temp_state)

        for _ in range(50): og.sim.step()

        # Make sure all objects are upright
        all_upright = True
        for obj in objects:
            upright = check_object_upright(obj)
            print("object, upright: ", obj.name, upright)
            if not upright:
                print(f"Object {obj.name} is not upright, randomizing again")
                all_upright = False
                break
        if all_upright:
            print("All objects are upright, breaking")
            break
        trial_number += 1

    for _ in range(10): og.sim.step()

def task_completion_check(env):
    box_of_crackers = env.scene.object_registry("name", "box_of_crackers")
    stand = env.scene.object_registry("name", "stand")
    box_inside_stand = box_of_crackers.states[object_states.Inside].get_value(other=stand)
    robot = env.robots[0]
    if (
        box_inside_stand
        and robot.is_grasping(candidate_obj=box_of_crackers).value == IsGraspingState.FALSE
        and gripper_far_from_object(robot, stand)
    ):
        return True
    return False
