"""
OmniGibson-specific DamageableEnvironment.

Wraps ``omnigibson.envs.env_base.Environment`` and adds damage tracking
for every object and robot in the scene.
"""

from __future__ import annotations

import inspect
import json
import random
import string
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

try:
    import torch as th
    import omnigibson as og
    from omnigibson.envs.env_base import Environment
    from omnigibson.envs.data_wrapper import DataPlaybackWrapper, DataCollectionWrapper
    from omnigibson.objects import REGISTERED_OBJECTS
    from omnigibson.robots import REGISTERED_ROBOTS
    from omnigibson.scenes import REGISTERED_SCENES
    from omnigibson.utils.python_utils import create_class_from_registry_and_config
except ImportError:
    th = None
    og = None
    Environment = object
    DataPlaybackWrapper = object
    DataCollectionWrapper = object
    REGISTERED_OBJECTS = {}
    REGISTERED_ROBOTS = {}
    REGISTERED_SCENES = {}
    create_class_from_registry_and_config = None

from damagesim.core.damageable_env import DamageableEnvironment
from damagesim.omnigibson.damageable_mixin import (
    OGDamageableMixin,
    DamageableDatasetObject,
    DamageablePrimitiveObject,
    DamageableUSDObject,
    DamageableControllableObject,
    DamageableLightObject,
    DamageableStatefulObject,
    DamageableFrankaPanda,
    DamageableFrankaMounted,
    DamageableTiago,
    DamageableR1Pro,
)
from damagesim.omnigibson.params import PARAMS, DAMAGEABLE_OBJECTS

# Flag to ensure we only patch the OG object registry once per process
_BEHAVIOR_DAMAGEABLE_PATCHED = False

# Mapping from base OG class names to their damageable counterparts
DAMAGEABLE_OBJECT_MAPPING = {
    "DatasetObject": DamageableDatasetObject,
    "PrimitiveObject": DamageablePrimitiveObject,
    "USDObject": DamageableUSDObject,
    "ControllableObject": DamageableControllableObject,
    "LightObject": DamageableLightObject,
    "StatefulObject": DamageableStatefulObject,
    "FrankaPanda": DamageableFrankaPanda,
    "FrankaMounted": DamageableFrankaMounted,
    "Tiago": DamageableTiago,
    "R1Pro": DamageableR1Pro,
}


def create_damageable_object_from_config(cls_name, cls_registry, cfg, cls_type_descriptor="object"):
    """
    Factory: instantiate an OG object from *cfg*, using its damageable variant
    when available.
    """
    assert cls_name in cls_registry, (
        f"Invalid {cls_type_descriptor} type received! "
        f"Valid: {list(cls_registry.keys())}, got: {cls_name}"
    )
    base_cls = cls_registry[cls_name]
    damageable_cls = DAMAGEABLE_OBJECT_MAPPING.get(base_cls.__name__)
    cls_to_use = damageable_cls if damageable_cls is not None else base_cls

    # Use OG base class for signature so required args (name, category, etc.) are forwarded
    sig_cls = base_cls.__bases__[1] if (
        len(base_cls.__bases__) >= 2 and base_cls.__name__.startswith("Damageable")
    ) else base_cls
    cls_kwargs = {}
    for k in inspect.signature(sig_cls.__init__).parameters:
        if k != "self" and k in cfg:
            cls_kwargs[k] = cfg[k]

    if damageable_cls is not None:
        if "damage_params" not in cfg:
            cat = cfg.get("category", "default")
            cls_kwargs["params"] = PARAMS.get(cat, PARAMS["default"])
        else:
            cls_kwargs["params"] = cfg["damage_params"]

    return cls_to_use(**cls_kwargs)


class OGDamageableEnvironment(DamageableEnvironment, Environment):
    """
    OmniGibson environment with integrated damage tracking.
    """

    def __init__(self, configs,
                 in_vec_env: bool = False,
                 reward_fn=None, **kwargs):
        # Load OG-specific damage config
        dtoc = kwargs.pop(
            "damage_trackable_objects_config",
            DAMAGEABLE_OBJECTS,
        )
        # Initialise the damage layer first so attributes exist before
        # Environment.__init__ (which may call reset).
        DamageableEnvironment.__init__(
            self, damage_trackable_objects_config=dtoc, **kwargs,
        )

        self._reward_fn = reward_fn
        self.task_name = configs["task"].get("activity_name", None) if "task" in configs else None

        # Now initialise the OG Environment (may trigger reset / load)
        Environment.__init__(self, configs, in_vec_env)

    # ── OG load hooks ──────────────────────────────────────────────────

    def load(self):
        self._loaded = False
        self._load_variables()

        og.sim.stop()
        self._load_scene()
        self._load_objects()
        self._load_robots()
        self._load_task()
        self._load_external_sensors()
        og.sim.play()

        # Sort objects
        self.objects = self.scene.objects.copy()
        self.objects.sort(key=lambda x: x.name)

        self.initialize_damageable_objects()

    def _load_scene(self):
        """Load the scene so that scene-file objects are created as damageable classes."""
        assert og.sim.is_stopped(), "Simulator must be stopped before loading scene!"

        global _BEHAVIOR_DAMAGEABLE_PATCHED
        if not _BEHAVIOR_DAMAGEABLE_PATCHED:
            for cname, dcls in {
                "DatasetObject": DamageableDatasetObject,
                "PrimitiveObject": DamageablePrimitiveObject,
                "USDObject": DamageableUSDObject,
                "ControllableObject": DamageableControllableObject,
                "LightObject": DamageableLightObject,
                "StatefulObject": DamageableStatefulObject,
            }.items():
                if cname in REGISTERED_OBJECTS:
                    REGISTERED_OBJECTS[cname] = dcls
            _BEHAVIOR_DAMAGEABLE_PATCHED = True

        self._scene = create_class_from_registry_and_config(
            cls_name=self.scene_config["type"],
            cls_registry=REGISTERED_SCENES,
            cfg=self.scene_config,
            cls_type_descriptor="scene",
        )
        og.sim.import_scene(self._scene)
        assert og.sim.is_stopped(), "Simulator must be stopped after loading scene!"

    def _load_robots(self):
        if len(self.scene.robots) == 0:
            assert og.sim.is_stopped()
            for robot_config in self.robots_config:
                if "name" not in robot_config:
                    robot_config["name"] = "robot_" + "".join(
                        random.choices(string.ascii_lowercase, k=6)
                    )
                position = robot_config.pop("position", None)
                orientation = robot_config.pop("orientation", None)
                pose_frame = robot_config.pop("pose_frame", "scene")
                if position is not None:
                    position = th.as_tensor(position, dtype=th.float32)
                if orientation is not None:
                    orientation = th.as_tensor(orientation, dtype=th.float32)

                robot = create_damageable_object_from_config(
                    cls_name=robot_config["type"],
                    cls_registry=REGISTERED_ROBOTS,
                    cfg=robot_config,
                    cls_type_descriptor="robot",
                )
                self.scene.add_object(robot)
                robot.set_position_orientation(
                    position=position, orientation=orientation, frame=pose_frame,
                )
        assert og.sim.is_stopped()

    def _load_objects(self):
        assert og.sim.is_stopped()
        for i, obj_config in enumerate(self.objects_config):
            if "name" not in obj_config:
                obj_config["name"] = f"obj{i}"
            position = obj_config.pop("position", None)
            orientation = obj_config.pop("orientation", None)
            obj = create_damageable_object_from_config(
                cls_name=obj_config["type"],
                cls_registry=REGISTERED_OBJECTS,
                cfg=obj_config,
                cls_type_descriptor="object",
            )
            self.scene.add_object(obj)
            obj.set_position_orientation(
                position=position, orientation=orientation, frame="scene",
            )
        assert og.sim.is_stopped()

    # ── Object discovery ────────────────────────────────────────────────

    def _get_all_objects(self) -> list:
        return self.objects


    # ── Reset ───────────────────────────────────────────────────────────

    def reset(self):
        obs, info = Environment.reset(self)
        self._reset_damage_tracking()

        obs = self._process_obs(obs)

        obj_damage_info = {}
        for obj in self._get_all_objects():
            if hasattr(obj, "track_damage") and obj.track_damage:
                obj_damage_info[obj.name] = obj.damage_info
        info["damage_info"] = obj_damage_info

        if self._health_visualization_enabled:
            self.update_health_visualization(obs)

        return obs, info

    # ── Step ────────────────────────────────────────────────────────────

    def step(self, action, n_render_iterations=1, episode_step_count=0,
             playback=False, init_skip_steps=0):
        if not self.damage_evaluators_initialized:
            self._initialize_all_evaluators()

        obs, reward, terminated, truncated, info = Environment.step(
            self, action, n_render_iterations,
        )

        should_update = episode_step_count > init_skip_steps - 1
        if should_update:
            damage_info = self._update_all_health()
            info["damage_info"] = damage_info
            if self._reward_fn is not None:
                reward, terminated = self._reward_fn(self, obs)

        obs = self._process_obs(obs)

        if self._health_visualization_enabled:
            active = self.update_health_visualization(obs)
            if not active:
                self._health_visualization_enabled = False

        return obs, reward, terminated, truncated, info

    # ── Observation processing ──────────────────────────────────────────

    def _process_obs(self, obs):
        self._append_health_to_obs(obs)

        # Convert health to torch tensor for OG consistency
        if th is not None and isinstance(obs.get("health"), np.ndarray):
            obs["health"] = th.tensor(obs["health"], dtype=th.float32)

        # EEF pose
        robot = self.robots[0]
        default_arm = getattr(robot, "default_arm", "right")
        eef_pose = robot.get_relative_eef_pose(default_arm)
        obs["eef_pos"] = eef_pose[0]
        obs["eef_ori"] = eef_pose[1]
        return obs

    def get_observation(self):
        obs, info = super().get_obs()
        obs = self._process_obs(obs)
        return obs, info

    # ── Health visualisation ────────────────────────────────────────────

    def enable_health_visualization(self):
        from damagesim.utils.visualization import setup_live_health_bars
        objs = self.get_damageable_objects()
        if not objs:
            print("Warning: no damageable objects found.")
            return False
        names = [o.name for o in objs]
        if self._health_visualization_enabled:
            self.disable_health_visualization()
        try:
            self._health_fig, self._health_ax, self._health_bars_dict = setup_live_health_bars(names)
            self._health_tracked_object_names = names
            self._health_visualization_enabled = True
            return True
        except Exception as e:
            print(f"Error enabling health viz: {e}")
            return False

    def disable_health_visualization(self):
        # Clear object highlights
        if self._health_tracked_object_names is not None:
            for obj_name in self._health_tracked_object_names:
                try:
                    obj = self.scene.object_registry("name", obj_name)
                    if obj is not None:
                        obj.highlighted = False
                except Exception:
                    pass
        if self._health_fig is not None:
            try:
                import matplotlib.pyplot as plt
                if plt.fignum_exists(self._health_fig.number):
                    plt.close(self._health_fig)
            except Exception:
                pass
            try:
                import matplotlib.pyplot as plt
                plt.ioff()
            except Exception:
                pass
        self._health_visualization_enabled = False
        self._health_fig = self._health_ax = self._health_bars_dict = None
        self._health_tracked_object_names = None

    def update_health_visualization(self, obs=None):
        if not self._health_visualization_enabled:
            return True
        from damagesim.utils.visualization import update_live_health_bars
        if obs is None:
            obs, _ = self.get_observation()
        health_arr = obs.get("health")
        if health_arr is None:
            return True
        if hasattr(health_arr, "cpu"):
            health_arr = health_arr.cpu().numpy()
        else:
            health_arr = np.asarray(health_arr)
        link_healths = {}
        for idx, name in enumerate(self.health_list_link_names):
            if idx < len(health_arr):
                link_healths[name] = health_arr[idx]
        current = {}
        for obj_name in self._health_tracked_object_names:
            vals = [v for k, v in link_healths.items() if k.startswith(f"{obj_name}@")]
            current[obj_name] = min(vals) if vals else 100.0

        # Live object coloring: tint objects red proportionally to damage
        for obj_name in self._health_tracked_object_names:
            try:
                obj = self.scene.object_registry("name", obj_name)
                if obj is not None:
                    intensity = 10000 * (100.0 - current[obj_name]) / 100.0
                    obj.set_highlight_properties(color=[255.0, 0.0, 0.0], intensity=intensity)
                    obj.highlighted = True
            except Exception:
                pass

        try:
            return update_live_health_bars(
                self._health_fig, self._health_ax, self._health_bars_dict,
                current, self._health_tracked_object_names,
            )
        except Exception:
            return False


# ═══════════════════════════════════════════════════════════════════════
# Data collection / playback wrappers
# ═══════════════════════════════════════════════════════════════════════

class OGDamageableDataCollectionWrapper(DataCollectionWrapper):
    """Extends OG DataCollectionWrapper to store health metadata."""

    def _optimize_sim_for_data_collection(self, viewport_camera_path):
        """Skip the base-class optimizations (disabling sensor render products
        and overriding the viewer camera path) so that teleop and other
        interactive callers keep the correct viewport and working sensors."""
        pass

    def enable_health_visualization(self):
        if hasattr(self.env, "enable_health_visualization"):
            return self.env.enable_health_visualization()
        return False

    def disable_health_visualization(self):
        if hasattr(self.env, "disable_health_visualization"):
            self.env.disable_health_visualization()

    def update_health_visualization(self, obs=None):
        if hasattr(self.env, "update_health_visualization"):
            return self.env.update_health_visualization(obs)
        return True

    @property
    def health_list_link_names(self):
        return getattr(self.env, "health_list_link_names", None)

    def process_traj_to_hdf5(self, traj_data, traj_grp_name,
                              nested_keys=("obs",), data_grp=None):
        for step in traj_data:
            state = step["state"]
            padded = th.zeros(self.max_state_size, dtype=th.float32)
            padded[: len(state)] = state
            step["state"] = padded

        health_list = []
        for obj in self._get_all_objects():
            if hasattr(obj, "track_damage") and obj.track_damage:
                for ln in obj.link_healths:
                    health_list.append(f"{obj.name}@{ln}")

        grp = super().process_traj_to_hdf5(traj_data, traj_grp_name, nested_keys, data_grp)
        grp.attrs["health_list_link_names"] = health_list
        return grp


class OGDamageableDataPlaybackWrapper(DataPlaybackWrapper):
    """Extends OG DataPlaybackWrapper to use OGDamageableEnvironment."""

    def _load_state_with_size_fallback(self, state, saved_size):
        try:
            og.sim.load_state(state[:saved_size], serialized=True)
            return saved_size
        except AssertionError as e:
            if "Invalid state deserialization" in str(e):
                for try_size in range(saved_size - 1, max(0, saved_size - 10), -1):
                    try:
                        og.sim.load_state(state[:try_size], serialized=True)
                        return try_size
                    except AssertionError:
                        continue
            raise

    @classmethod
    def create_from_hdf5(cls, input_path, output_path, **kwargs):
        """Create wrapper using OGDamageableEnvironment."""
        import h5py, json as _json
        from omnigibson.macros import gm
        from omnigibson.utils.data_utils import merge_scene_files
        from omnigibson.envs.env_wrapper import create_wrapper

        include_contacts = kwargs.pop("include_contacts", True)
        include_task = kwargs.pop("include_task", True)
        include_task_obs = kwargs.pop("include_task_obs", True)
        include_env_wrapper = kwargs.pop("include_env_wrapper", False)
        additional_wrapper_configs = kwargs.pop("additional_wrapper_configs", None)
        overwrite_config = kwargs.pop("overwrite_config", None)
        full_scene_file = kwargs.pop("full_scene_file", None)
        load_room_instances = kwargs.pop("load_room_instances", None)
        robot_obs_modalities = kwargs.pop("robot_obs_modalities", ())
        robot_proprio_keys = kwargs.pop("robot_proprio_keys", None)
        robot_sensor_config = kwargs.pop("robot_sensor_config", None)
        external_sensors_config = kwargs.pop("external_sensors_config", None)
        include_sensor_names = kwargs.pop("include_sensor_names", None)
        exclude_sensor_names = kwargs.pop("exclude_sensor_names", None)
        include_robot_control = kwargs.pop("include_robot_control", True)
        append_to_input_path = kwargs.pop("append_to_input_path", False)
        activity_name = kwargs.pop("activity_name", None)
        
        f = h5py.File(input_path, "a" if append_to_input_path else "r")
        config = (
            _json.loads(f["data"].attrs["config"])
            if overwrite_config is None
            else _json.loads(overwrite_config)
        )

        if include_contacts:
            config["env"]["action_frequency"] = 30.0
            config["env"]["rendering_frequency"] = 30.0
            config["env"]["physics_frequency"] = 30.0
        else:
            config["env"]["action_frequency"] = 30.0
            config["env"]["rendering_frequency"] = 30.0
            config["env"]["physics_frequency"] = 120.0
            gm.VISUAL_ONLY = True

        config["env"]["flatten_obs_space"] = True
        config["scene"]["scene_file"] = _json.loads(f["data"].attrs["scene_file"])

        if full_scene_file:
            with open(full_scene_file) as fj:
                full_json = _json.load(fj)
            config["scene"]["scene_file"] = merge_scene_files(
                scene_a=full_json,
                scene_b=config["scene"]["scene_file"],
                keep_robot_from="b",
            )
            config["scene"]["load_room_types"] = None
            config["scene"]["load_room_instances"] = load_room_instances

        if not include_task:
            config["task"] = {"type": "DummyTask"}
        config["task"]["include_obs"] = include_task_obs
        config["task"]["activity_name"] = activity_name

        if config["task"]["type"] == "BehaviorTask":
            config["task"]["online_object_sampling"] = False
            config["task"]["use_presampled_robot_pose"] = False

        if load_room_instances is not None:
            config["scene"]["load_room_instances"] = load_room_instances

        config["objects"] = []

        for rcfg in config["robots"]:
            rcfg["obs_modalities"] = list(robot_obs_modalities)
            rcfg["include_sensor_names"] = include_sensor_names
            rcfg["exclude_sensor_names"] = exclude_sensor_names
            if robot_proprio_keys is not None:
                rcfg["proprio_obs"] = robot_proprio_keys
            if robot_sensor_config is not None:
                rcfg["sensor_config"] = robot_sensor_config
        if external_sensors_config is not None:
            config["env"]["external_sensors"] = external_sensors_config

        # Patch OG registry
        global _BEHAVIOR_DAMAGEABLE_PATCHED
        if not _BEHAVIOR_DAMAGEABLE_PATCHED:
            for cname, dcls in {
                "DatasetObject": DamageableDatasetObject,
                "PrimitiveObject": DamageablePrimitiveObject,
                "USDObject": DamageableUSDObject,
                "ControllableObject": DamageableControllableObject,
                "LightObject": DamageableLightObject,
                "StatefulObject": DamageableStatefulObject,
            }.items():
                if cname in REGISTERED_OBJECTS:
                    REGISTERED_OBJECTS[cname] = dcls
            _BEHAVIOR_DAMAGEABLE_PATCHED = True

        env = OGDamageableEnvironment(configs=config)
        if include_env_wrapper:
            env = create_wrapper(env=env)
        if additional_wrapper_configs:
            for wcfg in additional_wrapper_configs:
                env = create_wrapper(env=env, wrapper_cfg=wcfg)

        return cls(
            env=env,
            input_path=input_path,
            output_path=output_path,
            include_robot_control=include_robot_control,
            include_contacts=include_contacts,
            **kwargs,
        )

    def playback_dataset(
        self, record_data=False, demo_ids=None, save_images=True
    ):
        """
        Playback all episodes from the input HDF5 file, and optionally record observation data if @record is True

        Args:
            record_data (bool): Whether to record data during playback or not
            video_writers (None or list of imageio.Writer): If specified, writer object that RGB frames will be written to
            demo_ids (None or list of int): If specified, a list of episode IDs to playback. If None, all episodes will be played
            save_images (bool): If True, save images to the output hdf5 file
        """
        results = []
        if demo_ids is None:
            for episode_id in range(self.input_hdf5["data"].attrs["n_episodes"]):
                results.append(
                    self.playback_episode(
                        episode_id=episode_id,
                        record_data=record_data,
                        save_images=save_images,
                    )
                )
        else:
            for episode_id in demo_ids:
                results.append(
                    self.playback_episode(
                        episode_id=episode_id,
                        record_data=record_data,
                        save_images=save_images,
                    )
                )
        return results
    
    def playback_episode(self,
                         episode_id,
                         record_data=True,
                         save_images=True,
                         break_after_n_steps=100):
        """
        Playback episode @episode_id, and optionally record observation data if @record is True.
        
        This method overrides the parent implementation to call set_damageable_object_params() on the
        wrapped environment right after scene.restore() is called.

        Args:
            episode_id (int): Episode to playback. This should be a valid demo ID number from the inputted collected
                data hdf5 file
            record_data (bool): Whether to record data during playback or not
            save_images (bool): If True, save images to the output hdf5 file
            break_after_n_steps (int): Number of steps to break after when save_images is True
        """
        import h5py
        import json
        from omnigibson.utils.python_utils import h5py_group_to_torch, create_object_from_init_info
        import omnigibson as og
        from omnigibson.controllers.controller_base import ControlType
        from omnigibson.systems.macro_particle_system import MacroPhysicalParticleSystem
        
        data_grp = self.input_hdf5["data"]
        assert f"demo_{episode_id}" in data_grp, f"No valid episode with ID {episode_id} found!"
        traj_grp = data_grp[f"demo_{episode_id}"]

        # Skip the first @init_skip_steps steps to get the initial health values
        # We do this beacause playback has some artifacts which I havent' understood yet.
        # Due to these artifacts (object being loaded at a very different place and then teleoported suddenly
        # leading to high impact forces), the initial health values are not correct.
        self.init_skip_steps = 4 # orginally 4

        
        # Grab episode data
        # Skip early if found malformed data
        try:
            # If obs["health"] and info["damage_info"] is populated, then fetch them
            # NOTE: This is important for pour water task because the water particles during playback are not
            # acting as expected (water particles seem to fly around sometimes). But, during data collection, it is working as expected. 
            # So, we obtain the correct health and damage info from the data collection hdf5 file and use them during playback.
            datacollection_health = None
            datacollection_damage_info = None
            if "obs" in traj_grp and "info" in traj_grp:
                datacollection_health = th.from_numpy(traj_grp["obs"]["health"][()])
                damage_infos = traj_grp["info"]["damage_info"][()]
                datacollection_damage_info = []
                for i in range(len(damage_infos)): datacollection_damage_info.append(json.loads(damage_infos[i]))

            transitions = json.loads(traj_grp.attrs["transitions"])
            traj_grp = h5py_group_to_torch(traj_grp)
            init_metadata = traj_grp["init_metadata"]
            action = traj_grp["action"]
            state = traj_grp["state"]
            state_size = traj_grp["state_size"]
            reward = traj_grp["reward"]
            terminated = traj_grp["terminated"]
            truncated = traj_grp["truncated"]
        
        except KeyError as e:
            print(f"Got error when trying to load episode {episode_id}:")
            print(f"Error: {str(e)}")
            return

        result = []
        
        # Reset environment and update this to be the new initial state. NOTE: It is important to call reset() before calling scene.restore()
        self.reset()
        self.scene.restore(self.scene_file, update_initial_file=True)

        # Update objects list in case any new objects were added by the scene restore
        self.objects = self.scene.objects.copy()
        self.objects.sort(key=lambda x: x.name)
        # Initializes all damageable objects
        self.env.initialize_damageable_objects()
        # Resets damage tracking and most importantly sets the health_list_link_names array        
        self.env._reset_damage_tracking()
        
        # Reset object attributes from the stored metadata
        with og.sim.stopped():
            for attr, vals in init_metadata.items():
                assert len(vals) == self.scene.n_objects
            for i, obj in enumerate(self._get_all_objects()):
                for attr, vals in init_metadata.items():
                    val = vals[i]
                    setattr(obj, attr, val.item() if val.ndim == 0 else val)
        
        # If not controlling robots, disable for all robots
        if not self.include_robot_control:
            for robot in self.robots:
                robot.control_enabled = False
                # Set all controllers to effort mode with zero gain, this keeps the robot still
                for controller in robot.controllers.values():
                    for i, dof in enumerate(controller.dof_idx):
                        dof_joint = robot.joints[robot.dof_names_ordered[dof]]
                        dof_joint.set_control_type(
                            control_type=ControlType.EFFORT,
                            kp=None,
                            kd=None,
                        )

        print(f"================= starting playback for demo {episode_id} ===================")
        
        # Restore to initial state
        # Ensure simulator is playing before loading state (required by load_state)
        if not og.sim.is_playing():
            og.sim.play()
        
        # Try loading state with saved size, but handle size mismatches gracefully
        saved_state_size = int(state_size[0])
        self._load_state_with_size_fallback(state[0], saved_state_size)
        for _ in range(10): og.sim.step()

        # We need to step the environment to get the initial observations propagated
        first_time_load_n_iteration = 10
        self.current_obs, _, _, _, init_info = self.env.step(
            action=action[0], n_render_iterations=self.n_render_iterations + first_time_load_n_iteration, playback=True, init_skip_steps=self.init_skip_steps
        )

        print("After reset health: ", self.current_obs["health"])
        # breakpoint()
        # # Print all object names in the scene (For debugging)
        # if replay_for_annotation:
        #     print(f"================= object names in the scene =================")
        #     all_objs = og.sim.scenes[0].objects
        #     print([o.name for o in all_objs])

        # if water system exists, set it to not visible
        if "water" in self.scene.systems:
            water_system = self.scene.get_system("water")
            for prototype in water_system.particle_prototypes: prototype.visible = False
            for instancer in water_system.particle_instancers.values(): instancer.visible = False

        for i, (a, s, ss, r, te, tr) in enumerate(
            zip(action, state[1:], state_size[1:], reward, terminated, truncated)
        ):
            if i % 50 == 0:
                print(f"step {i} completed")
                robot = self.scene.robots[0]
                if self.task.__class__.__name__ == "BehaviorTask":
                    if self.task.activity_name == "attach_a_camera_to_a_tripod":
                        camera = self.scene.object_registry("name", "digital_camera_87")
                        tripod = self.scene.object_registry("name", "camera_tripod_86")
                        print(f"healths: camera {camera.health}, tripod {tripod.health}, robot {robot.health}")
                    elif self.task.activity_name == "make_microwave_popcorn":
                        microwave = self.scene.object_registry("name", "microwave_hjjxmi_0")
                        print(f"healths: microwave {microwave.health}, robot {robot.health}")
                    elif self.task.activity_name == "clean_a_trumpet":
                        scrub = self.scene.object_registry("name", "scrub_brush_86")
                        print(f"healths: scrub {scrub.health}, robot {robot.health}")
        
            # # For debugging
            # if i > 20:
            #     break

            if i == self.init_skip_steps:
                # Update link positions and velocities for all damage evaluators
                for obj in self._get_all_objects():
                    if hasattr(obj, "track_damage") and obj.track_damage:
                        for evaluator in obj.damage_evaluators:
                            if evaluator.name == "mechanical":
                                evaluator.update_link_positions_and_velocities()
            
            if i == self.init_skip_steps + 1:
                # Save the initial obs
                obs_data = self._process_obs(obs=self.current_obs, info=info)
                if not save_images:
                    obs_data_modified = dict()
                    for key in obs_data:
                        if key.endswith("rgb") or key.endswith("depth") or key.endswith("seg_instance") or key.endswith("seg_semantic"):
                            continue
                    step_data = {"obs": obs_data_modified}
                else:
                    step_data = {"obs": obs_data}

                # Overwrite the health and damage info with the datacollection values
                if datacollection_health is not None:
                    step_data["obs"]["health"] = datacollection_health[i]
                self.current_traj_history.append(step_data)
                print("After first computation of health: ", self.current_obs["health"])

            # Execute any transitions that should occur at this current step
            # print("Action", a)
            if str(i) in transitions:
                cur_transitions = transitions[str(i)]
                scene = og.sim.scenes[0]
                for add_sys_name in cur_transitions["systems"]["add"]:
                    scene.get_system(add_sys_name, force_init=True)
                for remove_sys_name in cur_transitions["systems"]["remove"]:
                    scene.clear_system(remove_sys_name)
                for remove_obj_name in cur_transitions["objects"]["remove"]:
                    obj = scene.object_registry("name", remove_obj_name)
                    scene.remove_object(obj)
                for j, add_obj_info in enumerate(cur_transitions["objects"]["add"]):
                    obj = create_object_from_init_info(add_obj_info)
                    scene.add_object(obj)
                    obj.set_position(th.ones(3) * 100.0 + th.ones(3) * 5 * j)
                # Step physics to initialize any new objects
                og.sim.step()
            
            # Restore the sim state, and take a very small step with the action to make sure physics are
            # properly propagated after the sim state update
            # Ensure simulator is playing before loading state (required by load_state)
            if not og.sim.is_playing():
                og.sim.play()
            self._load_state_with_size_fallback(s, int(ss))

            # Restore the sim state, and take a very small step with the action to make sure physics are
            # properly propagated after the sim state update
            # Ensure simulator is playing before loading state (required by load_state)
            if not og.sim.is_playing():
                og.sim.play()
            self._load_state_with_size_fallback(s, int(ss))
            if not self.include_contacts:
                # When all objects/systems are visual-only, keep them still on every step
                for obj in self._get_all_objects():
                    obj.keep_still()
                for system in self.scene.systems:
                    # TODO: Implement keep_still for other systems
                    if isinstance(system, MacroPhysicalParticleSystem):
                        system.set_particles_velocities(
                            lin_vels=th.zeros((system.n_particles, 3)), ang_vels=th.zeros((system.n_particles, 3))
                        )
            self.current_obs, _, _, _, info = self.env.step(action=a, n_render_iterations=self.n_render_iterations, episode_step_count=i, playback=True, init_skip_steps=self.init_skip_steps)
            # If recording, record data
            if record_data and i > self.init_skip_steps:
                # for link_name in info["damage_info"]["franka0"]:
                #     print("damage: ", link_name, info["damage_info"]["franka0"][link_name]["mechanical"]["damage"], info["damage_info"]["franka0"][link_name]["thermal"]["damage"])
                # print("temperature: ", info["damage_info"]["franka0"]["panda_link6"]["thermal"]["temperature"])
                step_data = self._parse_step_data(
                    action=a,
                    obs=self.current_obs,
                    reward=r,
                    terminated=te,
                    truncated=tr,
                    info=info,
                    datacollection_health=datacollection_health[i] if datacollection_health is not None else None,
                    datacollection_damage_info=datacollection_damage_info[i] if datacollection_damage_info is not None else None,
                    save_images=save_images,
                )
                if self.flush_every_n_steps > 0:
                    if i == 0:
                        self.current_traj_grp, self.traj_dsets = self.allocate_traj_to_hdf5(
                            step_data, f"demo_{episode_id}", num_samples=len(action)
                        )
                    if i % self.flush_every_n_steps == 0:
                        self.flush_partial_traj(num_samples=len(action))
                # append to current trajectory history
                self.current_traj_history.append(step_data)

            self.current_episode_step_count += 1
            self.step_count += 1

        if record_data:
            if self.flush_every_n_steps > 0:
                self.flush_partial_traj(num_samples=len(action))
            self.flush_current_traj(traj_grp_name=f"demo_{episode_id}")

        print("Final health: ", self.current_obs["health"])
        
        return result

    def _parse_step_data(self, action, obs, reward, terminated, truncated, info, datacollection_health=None, datacollection_damage_info=None, save_images=True):
        # Store action, obs, reward, terminated, truncated, info
        step_data = dict()
        obs_data = self._process_obs(obs=obs, info=info)
        if not save_images:
            obs_data_modified = dict()
            for key in obs_data:
                if key.endswith("rgb") or key.endswith("depth") or key.endswith("seg_instance") or key.endswith("seg_semantic"):
                    continue
                else:
                    obs_data_modified[key] = obs_data[key]
            step_data["obs"] = obs_data_modified
        else:
            step_data["obs"] = obs_data
        step_data["action"] = action
        step_data["reward"] = reward
        step_data["terminated"] = terminated
        step_data["truncated"] = truncated
        step_data["info"] = info

        # Overwrite the health and damage info with the datacollection values
        if datacollection_health is not None:
            step_data["obs"]["health"] = datacollection_health
        if datacollection_damage_info is not None:
            step_data["info"]["damage_info"] = datacollection_damage_info
        return step_data

    def process_traj_to_hdf5(self, traj_data, traj_grp_name, nested_keys=("obs",), data_grp=None):
        """
        Processes trajectory data and stores them in HDF5, with proper health metadata collection.
        
        This method overrides the parent implementation to:
        - Ensure health is initialized before collecting metadata
        - Include robots in health metadata
        - Add proper error handling
        
        Args:
            traj_data (list of dict): Trajectory data, where each entry is a keyword-mapped set of data for a single
                sim step
            traj_grp_name (str): Name of the trajectory group to store
            nested_keys (list of str): Name of key(s) corresponding to nested data in @traj_data
            data_grp (None or h5py.Group): If specified, the h5py Group under which a new group with name
                @traj_grp_name will be created. If None, will default to "data" group

        Returns:
            hdf5.Group: Generated hdf5 group storing the recorded trajectory data
        """
        
        # Collect health metadata with proper initialization and error handling BEFORE calling parent
        # This ensures health is initialized and we include robots
        health_list = []

        for obj in self._get_all_objects():
            if hasattr(obj, "track_damage") and obj.track_damage:
                for link_name, health in obj.link_healths.items():
                    health_list.append(f"{obj.name}@{link_name}")        

        # Call parent method to handle the rest of the data processing
        traj_grp = super().process_traj_to_hdf5(traj_data, traj_grp_name, nested_keys, data_grp)
        
        # Add health list link names to the trajectory group
        traj_grp.attrs["health_list_link_names"] = health_list

        return traj_grp


