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
except ImportError:
    th = None
    og = None
    Environment = object
    DataPlaybackWrapper = object
    DataCollectionWrapper = object
    REGISTERED_OBJECTS = {}
    REGISTERED_ROBOTS = {}

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
from damagesim.omnigibson.params import PARAMS

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


def _load_og_damage_config() -> dict:
    config_path = Path(__file__).parent / "params" / "damageable_objects.yaml"
    if not config_path.exists():
        return {}
    try:
        with config_path.open("r") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


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

    cls_kwargs = {}
    for k in inspect.signature(base_cls.__init__).parameters:
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

    def __init__(self, configs, in_vec_env: bool = False,
                 debug_physics_frequency: bool = False,
                 reward_fn=None, **kwargs):
        # Load OG-specific damage config
        dtoc = kwargs.pop(
            "damage_trackable_objects_config",
            _load_og_damage_config(),
        )
        # Initialise the damage layer first so attributes exist before
        # Environment.__init__ (which may call reset).
        DamageableEnvironment.__init__(
            self, damage_trackable_objects_config=dtoc, **kwargs,
        )

        self._debug_physics_frequency = debug_physics_frequency
        self._reward_fn = reward_fn

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

        self.initialize_damageable_objects()
        self.set_damageable_object_params()

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
        return list(self.scene.objects)

    # ── Params ──────────────────────────────────────────────────────────

    def set_damageable_object_params(self):
        for obj in self.scene.objects:
            if not (hasattr(obj, "track_damage") and obj.track_damage):
                continue
            cat = getattr(obj, "category", "default")
            if cat in PARAMS:
                obj.set_params(PARAMS[cat])
                # Per-class damageable links
                links_key = PARAMS[cat].get("damageable_links")
                cls_links_key = PARAMS[cat].get(
                    f"{obj.__class__.__name__.lower()}_damageable_links"
                )
                if cat == "agent" and cls_links_key is not None:
                    obj.set_damageable_links(cls_links_key)
                elif links_key is not None:
                    obj.set_damageable_links(links_key)
            else:
                obj.set_params(PARAMS["default"])

    # ── Reset ───────────────────────────────────────────────────────────

    def reset(self):
        obs, info = Environment.reset(self)
        self._reset_damage_tracking()

        obs = self._process_obs(obs)

        obj_damage_info = {}
        for obj in self.scene.objects:
            if hasattr(obj, "track_damage") and obj.track_damage:
                obj_damage_info[obj.name] = obj.damage_info
        info["damage_info"] = obj_damage_info

        self.health_list_part_names = self._build_health_list()

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
        if self._health_fig is not None:
            try:
                import matplotlib.pyplot as plt
                plt.close(self._health_fig)
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
        for idx, name in enumerate(self.health_list_part_names):
            if idx < len(health_arr):
                link_healths[name] = health_arr[idx]
        current = {}
        for obj_name in self._health_tracked_object_names:
            vals = [v for k, v in link_healths.items() if k.startswith(f"{obj_name}@")]
            current[obj_name] = min(vals) if vals else 100.0
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
    def health_list_part_names(self):
        return getattr(self.env, "health_list_part_names", None)

    def process_traj_to_hdf5(self, traj_data, traj_grp_name,
                              nested_keys=("obs",), data_grp=None):
        for step in traj_data:
            state = step["state"]
            padded = th.zeros(self.max_state_size, dtype=th.float32)
            padded[: len(state)] = state
            step["state"] = padded

        health_list = []
        for obj in self.scene.objects:
            if hasattr(obj, "track_damage") and obj.track_damage:
                for ln in obj.part_healths:
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

