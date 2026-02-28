"""
Robosuite / RoboCasa DamageableEnvironment.

Designed to be mixed in with a Robosuite environment class.
Handles damage tracking, segmentation observations, and health visualization.
"""

from __future__ import annotations

import inspect
import re
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from damagesim.core.damageable_env import DamageableEnvironment
from damagesim.core.damageable_mixin import DamageableMixin
from damagesim.robosuite.damageable_mixin import (
    RSDamageableMixin,
    DamageableRobotMixin,
    DamageableFixtureMixin,
    get_damageable_robot_class,
    create_damageable_from_fixture,
    DAMAGEABLE_OBJECT_MAPPING,
    DamageableMJCFObject,
    DamageableBox,
    DamageableBall,
    DamageableCylinder,
    DamageableCapsule,
)
from damagesim.robosuite.params import (
    DAMAGEABLE_OBJECTS,
    OBJECT_PARAMS,
    get_params_for_object,
)


# ═══════════════════════════════════════════════════════════════════════
# Segmentation helpers (ported from original repo)
# ═══════════════════════════════════════════════════════════════════════

def normalize_class_name(class_name: str) -> str:
    robot_keywords = ["robot", "gripper", "panda", "finger", "hand", "arm", "link", "mount"]
    class_lower = class_name.lower()
    for kw in robot_keywords:
        if kw in class_lower:
            return "robot"
    return class_name


def build_segmentation_mapping(env, seg_type: str = "class"):
    geom_ids_to_classes = dict(getattr(env.model, "_geom_ids_to_classes", {}))
    for obj_name in getattr(env, "obj_body_id", {}).keys():
        for gid in range(env.sim.model.ngeom):
            gn = env.sim.model.geom_id2name(gid)
            if gn and obj_name.lower() in gn.lower():
                if gid not in geom_ids_to_classes:
                    geom_ids_to_classes[gid] = obj_name

    if seg_type == "element":
        return None, None

    if seg_type == "class":
        class_to_id: Dict[str, int] = {}
        mapping: Dict[int, int] = {}
        next_id = 1
        for gid, cname in geom_ids_to_classes.items():
            nc = normalize_class_name(cname)
            if nc not in class_to_id:
                class_to_id[nc] = next_id
                next_id += 1
            mapping[gid] = class_to_id[nc]
        id_to_class = {0: "background"}
        id_to_class.update({v: k for k, v in class_to_id.items()})
        return mapping, id_to_class

    if seg_type == "instance":
        inst_to_id: Dict[tuple, int] = {}
        mapping = {}
        next_id = 1
        for gid, cname in geom_ids_to_classes.items():
            key = (cname, gid)
            if key not in inst_to_id:
                inst_to_id[key] = next_id
                next_id += 1
            mapping[gid] = inst_to_id[key]
        id_to_class = {0: "background"}
        for (cname, _), sid in inst_to_id.items():
            id_to_class[sid] = cname
        return mapping, id_to_class

    return None, None


def build_segmentation_lut(mapping, max_id: int = 10000):
    if mapping is None:
        return None
    lut = np.zeros(max_id, dtype=np.int32)
    for gid, sid in mapping.items():
        if gid < max_id:
            lut[gid] = sid
    return lut


def apply_segmentation_mapping(seg_frame, lut):
    raw_ids = seg_frame[:, :, 1].astype(np.int32)
    if lut is None:
        return raw_ids
    raw_ids = np.clip(raw_ids, 0, len(lut) - 1)
    return lut[raw_ids]


def convert_obs_to_float32(obs):
    if isinstance(obs, dict):
        for k, v in obs.items():
            obs[k] = convert_obs_to_float32(v)
    elif isinstance(obs, np.ndarray) and obs.dtype == np.float64:
        obs = obs.astype(np.float32)
    elif isinstance(obs, list):
        for i, v in enumerate(obs):
            obs[i] = convert_obs_to_float32(v)
    return obs


def create_damageable_object_from_config(cls_name, cls_registry, cfg):
    if cls_name not in cls_registry:
        raise ValueError(f"Invalid object type '{cls_name}'")
    base_cls = cls_registry[cls_name]
    damageable_cls = DAMAGEABLE_OBJECT_MAPPING.get(base_cls.__name__)
    cls_to_use = damageable_cls if damageable_cls is not None else base_cls
    cls_kwargs = {}
    for k in inspect.signature(base_cls.__init__).parameters:
        if k != "self" and k in cfg:
            cls_kwargs[k] = cfg[k]
    if damageable_cls is not None:
        if "params" not in cfg:
            obj_name = cfg.get("name", "default")
            obj_type = cfg.get("object_type", None)
            cls_kwargs["params"] = get_params_for_object(obj_name, obj_type)
        else:
            cls_kwargs["params"] = cfg["params"]
    return cls_to_use(**cls_kwargs)


def create_damageable_from_object(obj):
    cn = type(obj).__name__
    if cn not in DAMAGEABLE_OBJECT_MAPPING:
        return None
    dcls = DAMAGEABLE_OBJECT_MAPPING[cn]
    cfg: dict = {"name": obj.name}
    if cn == "MJCFObject":
        fname = getattr(obj, "fname", None)
        if fname is None:
            return None
        cfg["fname"] = fname
    cfg["params"] = get_params_for_object(obj.name, "default")
    try:
        return dcls(**cfg)
    except Exception as e:
        print(f"Warning: failed to create damageable {obj.name}: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════
# Main environment
# ═══════════════════════════════════════════════════════════════════════

class RSDamageableEnvironment(DamageableEnvironment):
    """
    Robosuite / RoboCasa environment with integrated damage tracking.

    Usage::

        from robocasa.environments.kitchen import KitchenEnv
        class MyEnv(RSDamageableEnvironment, KitchenEnv):
            pass

    Or as a standalone wrapper (see helper ``make_damageable_env``).
    """

    def __init__(
        self,
        damage_config: dict | None = None,
        render_segmentation: bool = True,
        task_name: str | None = None,
        low_dim: bool = False,
        *args,
        **kwargs,
    ):
        dtoc = kwargs.pop("damage_trackable_objects_config", DAMAGEABLE_OBJECTS)
        DamageableEnvironment.__init__(
            self, damage_trackable_objects_config=dtoc,
        )

        self.task_name = task_name
        self.render_segmentation = render_segmentation
        self.low_dim = low_dim
        self.use_external_camera = True
        self.damageable_robots: list = []

        # Call the Robosuite env __init__ (next in MRO)
        super(DamageableEnvironment, self).__init__(*args, **kwargs)

        self._setup_damageable_objects()
        if self._damage_config.get("auto_replace_objects", True):
            self._replace_objects_with_damageable()

    # ── Robot loading ───────────────────────────────────────────────────

    def _load_robots(self):
        for idx, (name, config) in enumerate(zip(self.robot_names, self.robot_configs)):
            dcls = get_damageable_robot_class(name)
            self.robots[idx] = dcls(robot_type=name, idn=idx, **config)
            self.robots[idx].load_model()

    # ── Object discovery ────────────────────────────────────────────────

    def _get_all_objects(self) -> list:
        all_objs: list = []
        if hasattr(self, "model") and hasattr(self.model, "mujoco_objects"):
            all_objs.extend(self.model.mujoco_objects)
        if hasattr(self, "objects") and isinstance(self.objects, dict):
            all_objs.extend(self.objects.values())
        if hasattr(self, "robots") and isinstance(self.robots, list):
            all_objs.extend(self.robots)
        if hasattr(self, "fixtures") and isinstance(self.fixtures, dict):
            all_objs.extend(self.fixtures.values())
        return all_objs

    # ── Setup helpers ───────────────────────────────────────────────────

    def _setup_damageable_objects(self):
        control_freq = getattr(self, "control_freq", None)
        for obj in self._get_all_objects():
            if isinstance(obj, DamageableMixin):
                obj.sim = self.sim
                if control_freq is not None:
                    obj.control_freq = control_freq

    def _replace_objects_with_damageable(self):
        control_freq = getattr(self, "control_freq", None)
        if hasattr(self, "model") and hasattr(self.model, "mujoco_objects"):
            for i, obj in enumerate(self.model.mujoco_objects):
                if isinstance(obj, DamageableMixin):
                    continue
                dobj = create_damageable_from_object(obj)
                if dobj is not None:
                    dobj.sim = self.sim
                    if control_freq:
                        dobj.control_freq = control_freq
                    self.model.mujoco_objects[i] = dobj
        if hasattr(self, "objects") and isinstance(self.objects, dict):
            for key, obj in list(self.objects.items()):
                if isinstance(obj, DamageableMixin):
                    continue
                dobj = create_damageable_from_object(obj)
                if dobj is not None:
                    dobj.sim = self.sim
                    if control_freq:
                        dobj.control_freq = control_freq
                    self.objects[key] = dobj
        if hasattr(self, "fixtures") and isinstance(self.fixtures, dict):
            for key, fixture in list(self.fixtures.items()):
                if isinstance(fixture, DamageableMixin):
                    continue
                df = create_damageable_from_fixture(fixture)
                if df is not None:
                    df.sim = self.sim
                    if control_freq:
                        df.control_freq = control_freq
                    self.fixtures[key] = df

    # ── Evaluator initialisation ────────────────────────────────────────

    def _initialize_damage_evaluators_on_first_step(self):
        control_freq = getattr(self, "control_freq", None)
        for obj in self._get_all_objects():
            if isinstance(obj, DamageableMixin):
                if obj.sim is None:
                    obj.sim = self.sim
                if control_freq is not None:
                    obj.control_freq = control_freq
                obj._initialize_health()
                obj._initialize_damage_evaluators()

    def initialize_damageable_objects(self):
        control_freq = getattr(self, "control_freq", None)
        dtnames: set = set()
        dtcats: set = set()
        has_restrictions = False

        default_cfg = self.damage_trackable_objects_config.get("default", {})
        cats = default_cfg.get("categories", []) or []
        names = default_cfg.get("names", []) or []
        dtcats.update(cats)
        dtnames.update(names)
        if cats or names:
            has_restrictions = True

        if self.task_name is not None:
            tcfg = self.damage_trackable_objects_config.get(self.task_name) or {}
            if not tcfg:
                snake = re.sub(r"(?<!^)(?=[A-Z])", "_", self.task_name).lower()
                tcfg = self.damage_trackable_objects_config.get(snake) or {}
            tc = tcfg.get("categories", []) or []
            tn = tcfg.get("names", []) or []
            dtcats.update(tc)
            dtnames.update(tn)
            if tc or tn:
                has_restrictions = True

        for obj in self._get_all_objects():
            if not isinstance(obj, DamageableMixin):
                continue
            if obj.sim is None and hasattr(self, "sim"):
                obj.sim = self.sim
            if control_freq is not None:
                obj.control_freq = control_freq

            obj_name = getattr(obj, "name", None)
            obj_category = getattr(obj, "category", None)

            should_track = False
            if not has_restrictions:
                should_track = True
            else:
                if obj_category is None:
                    cn = type(obj).__name__.lower()
                    for c in dtcats:
                        if c in cn:
                            obj_category = c
                            break
                name_matches = False
                if obj_name:
                    on = obj_name.lower()
                    for c in dtcats:
                        if c and (c.lower() == on or c.lower() in on or on in c.lower()):
                            name_matches = True
                            break
                should_track = (
                    obj_category in dtcats
                    or obj_name in dtnames
                    or name_matches
                )
                if "agent" in dtcats:
                    is_robot = (
                        (hasattr(self, "robots") and obj in self.robots)
                        or hasattr(obj, "robot_type")
                        or (obj_name and "robot" in obj_name.lower())
                    )
                    if is_robot:
                        should_track = True

            if should_track:
                if hasattr(self, "sim"):
                    obj.sim = self.sim
                obj.set_track_damage(True)
                obj._initialize_damage_evaluators()
                self.damage_evaluators_initialized = True
                db = None
                if obj_name and obj_name in OBJECT_PARAMS and "damageable_bodies" in OBJECT_PARAMS[obj_name]:
                    db = OBJECT_PARAMS[obj_name]["damageable_bodies"]
                obj.set_damageable_bodies(db)
                obj._initialize_health()
            else:
                obj.set_track_damage(False)

    # ── Observation processing ──────────────────────────────────────────

    def _process_obs(self, obs):
        obs["health"] = []
        for obj in self._get_all_objects():
            if hasattr(obj, "track_damage") and obj.track_damage:
                for bn, h in obj.part_healths.items():
                    obs["health"].append(h)
        obs["health"] = np.array(obs["health"], dtype=np.float32)

        obs["object_health_states"] = {}
        obs["robot_health_states"] = {}
        for obj in self._get_all_objects():
            if isinstance(obj, DamageableMixin) and obj.track_damage:
                hs = obj.get_health_state()
                is_robot = (
                    (hasattr(self, "robots") and obj in self.robots)
                    or hasattr(obj, "robot_type")
                    or (hasattr(obj, "name") and obj.name and "robot" in obj.name.lower())
                )
                if is_robot:
                    obs["robot_health_states"][obj.name] = hs
                else:
                    obs["object_health_states"][obj.name] = hs

        if self.render_segmentation:
            seg = self.obtain_segmentation_observations()
            obs.update(seg)

        if getattr(self, "use_camera_obs", False):
            for cn in self.camera_names:
                if f"{cn}_image" in obs:
                    obs[f"{cn}_image"] = obs[f"{cn}_image"][::-1]
                if f"{cn}_depth" in obs:
                    obs[f"{cn}_depth"] = obs[f"{cn}_depth"][::-1]

        if self.use_external_camera:
            self.initialize_viewer()
            obs["external_camera_image"] = self.sim.render(width=1280, height=1280)[::-1]

        # Build combined proprio
        proprio = []
        for key, _ in [
            ("robot0_joint_pos_sin", 7),
            ("robot0_joint_pos_cos", 7),
            ("robot0_eef_pos", 3),
            ("robot0_eef_quat", 4),
            ("robot0_gripper_qpos", 2),
        ]:
            if key in obs:
                proprio.append(obs[key])
        if proprio:
            obs["robot0_proprio"] = np.concatenate(proprio, axis=0).astype(np.float32)

        if "robot0_base_to_eef_pos" in obs:
            obs["eef_pos"] = obs["robot0_base_to_eef_pos"].astype(np.float32)
        if "robot0_base_to_eef_quat" in obs:
            obs["eef_quat"] = obs["robot0_base_to_eef_quat"].astype(np.float32)

        obs_info: dict = {}
        if self.render_segmentation:
            rt = "robot0"
            obs_info[rt] = {}
            for cn in self.camera_names:
                obs_info[rt][cn] = {"seg_instance": self.seg_id_to_class}

        convert_obs_to_float32(obs)
        return obs, obs_info

    # ── Segmentation ────────────────────────────────────────────────────

    def setup_segmentation_observations(self, segmentation_type: str = "class"):
        seg_mapping, self.seg_id_to_class = build_segmentation_mapping(self, segmentation_type)
        self.seg_lut = build_segmentation_lut(seg_mapping)

    def obtain_segmentation_observations(self, segmentation_type: str = "class"):
        seg_images: dict = {}
        camera_names = list(self.camera_names)
        if self.use_external_camera:
            camera_names.append("external_camera")
        for idx, cn in enumerate(camera_names):
            if cn == "external_camera":
                self.initialize_viewer()
                seg_frame = self.sim.render(width=1280, height=1280, segmentation=True)
            else:
                seg_frame = self.sim.render(
                    width=self.camera_widths[idx],
                    height=self.camera_heights[idx],
                    camera_name=cn,
                    segmentation=True,
                )
            seg_frame = seg_frame[::-1]
            mapped = apply_segmentation_mapping(seg_frame, self.seg_lut)
            seg_images[f"{cn}_segmentation_{segmentation_type}"] = mapped.astype(np.uint8)
        return seg_images

    def initialize_viewer(self):
        try:
            import mujoco
        except ImportError:
            return
        if not hasattr(self, "viewer") or self.viewer is None:
            return
        if not hasattr(self.viewer, "viewer") or self.viewer.viewer is None:
            try:
                if hasattr(self.viewer, "update"):
                    self.viewer.update()
                elif hasattr(self, "render"):
                    self.render()
            except Exception:
                return
        if not hasattr(self.viewer, "viewer") or self.viewer.viewer is None:
            return
        try:
            if hasattr(self.viewer, "update"):
                self.viewer.update()
            elif hasattr(self.viewer.viewer, "sync"):
                self.viewer.viewer.sync()
        except Exception:
            pass
        vcam = self.viewer.viewer.cam
        rc = self.sim._render_context_offscreen
        rcam = mujoco.MjvCamera()
        rcam.type = vcam.type
        rcam.fixedcamid = vcam.fixedcamid if vcam.type == 2 else -1
        rcam.lookat = vcam.lookat.copy()
        rcam.distance = vcam.distance
        rcam.azimuth = vcam.azimuth
        rcam.elevation = vcam.elevation
        rc.cam = rcam

    # ── Step / Reset ────────────────────────────────────────────────────

    def reset(self):
        obs = super(DamageableEnvironment, self).reset()
        if self._damage_config.get("auto_replace_objects", True):
            self._replace_objects_with_damageable()
        self.initialize_damageable_objects()

        obj_damage_info: dict = {}
        for obj in self._get_all_objects():
            if isinstance(obj, DamageableMixin) and obj.track_damage:
                obj.reset_damage_evaluators()
                obj_damage_info[obj.name] = obj.damage_info
        if self.render_segmentation:
            self.setup_segmentation_observations()
        obs, obs_info = self._process_obs(obs)
        info = {"damage_info": obj_damage_info, "obs_info": obs_info}

        health_list = []
        for obj in self._get_all_objects():
            if isinstance(obj, DamageableMixin) and obj.track_damage:
                for bn in obj.part_healths:
                    health_list.append(f"{obj.name}@{bn}")
        self.health_list_link_names = health_list

        return obs, info

    def step(self, action):
        if not self.damage_evaluators_initialized:
            self._initialize_damage_evaluators_on_first_step()
            self.damage_evaluators_initialized = True

        obs, reward, done, info = super(DamageableEnvironment, self).step(action)

        obj_damage_info: dict = {}
        if not self.lock_health:
            for obj in self._get_all_objects():
                if isinstance(obj, DamageableMixin) and obj.track_damage:
                    obj.update_health()
                    obj_damage_info[obj.name] = obj.damage_info
        info["damage_info"] = obj_damage_info
        obs, obs_info = self._process_obs(obs)
        info["obs_info"] = obs_info
        return obs, reward, done, info

    # ── Misc ────────────────────────────────────────────────────────────

    def get_observations(self):
        obs = super(DamageableEnvironment, self)._get_observations()
        obj_damage_info: dict = {}
        if not self.lock_health:
            for obj in self._get_all_objects():
                if isinstance(obj, DamageableMixin) and obj.track_damage:
                    obj_damage_info[obj.name] = obj.damage_info
        obs, obs_info = self._process_obs(obs)
        return obs, {"damage_info": obj_damage_info, "obs_info": obs_info}

    def get_all_damageable_objects(self) -> list:
        return [
            obj for obj in self._get_all_objects()
            if isinstance(obj, DamageableMixin) and obj.track_damage
        ]

    def initialize_env_health(self):
        for obj in self._get_all_objects():
            if isinstance(obj, DamageableMixin) and obj.track_damage:
                obj._initialize_health()

    def lock_health_changes(self):
        self.lock_health = True

    def unlock_health_changes(self):
        self.lock_health = False

