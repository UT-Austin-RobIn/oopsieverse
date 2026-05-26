"""
Simulator-agnostic base for a damage-tracking environment wrapper.

This class encapsulates the shared lifecycle that both the OmniGibson and
Robosuite environments follow:

    1. After ``__init__``: discover / replace objects with damageable variants.
    2. ``reset()``:  reinitialise health, reset evaluators.
    3. ``step()``:   run physics, then update health for every tracked object.
    4. ``_process_obs()``: append health arrays to observations.

Simulator-specific subclasses override hooks for discovering objects,
loading configs, and running the underlying sim step.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np
import yaml

from damagesim.core.damageable_mixin import DamageableMixin


class DamageableEnvironment:
    """
    Base damage-tracking environment wrapper (mixed in with the real env).

    Designed to be used via multiple inheritance with the concrete simulator
    environment as the *other* parent:

        class MyDamageableEnv(DamageableEnvironment, SimBaseEnv):
            ...

    Args:
        damage_config: Dict overriding default damage flags.
        damage_trackable_objects_config: Pre-loaded YAML config, or ``None``
            to auto-discover from ``params/damageable_objects.yaml``.
    """

    def __init__(
        self,
        damage_config: Optional[dict] = None,
        damage_trackable_objects_config: Optional[dict] = None,
        **kwargs,
    ):
        # ── Damage flags ────────────────────────────────────────────────
        self._damage_config: dict = {
            "enable_damage": True,
            # TODO: Check if these are needed
            "auto_replace_objects": True,
            "track_robot_damage": True,
        }
        if damage_config is not None:
            self._damage_config.update(damage_config)

        self.lock_health: bool = not self._damage_config["enable_damage"]
        self.damage_evaluators_initialized: bool = False

        # ── Object-tracking config ──────────────────────────────────────
        if damage_trackable_objects_config is not None:
            self.damage_trackable_objects_config = damage_trackable_objects_config
        else:
            self.damage_trackable_objects_config = self._load_damage_trackable_objects_config()

        # Health visualisation state (can be used by any backend)
        self._health_visualization_enabled: bool = False
        self._health_fig = None
        self._health_ax = None
        self._health_bars_dict = None
        self._health_tracked_object_names: Optional[List[str]] = None
        self._damage_color_manager = None
        self.health_list_link_names: List[str] = []

    # ── Config loading ──────────────────────────────────────────────────

    @staticmethod
    def _load_damage_trackable_objects_config() -> dict:
        """
        Load ``damageable_objects.yaml`` from the ``params/`` directory
        adjacent to the *calling* module, with a silent fallback to ``{}``.

        Subclasses may override to point at a different config path.
        """
        # Default: empty → track everything
        return {}

    # ── Object discovery (subclass hooks) ───────────────────────────────

    def _get_all_objects(self) -> list:
        """
        Return every entity in the scene (objects + robots + fixtures).

        Subclasses **must** override this.
        """
        raise NotImplementedError

    # ── Health lock ─────────────────────────────────────────────────────

    def lock_health_changes(self) -> None:
        self.lock_health = True

    def unlock_health_changes(self) -> None:
        self.lock_health = False

    # ── Initialise damage tracking for discovered objects ───────────────

    def initialize_damageable_objects(self) -> None:
        """
        Walk over all objects and enable damage tracking.

        Two modes, selected automatically based on config:

        1. **Task-specific mode** (allowlist): if the current task name has an
           entry in the config, *only* the categories / names listed there are
           tracked.
        2. **Default mode** (denylist): track every ``DamageableMixin`` object
           whose category is NOT in the ``skip_categories`` list under the
           ``default`` config key.
        """
        task_name = getattr(self, "task_name", None) or getattr(self, "_task_name", None)
        task_cfg = None
        if task_name and task_name in self.damage_trackable_objects_config:
            task_cfg = self.damage_trackable_objects_config[task_name] or {}

        if task_name != "default":
            # ── Task-specific mode: allowlist ────────────────────────────
            allowed_categories: Set[str] = set(task_cfg.get("categories", []) or [])
            allowed_names: Set[str] = set(task_cfg.get("names", []) or [])

            for obj in self._get_all_objects():
                if not isinstance(obj, DamageableMixin):
                    continue

                obj_name = getattr(obj, "name", None)
                obj_category = getattr(obj, "category", None)
                should_track = False

                if obj_category in allowed_categories:
                    should_track = True
                if obj_name in allowed_names:
                    should_track = True

                # Fuzzy name <-> category match
                if not should_track and obj_name:
                    name_lower = obj_name.lower()
                    for cat in allowed_categories:
                        if cat and (cat.lower() in name_lower or name_lower in cat.lower()):
                            should_track = True
                            break

                if "agent" in allowed_categories and self._is_robot(obj):
                    should_track = True

                if should_track:
                    obj.set_track_damage(True)
                    obj.set_damageable_links_and_params()
                    obj.initialize_health()
                    obj._initialize_damage_evaluators()
                else:
                    obj.set_track_damage(False)
        else:
            # ── Default mode: track everything except skip_categories ────
            default_cfg = self.damage_trackable_objects_config.get("default", {})
            skip_categories: Set[str] = set(default_cfg.get("skip_categories", []) or [])

            for obj in self._get_all_objects():
                if not isinstance(obj, DamageableMixin):
                    continue

                obj_category = getattr(obj, "category", None)
                should_track = obj_category not in skip_categories

                if should_track:
                    obj.set_track_damage(True)
                    obj.set_damageable_links_and_params()
                    obj.initialize_health()
                    obj._initialize_damage_evaluators()
                else:
                    obj.set_track_damage(False)

    def _is_robot(self, obj) -> bool:
        """Heuristic to decide if *obj* is a robot."""
        if hasattr(obj, "robot_type"):
            return True
        name = getattr(obj, "name", "") or ""
        if "robot" in name.lower() and "platform" not in name.lower():
            return True
        # Check if obj is in the robots list (set by sim-specific subclass)
        robots = getattr(self, "robots", [])
        if isinstance(robots, (list, tuple)) and obj in robots:
            return True
        return False

    # ── Reset ───────────────────────────────────────────────────────────

    def _reset_damage_tracking(self) -> None:
        """
        Reset health and evaluators for all tracked objects.

        Intended to be called from the subclass ``reset()`` after the base
        env reset has completed.
        """
        for obj in self._get_all_objects():
            if isinstance(obj, DamageableMixin) and obj.track_damage:
                obj.reset_damage_evaluators()
                obj.initialize_health()

        self.health_list_link_names = self._build_health_list()

    def _build_health_list(self) -> List[str]:
        """Return ``['obj_name@part_name', ...]`` for all tracked parts."""
        result: List[str] = []
        for obj in self._get_all_objects():
            if isinstance(obj, DamageableMixin) and obj.track_damage:
                for link_name in obj.damageable_links:
                    result.append(f"{obj.name}@{link_name}")
        return result

    # ── Step helpers ────────────────────────────────────────────────────

    def _initialize_all_evaluators(self) -> None:
        """
        Lazily initialise damage evaluators on the first ``env.step()``.
        """
        for obj in self._get_all_objects():
            if isinstance(obj, DamageableMixin) and obj.track_damage:
                obj._initialize_damage_evaluators()
        self.damage_evaluators_initialized = True

    def _update_all_health(self) -> Dict[str, dict]:
        """
        Run ``update_health()`` on every tracked object and collect damage info.

        Returns:
            Dict mapping object names to their ``damage_info``.
        """
        obj_damage_info: Dict[str, dict] = {}
        if self.lock_health:
            return obj_damage_info

        for obj in self._get_all_objects():
            if isinstance(obj, DamageableMixin) and obj.track_damage:
                obj.update_health()
                obj_damage_info[obj.name] = obj.damage_info
        return obj_damage_info

    # ── Observation post-processing ─────────────────────────────────────

    def _append_health_to_obs(self, obs: dict) -> dict:
        """
        Append a ``health`` array (float32) to the observation dict.
        """
        health_values: List[float] = []
        for obj in self._get_all_objects():
            if isinstance(obj, DamageableMixin) and obj.track_damage:
                for link_name in obj.damageable_links:
                    health_values.append(obj.link_healths[link_name])
        obs["health"] = np.array(health_values, dtype=np.float32)
        return obs

    # ── Query helpers ───────────────────────────────────────────────────

    def get_damageable_objects(self) -> list:
        """Return list of all objects with damage tracking enabled."""
        return [
            obj
            for obj in self._get_all_objects()
            if isinstance(obj, DamageableMixin) and obj.track_damage
        ]

    def initialize_env_health(self) -> None:
        """Re-initialise health for all tracked objects without resetting evaluators."""
        for obj in self._get_all_objects():
            if isinstance(obj, DamageableMixin) and obj.track_damage:
                obj.initialize_health()

    def get_env_health(self) -> dict:
        """Return the health of the environment."""
        return {obj.name: obj.health for obj in self._get_all_objects() if isinstance(obj, DamageableMixin) and obj.track_damage}

