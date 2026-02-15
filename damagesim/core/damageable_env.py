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
        Walk over all objects and enable damage tracking for those matching
        the YAML config (or all ``DamageableMixin`` instances when the config
        is empty / absent).
        """
        # Derive allowed categories / names from config
        damage_trackable_names: Set[str] = set()
        damage_trackable_categories: Set[str] = set()
        has_restrictions = False

        default_cfg = self.damage_trackable_objects_config.get("default", {})
        cats = default_cfg.get("categories", []) or []
        names = default_cfg.get("names", []) or []
        damage_trackable_categories.update(cats)
        damage_trackable_names.update(names)
        if cats or names:
            has_restrictions = True

        # Task-specific rules (subclass may set self._task_name or similar)
        task_name = getattr(self, "task_name", None) or getattr(self, "_task_name", None)
        if task_name and task_name in self.damage_trackable_objects_config:
            task_cfg = self.damage_trackable_objects_config[task_name] or {}
            t_cats = task_cfg.get("categories", []) or []
            t_names = task_cfg.get("names", []) or []
            damage_trackable_categories.update(t_cats)
            damage_trackable_names.update(t_names)
            if t_cats or t_names:
                has_restrictions = True

        for obj in self._get_all_objects():
            if not isinstance(obj, DamageableMixin):
                continue

            obj_name = getattr(obj, "name", None)
            obj_category = getattr(obj, "category", None)

            should_track = False

            if not has_restrictions:
                # No config restrictions → track everything
                should_track = True
            else:
                # Check direct category / name match
                if obj_category in damage_trackable_categories:
                    should_track = True
                if obj_name in damage_trackable_names:
                    should_track = True

                # Fuzzy name ↔ category match
                if not should_track and obj_name:
                    name_lower = obj_name.lower()
                    for cat in damage_trackable_categories:
                        if cat and (cat.lower() in name_lower or name_lower in cat.lower()):
                            should_track = True
                            break

                # Always track robots if "agent" is in tracked categories
                if "agent" in damage_trackable_categories and self._is_robot(obj):
                    should_track = True

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
        if "robot" in name.lower():
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

