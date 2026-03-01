"""
Simulator-agnostic mixin that adds health / damage tracking to any object.

Terminology
-----------
This base class uses the word **link** to refer to the atomic rigid element
that health is tracked on.  In OmniGibson that element is a USD *link*; in
MuJoCo / Robosuite it is a *body*.  Simulator-specific subclasses should
provide the expected aliases (``link_healths``, ``body_healths``, …) as
thin property wrappers.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class DamageableMixin:
    """
    Mixin class that adds health and damage tracking to simulator objects.

    This class is designed to be used via multiple inheritance.  It should
    appear **before** the simulator-specific object class in the MRO so that
    ``super().__init__`` reaches the correct base.

    Args (via ``**kwargs``):
        params: Damage parameter dict for this object.
    """

    def __init__(self, *args, params: Optional[dict] = None, **kwargs):
        super().__init__(*args, **kwargs)

        # ── Core state ──────────────────────────────────────────────────
        self.track_damage: bool = False
        self.damageable_links: List[str] = []
        self.damage_params: dict = params or {}
        self.damage_evaluators: list = []

        self.link_healths: Dict[str, float] = {}
        self.damage_info: Dict[str, dict] = {}


    # ── Evaluator registry (set by subclass / backend __init__.py) ─────

    def _get_evaluator_registry(self) -> dict:
        """
        Return the mapping ``evaluator_name → EvaluatorClass``.

        Subclasses MUST override this to point at the correct backend
        evaluator registry.
        """
        raise NotImplementedError(
            "Subclass must implement _get_evaluator_registry()"
        )

    # ── Health initialisation ───────────────────────────────────────────

    def _get_all_part_names(self) -> List[str]:
        """
        Return a list of **all** part names belonging to this entity.

        Subclasses must override – for OG this reads ``self.links.keys()``,
        for Robosuite it reads body names from the MuJoCo model.
        """
        raise NotImplementedError(
            "Subclass must implement _get_all_part_names()"
        )

    def initialize_health(self) -> None:
        """Set all damageable-part healths to 100."""
        self.damage_info = {}
        for link_name in self.damageable_links:
            self.link_healths[link_name] = 100.0

    # ── Evaluator lifecycle ─────────────────────────────────────────────

    def _initialize_damage_evaluators(self) -> None:
        """
        Instantiate evaluators listed in ``self.damage_params["damage_evaluators"]``.

        Clears existing evaluators first to prevent duplication on repeated
        calls (e.g. after ``env.reset``).
        """
        self.damage_evaluators = []
        registry = self._get_evaluator_registry()

        for name in self.damage_params.get("damage_evaluators", []):
            if name not in registry:
                print(f"Warning: unknown damage evaluator '{name}'")
                continue
            cls = registry[name]
            eval_params = self.damage_params.get(name, {})
            self.damage_evaluators.append(cls(self, **eval_params))

    def reset_damage_evaluators(self) -> None:
        """Reset tracking state in every evaluator (called on ``env.reset``)."""
        for ev in self.damage_evaluators:
            ev.reset_tracking()

    def reinitialize_damage_tracking(self) -> None:
        """Force re-initialisation (e.g. after teleporting an object)."""
        for ev in self.damage_evaluators:
            if hasattr(ev, "reinitialize_tracking"):
                ev.reinitialize_tracking()

    # ── Setters ─────────────────────────────────────────────────────────
    
    def _set_damageable_links(self, value):
        self.damageable_links = list(value)
    
    def set_track_damage(self, enabled: bool) -> None:
        self.track_damage = enabled

    def set_params(self, params: dict) -> None:
        self.damage_params = params

    def set_link_healths(self, value: Dict[str, float]):
        self.link_healths = value
    
    def set_damageable_links_and_params(self) -> None:
        """
        Set which links track damage and the parameters for those links.
        """
        pass


    # ── Getters ──────────────────────────────────────────────────

    @property
    def health(self) -> float:
        """Minimum health across all tracked parts (100 = full)."""
        if not self.link_healths:
            return 100.0
        return min(self.link_healths.values())

    def is_destroyed(self) -> bool:
        return self.health <= 0.0

    # ── Health update (called every env.step) ───────────────────────────

    def update_health(self) -> None:
        """
        Run all evaluators, apply damage, and record ``damage_info``.
        """
        self.damage_info = {}

        for evaluator in self.damage_evaluators:
            part_damages = evaluator.generate_damage()

            for part_name, damage in part_damages.items():
                if part_name not in self.link_healths:
                    self.link_healths[part_name] = 100.0

                new_health = max(0.0, self.link_healths[part_name] - damage)
                self.link_healths[part_name] = new_health

                if part_name not in self.damage_info:
                    self.damage_info[part_name] = {}

                self._record_evaluator_info(evaluator, part_name, damage)

    def _record_evaluator_info(self, evaluator, part_name: str, damage: float) -> None:
        """
        Store per-evaluator diagnostics in ``self.damage_info``.

        Centralises the bookkeeping that was duplicated across both repos.
        """
        ev_name = evaluator.name
        info = self.damage_info.setdefault(part_name, {}).setdefault(ev_name, {})
        info["damage"] = damage

        if ev_name == "mechanical":
            info["impact_forces"] = (
                evaluator.impact_forces.get(part_name, [0.0])[-1]
            )
            # Use the generic attribute name from the core evaluator
            contacts_attr = "contacts_by_part"
            if hasattr(evaluator, "contacts_by_link"):
                contacts_attr = "contacts_by_link"
            elif hasattr(evaluator, "contacts_by_body"):
                contacts_attr = "contacts_by_body"

            info["filtered_raw_sim_forces"] = (
                evaluator.filtered_raw_sim_forces.get(part_name, [0.0])[-1]
            )
            info["filtered_qs_forces"] = (
                evaluator.filtered_qs_forces.get(part_name, [0.0])[-1]
            )
            info["unfiltered_raw_sim_forces"] = (
                evaluator.unfiltered_raw_sim_forces.get(part_name, [0.0])[-1]
            )
            info["unfiltered_qs_forces"] = (
                evaluator.unfiltered_qs_forces.get(part_name, [0.0])[-1]
            )
            contacts_store = getattr(evaluator, contacts_attr, {})
            info["contacts"] = contacts_store.get(part_name, [[]])[-1]

        elif ev_name == "thermal":
            temp = getattr(evaluator, "current_temperature", None)
            info["temperature"] = temp
            info["heating_threshold"] = getattr(evaluator, "heating_threshold", None)
            info["cooling_threshold"] = getattr(evaluator, "cooling_threshold", None)

        elif ev_name == "electrical":
            if hasattr(evaluator, "get_contact_summary"):
                summary = evaluator.get_contact_summary()
                details_key = "link_details" if "link_details" in summary else "part_details"
                details = summary.get(details_key, {})
                info["particle_count"] = details.get(part_name, {}).get("particle_count", 0)
            else:
                info["particle_count"] = 0

    # ── Reset helpers ───────────────────────────────────────────────────

    def reset_health(self) -> None:
        """Reset all part healths to 100."""
        for part_name in list(self.link_healths.keys()):
            self.link_healths[part_name] = 100.0
