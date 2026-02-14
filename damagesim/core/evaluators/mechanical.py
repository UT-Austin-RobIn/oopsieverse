"""
Simulator-agnostic base for mechanical damage evaluation.

Damage model
------------
1. **Impact force** – computed from finite-differenced linear acceleration:
       F_impact = mass × |Δv / Δt|
   (OG also tracks angular acceleration; the base exposes a hook for it.)

2. **Quasistatic (QS) force** – sum of contact forces with the component
   aligned with the acceleration direction removed:
       F_qs = Σ ‖F_contact − (F_contact · â) â‖   for each contact

3. **Damage potential** –
       D = impact_sensitivity × F_impact + qs_sensitivity × F_qs

4. **Damage** –
       damage = clamp(0, 100, (D − threshold) × scale)

Subclasses must implement a small set of abstract helpers that retrieve
velocities, contacts, masses, and time-steps from the concrete simulator.
"""

from abc import abstractmethod
from typing import Dict, List, Optional

import numpy as np

from damagesim.core.evaluators.base import DamageEvaluator


class MechanicalDamageEvaluator(DamageEvaluator):
    """
    Base mechanical damage evaluator with shared physics logic.

    Simulator-specific backends override the ``_get_*`` helpers.

    Args:
        entity: Object or robot being evaluated.
        damage_threshold: Minimum damage potential before damage is applied.
        damage_scale: Multiplier on damage above threshold.
        impact_damage_sensitivity: Multiplier on impact force.
        qs_damage_sensitivity: Multiplier on quasistatic force.
        part_config_overrides: Optional per-part overrides. Keys are substrings
            matched (case-insensitive) against part names; values are dicts
            with optional keys ``damage_threshold``, ``damage_scale``,
            ``impact_damage_sensitivity``, ``qs_damage_sensitivity``.
    """

    def __init__(
        self,
        entity,
        damage_threshold: float = 0.0,
        damage_scale: float = 1.0,
        impact_damage_sensitivity: float = 1.0,
        qs_damage_sensitivity: float = 1.0,
        part_config_overrides: Optional[Dict[str, dict]] = None,
        # Accept and ignore unknown kwargs so sim-specific params pass through
        **kwargs,
    ) -> None:
        super().__init__(entity, damage_threshold, damage_scale)
        self.damage_scale = float(damage_scale)
        self.impact_damage_sensitivity = float(impact_damage_sensitivity)
        self.qs_damage_sensitivity = float(qs_damage_sensitivity)
        self.part_config_overrides: Dict[str, dict] = {
            k.lower(): v for k, v in (part_config_overrides or {}).items()
        }
        self.name = "mechanical"

        # ── Tracking state ──────────────────────────────────────────────
        self.prev_linear_velocities: Dict[str, np.ndarray] = {}
        self._tracking_initialized: bool = False

        # ── Logging / diagnostics ───────────────────────────────────────
        self.damage_potentials: Dict[str, float] = {}
        self.impact_forces: Dict[str, List[float]] = {}
        self.unfiltered_raw_sim_forces: Dict[str, List[float]] = {}
        self.filtered_raw_sim_forces: Dict[str, List[float]] = {}
        self.unfiltered_qs_forces: Dict[str, List[float]] = {}
        self.filtered_qs_forces: Dict[str, List[float]] = {}
        self.contacts_by_part: Dict[str, list] = {}

    # ── Abstract helpers (must be implemented by each backend) ──────────

    @abstractmethod
    def _get_damageable_part_names(self) -> List[str]:
        """Return names of parts that should be evaluated for damage."""
        ...

    @abstractmethod
    def _get_part_linear_velocity(self, part_name: str) -> np.ndarray:
        """Return the current linear velocity [vx, vy, vz] for *part_name*."""
        ...

    @abstractmethod
    def _get_part_contacts(self, part_name: str) -> List[dict]:
        """
        Return contacts for *part_name*.

        Each element must be a dict with at least:
            ``force``  – np.ndarray of shape (3,)
            ``magnitude`` – float (norm of force vector)
        """
        ...

    @abstractmethod
    def _get_timestep(self) -> float:
        """Return the time-step (seconds) for one ``env.step()``."""
        ...

    @abstractmethod
    def _get_part_mass(self, part_name: str) -> float:
        """Return mass (kg) for *part_name*."""
        ...

    # ── Optional hook for angular acceleration (used by OG) ─────────────

    def _get_angular_impact_magnitude(self, part_name: str, dt: float) -> float:
        """
        Return angular-acceleration–based impact force magnitude.

        The default implementation returns 0.  OG overrides this to include
        angular velocity finite-differencing.
        """
        return 0.0

    # ── Core damage logic (shared) ──────────────────────────────────────

    def generate_damage(self) -> Dict[str, float]:  # noqa: C901 – complexity acceptable
        part_damages: Dict[str, float] = {}
        dt = self._get_timestep()
        eps = 1e-8

        for part_name in self._get_damageable_part_names():
            # Ensure logging lists exist
            if part_name not in self.impact_forces:
                self.impact_forces[part_name] = []
                self.unfiltered_raw_sim_forces[part_name] = []
                self.filtered_raw_sim_forces[part_name] = []
                self.unfiltered_qs_forces[part_name] = []
                self.filtered_qs_forces[part_name] = []
                self.contacts_by_part[part_name] = []

            # ── Contacts ────────────────────────────────────────────────
            contacts = self._get_part_contacts(part_name)
            self.contacts_by_part[part_name].append(contacts)

            # ── Linear velocity / acceleration ──────────────────────────
            vel_current = np.asarray(self._get_part_linear_velocity(part_name), dtype=np.float64)
            has_prev = part_name in self.prev_linear_velocities

            delta_v = np.zeros(3, dtype=np.float64)
            accel = np.zeros(3, dtype=np.float64)
            impact_force_mag = 0.0

            if has_prev:
                vel_prev = self.prev_linear_velocities[part_name]
                delta_v = vel_current - vel_prev
                accel = delta_v / max(dt, eps)
                accel_mag = float(np.linalg.norm(accel))
                mass = self._get_part_mass(part_name)
                impact_force_mag = mass * accel_mag

            # Optional angular component
            angular_impact = self._get_angular_impact_magnitude(part_name, dt)
            # Match OG: take max of linear and (scaled) angular
            impact_force_mag = max(impact_force_mag, angular_impact)

            self.impact_forces[part_name].append(impact_force_mag)
            self.prev_linear_velocities[part_name] = vel_current.copy()

            # ── Quasistatic force decomposition ─────────────────────────
            delta_v_norm = float(np.linalg.norm(delta_v))
            adjust = delta_v_norm >= eps

            raw_sim_force = 0.0
            qs_force = 0.0

            if contacts:
                for c in contacts:
                    raw_sim_force += c["magnitude"]

                if not adjust:
                    qs_force = raw_sim_force
                else:
                    accel_unit = accel / float(np.linalg.norm(accel))
                    for c in contacts:
                        f_vec = np.asarray(c["force"], dtype=np.float64)
                        proj = float(np.dot(f_vec, accel_unit))
                        if proj > 0:
                            eff = f_vec - proj * accel_unit
                            qs_force += float(np.linalg.norm(eff))
                        else:
                            qs_force += c["magnitude"]

            self.unfiltered_raw_sim_forces[part_name].append(raw_sim_force)
            self.filtered_raw_sim_forces[part_name].append(raw_sim_force)
            self.unfiltered_qs_forces[part_name].append(qs_force)
            self.filtered_qs_forces[part_name].append(qs_force)

            # ── Per-part overrides ──────────────────────────────────────
            p_impact_sens = self.impact_damage_sensitivity
            p_qs_sens = self.qs_damage_sensitivity
            p_threshold = self.damage_threshold
            p_scale = self.damage_scale

            part_lower = part_name.lower()
            for key, overrides in self.part_config_overrides.items():
                if key in part_lower:
                    p_impact_sens = overrides.get("impact_damage_sensitivity", p_impact_sens)
                    p_qs_sens = overrides.get("qs_damage_sensitivity", p_qs_sens)
                    p_threshold = overrides.get("damage_threshold", p_threshold)
                    p_scale = overrides.get("damage_scale", p_scale)
                    break

            # ── Damage computation ──────────────────────────────────────
            impact_potential = p_impact_sens * impact_force_mag
            qs_potential = p_qs_sens * qs_force
            total_potential = impact_potential + qs_potential
            self.damage_potentials[part_name] = total_potential

            damage = min(100.0, max(0.0, (total_potential - p_threshold)) * p_scale)
            part_damages[part_name] = damage

        return part_damages

    # ── Reset / reinitialise ────────────────────────────────────────────

    def reset_tracking(self) -> None:
        self.prev_linear_velocities.clear()
        self.damage_potentials.clear()
        self.impact_forces.clear()
        self.unfiltered_raw_sim_forces.clear()
        self.filtered_raw_sim_forces.clear()
        self.unfiltered_qs_forces.clear()
        self.filtered_qs_forces.clear()
        self.contacts_by_part.clear()
        self._tracking_initialized = False

    def reinitialize_tracking(self) -> None:
        self._tracking_initialized = False
        # Subclasses may re-read current velocities here

    def get_current_raw_force(self) -> float:
        """Max of latest impact + QS forces across all tracked parts."""
        mx = 0.0
        for pn in self.impact_forces:
            if self.impact_forces[pn]:
                mx = max(mx, self.impact_forces[pn][-1])
        for pn in self.filtered_qs_forces:
            if self.filtered_qs_forces[pn]:
                mx = max(mx, self.filtered_qs_forces[pn][-1])
        return mx

