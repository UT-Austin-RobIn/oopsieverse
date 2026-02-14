"""
OmniGibson-specific mechanical damage evaluator.

Extends the core ``MechanicalDamageEvaluator`` with:
* Position-based velocity computation (``dx / dt``).
* Angular-acceleration impact force (via quaternion finite-differencing).
* Contact reading from OG ``link.contact_list()``.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

try:
    import torch as th
except ImportError:
    th = None

try:
    import omnigibson as og
    import omnigibson.utils.transform_utils as T
except ImportError:
    og = None
    T = None

from damagesim.core.evaluators.mechanical import MechanicalDamageEvaluator


# ── Helper ──────────────────────────────────────────────────────────────

def _angular_velocity_from_quat(q_prev, q_curr, dt):
    """Finite-difference angular velocity from two quaternions (x,y,z,w)."""
    q_rel = T.quat_multiply(T.quat_inverse(q_prev), q_curr)
    if q_rel[3] < 0:
        q_rel = -q_rel
    angle = 2 * th.acos(th.clamp(q_rel[3], -1.0, 1.0))
    sin_half = th.sqrt(1 - q_rel[3] ** 2)
    axis = q_rel[:3] / sin_half if sin_half >= 1e-6 else q_rel[:3]
    return axis * angle / dt


class OGMechanicalDamageEvaluator(MechanicalDamageEvaluator):
    """
    OmniGibson mechanical damage evaluator.

    Uses position displacement for linear velocity and quaternion
    finite-differencing for angular velocity.

    Extra constructor args accepted via ``**kwargs``:
        link_config_overrides: Synonym for ``part_config_overrides``
            (kept for backward compat with existing OG param dicts).
    """

    def __init__(self, entity, *, link_config_overrides=None, **kwargs):
        # Map OG-specific param name to the generic core name
        pco = kwargs.pop("part_config_overrides", None)
        if link_config_overrides is not None:
            pco = link_config_overrides
        super().__init__(entity, part_config_overrides=pco, **kwargs)

        # ── Extra tracking for position-based velocity ──────────────────
        self.prev_link_positions: Dict[str, th.Tensor] = {}
        self.prev_link_quats: Dict[str, th.Tensor] = {}
        self.prev_link_angular_velocities: Dict[str, th.Tensor] = {}

        # Alias for backward compat (OG mixin reads ``contacts_by_link``)
        self.contacts_by_link = self.contacts_by_part

        # Initialise tracking from current sim state
        self._init_og_tracking()

    # ── Tracking initialisation ─────────────────────────────────────────

    def _init_og_tracking(self):
        if not hasattr(self.entity, "links"):
            return
        for link_name, link in self.entity.links.items():
            try:
                pos, quat = link.get_position_orientation()
                self.prev_link_positions[link_name] = pos.clone()
                self.prev_link_quats[link_name] = quat.clone()
                self.prev_link_angular_velocities[link_name] = link.get_angular_velocity().clone()
            except Exception:
                pass

    # ── Abstract-method implementations ─────────────────────────────────

    def _get_damageable_part_names(self) -> List[str]:
        parts = getattr(self.entity, "damageable_links", None)
        if parts is None:
            parts = getattr(self.entity, "damageable_parts", [])
        return list(parts)

    def _get_part_linear_velocity(self, part_name: str) -> np.ndarray:
        """Compute linear velocity from position displacement."""
        link = self.entity.links[part_name]
        pos_current = link.get_position_orientation()[0]
        dt = max(self._get_timestep(), 1e-8)

        if part_name in self.prev_link_positions:
            displacement = pos_current - self.prev_link_positions[part_name]
            vel = displacement / dt
        else:
            vel = th.zeros(3) if th is not None else np.zeros(3)

        # Update position tracking
        self.prev_link_positions[part_name] = pos_current.clone()
        return vel.cpu().numpy() if hasattr(vel, "cpu") else np.asarray(vel)

    def _get_part_contacts(self, part_name: str) -> List[dict]:
        link = self.entity.links[part_name]
        try:
            raw = link.contact_list()
        except Exception:
            raw = []

        contacts = []
        dt = max(self._get_timestep(), 1e-8)
        for c in raw:
            impulse = np.asarray(c.impulse, dtype=np.float64)
            force = impulse / dt
            mag = float(np.linalg.norm(force))
            if mag > 1e-8:
                contacts.append({"force": force, "magnitude": mag})
        return contacts

    def _get_timestep(self) -> float:
        return float(og.sim.get_sim_step_dt())

    def _get_part_mass(self, part_name: str) -> float:
        link = self.entity.links[part_name]
        return float(getattr(link, "mass", 1.0))

    # ── Angular impact hook ─────────────────────────────────────────────

    def _get_angular_impact_magnitude(self, part_name: str, dt: float) -> float:
        """Compute angular-acceleration impact force via quaternion differencing."""
        link = self.entity.links[part_name]
        quat_current = link.get_position_orientation()[1]

        if part_name not in self.prev_link_quats:
            self.prev_link_quats[part_name] = quat_current.clone()
            self.prev_link_angular_velocities[part_name] = link.get_angular_velocity().clone()
            return 0.0

        try:
            ang_vel_current = _angular_velocity_from_quat(
                self.prev_link_quats[part_name], quat_current, max(dt, 1e-8)
            )
            ang_vel_prev = self.prev_link_angular_velocities[part_name]
            delta_ang = ang_vel_current - ang_vel_prev
            ang_accel = delta_ang / max(dt, 1e-8)
            mass = self._get_part_mass(part_name)
            ang_impact = mass * float(th.linalg.vector_norm(ang_accel).item())
        except Exception:
            ang_impact = 0.0

        # Update quaternion tracking
        self.prev_link_quats[part_name] = quat_current.clone()
        try:
            self.prev_link_angular_velocities[part_name] = ang_vel_current.clone()
        except Exception:
            self.prev_link_angular_velocities[part_name] = link.get_angular_velocity().clone()

        # Scale angular by 1/5 to make comparable to linear (empirical, as in original)
        return ang_impact / 5.0

    # ── Reset ───────────────────────────────────────────────────────────

    def reset_tracking(self) -> None:
        super().reset_tracking()
        self.prev_link_positions.clear()
        self.prev_link_quats.clear()
        self.prev_link_angular_velocities.clear()
        self._init_og_tracking()
        # Re-alias after base clears contacts_by_part
        self.contacts_by_link = self.contacts_by_part

    def update_link_positions_and_velocities(self):
        """Snapshot current positions and velocities (useful after teleport)."""
        for link_name, link in self.entity.links.items():
            pos, quat = link.get_position_orientation()
            self.prev_link_positions[link_name] = pos.clone()
            self.prev_link_quats[link_name] = quat.clone()
            # Also reset the base's velocity tracking
            vel = (pos - self.prev_link_positions.get(link_name, pos)) / max(self._get_timestep(), 1e-8)
            self.prev_linear_velocities[link_name] = vel.cpu().numpy() if hasattr(vel, "cpu") else np.asarray(vel)
            self.prev_link_angular_velocities[link_name] = link.get_angular_velocity().clone()

