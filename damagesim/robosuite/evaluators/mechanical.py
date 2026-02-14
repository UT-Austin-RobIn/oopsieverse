"""
Robosuite / MuJoCo-specific mechanical damage evaluator.

Extends the core ``MechanicalDamageEvaluator`` with:
* MuJoCo ``cvel`` for body linear velocity.
* MuJoCo contact array for contact forces.
* Body mass from ``model.body_mass``.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from damagesim.core.evaluators.mechanical import MechanicalDamageEvaluator


class RSMechanicalDamageEvaluator(MechanicalDamageEvaluator):
    """
    Robosuite mechanical damage evaluator.

    Reads physics state directly from the MuJoCo ``sim`` reference stored
    on the entity (set by ``DamageableEnvironment``).

    Extra constructor args accepted via ``**kwargs``:
        link_config_overrides: Synonym for ``part_config_overrides``
            (kept for backward compatibility with existing param dicts).
    """

    def __init__(self, entity, *, link_config_overrides=None, **kwargs):
        pco = kwargs.pop("part_config_overrides", None)
        if link_config_overrides is not None:
            pco = link_config_overrides
        super().__init__(entity, part_config_overrides=pco, **kwargs)

        # Alias for backward compat (RS mixin reads ``contacts_by_body``)
        self.contacts_by_body = self.contacts_by_part

        # Body-ID cache (populated lazily)
        self._body_id_cache: Dict[str, int] = {}

    # ── Helpers ─────────────────────────────────────────────────────────

    def _sim(self):
        return getattr(self.entity, "sim", None)

    def _body_id(self, body_name: str) -> Optional[int]:
        if body_name in self._body_id_cache:
            return self._body_id_cache[body_name]
        sim = self._sim()
        if sim is None:
            return None
        try:
            bid = sim.model.body_name2id(body_name)
            self._body_id_cache[body_name] = bid
            return bid
        except Exception:
            return None

    # ── Abstract-method implementations ─────────────────────────────────

    def _get_damageable_part_names(self) -> List[str]:
        parts = getattr(self.entity, "damageable_bodies", None)
        if parts is None:
            parts = getattr(self.entity, "damageable_parts", [])
        return list(parts)

    def _get_part_linear_velocity(self, part_name: str) -> np.ndarray:
        """Read linear velocity from MuJoCo cvel (spatial velocity of COM)."""
        sim = self._sim()
        bid = self._body_id(part_name)
        if sim is None or bid is None:
            return np.zeros(3)
        # cvel is 6D: [ang_x, ang_y, ang_z, lin_x, lin_y, lin_z]
        return sim.data.cvel[bid][3:6].copy()

    def _get_part_contacts(self, part_name: str) -> List[dict]:
        sim = self._sim()
        bid = self._body_id(part_name)
        if sim is None or bid is None:
            return []

        model = sim.model
        data = sim.data

        # Collect geoms belonging to this body
        body_geoms = set()
        for gid in range(model.ngeom):
            if model.geom_bodyid[gid] == bid:
                body_geoms.add(gid)
        if not body_geoms:
            return []

        contacts = []
        for i in range(data.ncon):
            contact = data.contact[i]
            if contact.geom1 not in body_geoms and contact.geom2 not in body_geoms:
                continue
            efc = contact.efc_address
            if efc < 0:
                continue
            normal_force = abs(data.efc_force[efc])
            normal_dir = contact.frame[:3]
            force_vec = normal_force * normal_dir
            if normal_force > 1e-6:
                contacts.append({
                    "body_id": bid,
                    "force": force_vec.copy(),
                    "magnitude": float(normal_force),
                    "geom1": contact.geom1,
                    "geom2": contact.geom2,
                })
        return contacts

    def _get_timestep(self) -> float:
        cf = getattr(self.entity, "control_freq", None)
        if cf is not None:
            return 1.0 / cf
        raise RuntimeError(
            f"control_freq not set on entity '{self.entity.name}'. "
            "Ensure DamageableEnvironment sets it before evaluation."
        )

    def _get_part_mass(self, part_name: str) -> float:
        sim = self._sim()
        bid = self._body_id(part_name)
        if sim is None or bid is None:
            return 1.0
        return float(sim.model.body_mass[bid])

    # ── Reset ───────────────────────────────────────────────────────────

    def reset_tracking(self) -> None:
        super().reset_tracking()
        self._body_id_cache.clear()
        # Re-alias after base clears contacts_by_part
        self.contacts_by_body = self.contacts_by_part

    def reinitialize_tracking(self) -> None:
        self._tracking_initialized = False
        self._body_id_cache.clear()

    def get_current_raw_force(self) -> float:
        mx = 0.0
        for pn in self.impact_forces:
            if self.impact_forces[pn]:
                mx = max(mx, self.impact_forces[pn][-1])
        for pn in self.filtered_qs_forces:
            if self.filtered_qs_forces[pn]:
                mx = max(mx, self.filtered_qs_forces[pn][-1])
        return mx

