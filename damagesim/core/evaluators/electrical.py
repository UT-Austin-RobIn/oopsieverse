"""
Simulator-agnostic base for electrical damage evaluation.

Damage model
------------
Electrical damage is modelled as contact with conductive particles
(typically water). For each part:

    damage = max(0, particle_count − threshold) × scale

Subclasses must implement ``_count_particles_per_part`` and
``_get_damageable_part_names`` to read particle contact info from
the concrete simulator.
"""

from abc import abstractmethod
from typing import Dict, List, Optional

from damagesim.core.evaluators.base import DamageEvaluator


class ElectricalDamageEvaluator(DamageEvaluator):
    """
    Base electrical damage evaluator.

    Args:
        entity: Object or robot being evaluated.
        damage_threshold: Minimum particle count before damage starts.
        scale: Damage multiplier above threshold.
        part_thresholds: Optional per-part overrides. Keys are substrings
            matched (case-insensitive) against part names; values are dicts
            with optional keys ``damage_threshold`` and ``scale``.
    """

    def __init__(
        self,
        entity,
        damage_threshold: float = 0.0,
        scale: float = 1.0,
        part_thresholds: Optional[Dict[str, dict]] = None,
        **kwargs,
    ) -> None:
        super().__init__(entity, damage_threshold, scale)
        self.name = "electrical"
        self.part_thresholds: Dict[str, dict] = {
            k.lower(): v for k, v in (part_thresholds or {}).items()
        }

    # ── Abstract helpers ────────────────────────────────────────────────

    @abstractmethod
    def _count_particles_per_part(self) -> Dict[str, int]:
        """Return mapping: part_name → number of contacting conductive particles."""
        ...

    @abstractmethod
    def _get_damageable_part_names(self) -> List[str]:
        """Return names of damageable parts."""
        ...

    # ── Per-part override resolution ────────────────────────────────────

    def _resolve_part_overrides(self, part_name: str):
        """Return (threshold, scale) for *part_name* after applying overrides."""
        threshold = self.damage_threshold
        scale = self.scale
        if not self.part_thresholds:
            return threshold, scale

        pname = part_name.lower()
        matches = [k for k in self.part_thresholds if k in pname]
        if not matches:
            return threshold, scale

        # Prefer the longest matching substring for specificity
        chosen = max(matches, key=len)
        ov = self.part_thresholds[chosen]
        if isinstance(ov, dict):
            threshold = float(ov.get("damage_threshold", threshold))
            scale = float(ov.get("scale", scale))
        return threshold, scale

    # ── Core logic ──────────────────────────────────────────────────────

    def generate_damage(self) -> Dict[str, float]:
        counts = self._count_particles_per_part()
        damages: Dict[str, float] = {}
        for part_name, count in counts.items():
            thr, scl = self._resolve_part_overrides(part_name)
            damages[part_name] = min(100.0, max(0.0, float(count) - thr) * scl)
        return damages

    def reset_tracking(self) -> None:
        """No persistent state to reset."""
        pass

    def get_contact_summary(self) -> Dict[str, object]:
        """Lightweight summary of current conductive-particle contact counts."""
        counts = self._count_particles_per_part()
        total = sum(counts.values())
        details = {n: {"particle_count": c} for n, c in counts.items()}
        return {
            "status": "active" if total > 0 else "no_contact",
            "total_contact": total,
            "part_details": details,
        }

