"""
OmniGibson-specific electrical damage evaluator.

Reads water-particle contacts via OG ``ContactParticles`` state.
"""

from __future__ import annotations

from typing import Dict, List, Optional

try:
    from omnigibson import object_states
except ImportError:
    object_states = None

from damagesim.core.evaluators.electrical import ElectricalDamageEvaluator


class OGElectricalDamageEvaluator(ElectricalDamageEvaluator):
    """
    OmniGibson electrical evaluator.

    Uses the scene particle system to count water contacts per link.

    Extra constructor args:
        water_system_name: Name of the particle system (default ``"water"``).
        link_thresholds: Per-link overrides (synonym for ``part_thresholds``).
    """

    def __init__(
        self,
        entity,
        *,
        water_system_name: str = "water",
        link_thresholds: Optional[Dict[str, dict]] = None,
        **kwargs,
    ):
        # Map OG-specific param name to generic name
        pt = kwargs.pop("part_thresholds", None)
        if link_thresholds is not None:
            pt = link_thresholds
        super().__init__(entity, part_thresholds=pt, **kwargs)

        self.water_system_name = water_system_name
        self._water_system = None
        self._water_initialized = False

    def _ensure_water_system(self):
        if self._water_initialized:
            return
        scene = getattr(self.entity, "scene", None)
        if scene is None:
            self._water_initialized = True
            return
        for name in [self.water_system_name, "water", "sludge", "fluid"]:
            try:
                if scene.is_physical_particle_system(name):
                    self._water_system = scene.get_system(name)
                    break
            except Exception:
                continue
        self._water_initialized = True

    def _get_damageable_part_names(self) -> List[str]:
        parts = getattr(self.entity, "damageable_links", None)
        if parts is None:
            parts = getattr(self.entity, "damageable_parts", [])
        if not parts and hasattr(self.entity, "links"):
            parts = list(self.entity.links.keys())
        return list(parts)

    def _count_particles_per_part(self) -> Dict[str, int]:
        self._ensure_water_system()
        if self._water_system is None:
            return {n: 0 for n in self._get_damageable_part_names()}

        results: Dict[str, int] = {}
        for link_name in self._get_damageable_part_names():
            count = 0
            try:
                link = self.entity.links[link_name]
                particles = self.entity.states[object_states.ContactParticles].get_value(
                    system=self._water_system, link=link
                )
                count = len(particles)
            except Exception:
                count = 0
            results[link_name] = count
        return results

