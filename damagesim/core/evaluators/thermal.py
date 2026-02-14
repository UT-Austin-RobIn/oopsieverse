"""
Simulator-agnostic base for thermal damage evaluation.

Damage model
------------
- If temperature > heating_threshold:
      damage = scale × |temperature − heating_threshold|
- If temperature < cooling_threshold:
      damage = scale × |temperature − cooling_threshold|
- Otherwise damage = 0.

Subclasses must implement ``_get_temperature`` to read the current
temperature from their simulator.
"""

from abc import abstractmethod
from typing import Dict, List

from damagesim.core.evaluators.base import DamageEvaluator


class ThermalDamageEvaluator(DamageEvaluator):
    """
    Base thermal damage evaluator.

    Args:
        entity: Object or robot being evaluated.
        heating_threshold: Temperature above which heating damage starts.
        scale: Damage multiplier.
        cooling_threshold: Temperature below which cooling damage starts.
    """

    def __init__(
        self,
        entity,
        heating_threshold: float,
        scale: float = 1.0,
        cooling_threshold: float = -1e10,
        **kwargs,
    ) -> None:
        super().__init__(entity, heating_threshold, scale)
        self.heating_threshold = float(heating_threshold)
        self.cooling_threshold = float(cooling_threshold)
        self.name = "thermal"
        self.current_temperature: float = 0.0

    # ── Abstract helpers ────────────────────────────────────────────────

    @abstractmethod
    def _get_temperature(self) -> float:
        """Return the current temperature of the entity."""
        ...

    @abstractmethod
    def _get_damageable_part_names(self) -> List[str]:
        """Return names of parts that should receive thermal damage."""
        ...

    # ── Core logic ──────────────────────────────────────────────────────

    def generate_damage(self) -> Dict[str, float]:
        damage = 0.0
        self.current_temperature = self._get_temperature()

        if self.current_temperature > self.heating_threshold:
            damage = min(100.0, abs(self.scale * (self.current_temperature - self.heating_threshold)))
        elif self.current_temperature < self.cooling_threshold:
            damage = min(100.0, abs(self.scale * (self.current_temperature - self.cooling_threshold)))

        return {part_name: damage for part_name in self._get_damageable_part_names()}

    def get_temperature(self) -> float:
        """Return the last-read temperature."""
        return self.current_temperature

