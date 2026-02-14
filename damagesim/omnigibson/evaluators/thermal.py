"""
OmniGibson-specific thermal damage evaluator.

Reads temperature from the OG ``Temperature`` object-state.
"""

from __future__ import annotations

from typing import List

try:
    from omnigibson import object_states
except ImportError:
    object_states = None

from damagesim.core.evaluators.thermal import ThermalDamageEvaluator


class OGThermalDamageEvaluator(ThermalDamageEvaluator):
    """
    OmniGibson thermal evaluator.

    Automatically adds a ``Temperature`` state to the entity if it doesn't
    already have one.
    """

    def __init__(self, entity, **kwargs):
        super().__init__(entity, **kwargs)

        # Ensure entity has a Temperature state
        if object_states is not None and object_states.Temperature not in self.entity.states:
            temp_state = object_states.Temperature(obj=self.entity)
            self.entity.add_state(temp_state)
            if getattr(self.entity, "_initialized", False):
                temp_state.initialize()

    def _get_temperature(self) -> float:
        if object_states is None:
            return 0.0
        if object_states.Temperature not in self.entity.states:
            try:
                temp_state = object_states.Temperature(obj=self.entity)
                self.entity.add_state(temp_state)
                if getattr(self.entity, "_initialized", False):
                    temp_state.initialize()
            except Exception:
                return 0.0
        try:
            return float(self.entity.states[object_states.Temperature].get_value())
        except Exception:
            return 0.0

    def _get_damageable_part_names(self) -> List[str]:
        parts = getattr(self.entity, "damageable_links", None)
        if parts is None:
            parts = getattr(self.entity, "damageable_parts", [])
        if not parts and hasattr(self.entity, "links"):
            parts = list(self.entity.links.keys())
        return list(parts)

