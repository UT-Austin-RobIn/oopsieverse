"""
Abstract base class for all damage evaluators.

A damage evaluator inspects the current state of an entity (object / robot)
and returns per-part damage values that are subtracted from health.
"""

from abc import ABC, abstractmethod
from typing import Dict


class DamageEvaluator(ABC):
    """
    Base class for damage evaluators.

    Subclasses must implement ``generate_damage`` which returns a mapping from
    part names (links in OG, bodies in MuJoCo) to scalar damage values.

    Args:
        entity: The object or robot being evaluated.
        damage_threshold: Minimum damage potential before any damage is applied.
        scale: Multiplier applied to the damage potential above threshold.
    """

    def __init__(
        self,
        entity,
        damage_threshold: float = 0.0,
        scale: float = 1.0,
    ) -> None:
        self.entity = entity
        self.damage_threshold = float(damage_threshold)
        self.scale = float(scale)
        self.name: str = "base"

    @abstractmethod
    def generate_damage(self) -> Dict[str, float]:
        """
        Compute per-part damage for the current simulation step.

        Returns:
            Dict mapping part name → damage value (≥ 0).
        """
        ...

    def reset_tracking(self) -> None:
        """Reset any internal tracking state (called on env.reset)."""
        pass

    def reinitialize_tracking(self) -> None:
        """
        Force re-initialisation of tracking state.

        Useful after teleporting objects so that velocity deltas don't
        produce spurious impact spikes.
        """
        pass

