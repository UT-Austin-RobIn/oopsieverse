"""
Environment registry for oopsieverse RoboCasa environments.

Provides centralized environment discovery and configuration.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Type


@dataclass
class EnvConfig:
    """Configuration for a registered environment."""

    env_class: Type
    damageable_class: Type
    camera_name: str = "robot0_agentview_right"
    robot: str = "PandaOmron"
    control_freq: int = 20
    damageable_objects: Optional[List[str]] = None


class EnvironmentRegistry:
    """Registry for environment classes and their configurations."""

    _registry: Dict[str, EnvConfig] = {}

    @classmethod
    def register(cls, name: str, config: EnvConfig):
        """Register an environment with its configuration."""
        cls._registry[name] = config

    @classmethod
    def get(cls, name: str) -> EnvConfig:
        """Retrieve environment configuration by name.

        Raises:
            ValueError: If the environment name is not registered.
        """
        if name not in cls._registry:
            available = ", ".join(cls._registry.keys())
            raise ValueError(f"Unknown environment: {name}. Available environments: {available}")
        return cls._registry[name]

    @classmethod
    def list_envs(cls) -> List[str]:
        """List all registered environment names."""
        return list(cls._registry.keys())


# ---------------------------------------------------------------------------
# Environment registrations
# ---------------------------------------------------------------------------

from envs.robocasa.pick_egg import PickEgg, DamageablePickEgg  # noqa: E402
from envs.robocasa.pastry_display import PastryDisplay, DamageablePastryDisplay

EnvironmentRegistry.register(
    "pick_egg",
    EnvConfig(
        env_class=PickEgg,
        damageable_class=DamageablePickEgg,
    ),
)

EnvironmentRegistry.register(
    "pastry_display",
    EnvConfig(
        env_class=PastryDisplay,
        damageable_class=DamageablePastryDisplay,
    ),
)
