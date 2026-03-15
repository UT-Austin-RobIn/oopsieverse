"""
Environment registry for oopsieverse RoboCasa environments.

Provides centralized environment discovery and configuration.
"""

from dataclasses import dataclass
from importlib import import_module
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


# ═══════════════════════════════════════════════════════════════════════
# Environment registrations (lazy module import)
# ═══════════════════════════════════════════════════════════════════════

_ROBOCASA_ENVS = [
    ("pick_egg", "pick_egg", "PickEgg", "DamageablePickEgg"),
    ("pastry_display", "pastry_display", "PastryDisplay", "DamageablePastryDisplay"),
    ("open_single_door", "open_single_door", "OpenSingleDoor", "DamageableOpenSingleDoor"),
    ("close_microwave", "close_microwave", "CloseMicrowave", "DamageableCloseMicrowave"),
    ("turn_on_faucet", "turn_on_faucet", "TurnOnFaucet", "DamageableTurnOnFaucet"),
    ("turn_on_microwave", "turn_on_microwave", "TurnOnMicrowave", "DamageableTurnOnMicrowave"),
    ("turn_on_stove", "turn_on_stove", "TurnOnStove", "DamageableTurnOnStove"),
    ("open_drawer", "open_drawer", "OpenDrawer", "DamageableOpenDrawer"),
    ("close_drawer", "close_drawer", "CloseDrawer", "DamageableCloseDrawer"),
    ("place_plate", "place_plate", "PlacePlate", "DamageablePlacePlate"),
    ("counter_to_microwave", "counter_to_microwave", "CounterToMicrowave", "DamageableCounterToMicrowave"),
    ("prepare_coffee", "prepare_coffee", "PrepareCoffee", "DamageablePrepareCoffee"),
    ("shelve_item", "shelve_item", "ShelveItem", "DamageableShelveItem"),
    ("prepare_breakfast", "prepare_breakfast", "PrepareBreakfast", "DamageablePrepareBreakfast"),
    ("dirty_dishes", "dirty_dishes", "DirtyDishes", "DamageableDirtyDishes"),
    ("nav_to_counter", "nav_to_counter", "NavToCounter", "DamageableNavToCounter"),
    ("wipe_counter", "wipe_counter", "WipeCounter", "DamageableWipeCounter"),
]


def _register(env_name: str, module_name: str, env_cls_name: str, damageable_cls_name: str):
    module_path = f"{__package__}.robocasa.{module_name}" if __package__ else f"robocasa.{module_name}"
    module = import_module(module_path)
    env_cls = getattr(module, env_cls_name)
    damageable_cls = getattr(module, damageable_cls_name)

    EnvironmentRegistry.register(
        env_name,
        EnvConfig(
            env_class=env_cls,
            damageable_class=damageable_cls,
        ),
    )


for _env_name, _module_name, _env_cls_name, _damageable_cls_name in _ROBOCASA_ENVS:
    _register(_env_name, _module_name, _env_cls_name, _damageable_cls_name)
