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
    playback_cameras: Optional[List[str]] = None
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
    ("serve_pastry", "serve_pastry", "ServePastry", "DamageableServePastry"),
    ("open_single_door", "open_single_door", "OpenSingleDoor", "DamageableOpenSingleDoor"),
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
    ("dishes_to_sink", "dishes_to_sink", "DishesToSink", "DamageableDishesToSink"),
    ("nav_lift_bowl", "nav_lift_bowl", "NavLiftBowl", "DamageableNavLiftBowl"),
    ("wipe_counter", "wipe_counter", "WipeCounter", "DamageableWipeCounter"),
]


_ROBOT_OVERRIDES = {
    "shelve_item": "Panda",
}

# Per-env camera overrides for robots that lack the default kitchen agentview.
_CAMERA_OVERRIDES = {
    "shelve_item": "shelve_item_right",
}

# Cameras available for playback when --camera all_cameras (Panda lacks agentview).
_PLAYBACK_CAMERA_OVERRIDES = {
    "shelve_item": [
        "shelve_item_left",
        "shelve_item_right",
        "robot0_robotview",
        "robot0_eye_in_hand",
    ],
}


def _register(env_name: str, module_name: str, env_cls_name: str, damageable_cls_name: str):
    module_path = f"{__package__}.robocasa.{module_name}" if __package__ else f"robocasa.{module_name}"
    module = import_module(module_path)
    env_cls = getattr(module, env_cls_name)
    damageable_cls = getattr(module, damageable_cls_name)

    cfg_kwargs = {"env_class": env_cls, "damageable_class": damageable_cls}
    if env_name in _ROBOT_OVERRIDES:
        cfg_kwargs["robot"] = _ROBOT_OVERRIDES[env_name]
    if env_name in _CAMERA_OVERRIDES:
        cfg_kwargs["camera_name"] = _CAMERA_OVERRIDES[env_name]
    if env_name in _PLAYBACK_CAMERA_OVERRIDES:
        cfg_kwargs["playback_cameras"] = _PLAYBACK_CAMERA_OVERRIDES[env_name]

    EnvironmentRegistry.register(env_name, EnvConfig(**cfg_kwargs))


for _env_name, _module_name, _env_cls_name, _damageable_cls_name in _ROBOCASA_ENVS:
    _register(_env_name, _module_name, _env_cls_name, _damageable_cls_name)
