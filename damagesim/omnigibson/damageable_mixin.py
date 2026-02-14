"""
OmniGibson-specific DamageableMixin and concrete damageable object / robot
classes.

These thin subclasses wire up the OG evaluator registry and provide the
``link_healths`` / ``damageable_links`` aliases expected by the rest of the
OG integration (environment wrapper, data wrappers, visualisations, …).
"""

from __future__ import annotations

from typing import Dict, List, Optional

from damagesim.core.damageable_mixin import DamageableMixin
from damagesim.omnigibson.params import PARAMS
from damagesim.omnigibson.evaluators import DAMAGE_EVALUATORS

# ── Lazy OG imports (fail gracefully when OG is not installed) ──────────
try:
    from omnigibson.objects.dataset_object import DatasetObject
    from omnigibson.objects.primitive_object import PrimitiveObject
    from omnigibson.objects.usd_object import USDObject
    from omnigibson.objects.controllable_object import ControllableObject
    from omnigibson.objects.light_object import LightObject
    from omnigibson.objects.stateful_object import StatefulObject
    from omnigibson.robots.franka import FrankaPanda
    from omnigibson.robots.franka_mounted import FrankaMounted
    from omnigibson.robots.tiago import Tiago
    from omnigibson.robots.r1pro import R1Pro
except ImportError:
    # Allows importing the module for type-checking / testing even without OG
    DatasetObject = object
    PrimitiveObject = object
    USDObject = object
    ControllableObject = object
    LightObject = object
    StatefulObject = object
    FrankaPanda = object
    FrankaMounted = object
    Tiago = object
    R1Pro = object


class OGDamageableMixin(DamageableMixin):
    """
    OmniGibson-specific DamageableMixin.

    Differences from the core base class:
    * Uses ``links`` / ``link_healths`` terminology.
    * Filters out ``usd_path`` kwarg (robots construct their own path).
    * Wires up the OG evaluator registry.
    """

    def __init__(self, *args, **kwargs):
        # OG robots generate their own USD path; remove it so super().__init__
        # doesn't pass it through to the OG base class twice.
        kwargs.pop("usd_path", None)
        super().__init__(*args, **kwargs)

    # ── Evaluator registry ──────────────────────────────────────────────

    def _get_evaluator_registry(self) -> dict:
        return DAMAGE_EVALUATORS

    # ── Part enumeration (OG → links) ──────────────────────────────────

    def _get_all_part_names(self) -> List[str]:
        if hasattr(self, "links"):
            return list(self.links.keys())
        return []

    # ── Setters ─────────────────────────────────────────────────────────
    def set_damageable_links_and_params(self) -> None:
        """
        Set which links track damage and the parameters for those links.
        """
        params = PARAMS
        if not self.track_damage:
            return
        cat = getattr(self, "category", "default")

        if cat in params:
            # 1. Set the parameters for the object. These parameters are shared across all the links of the object.
            # We provide functionality to override these parameters for each part / link of the object through link_config_overrides
            self.set_params(params[cat])

            # 2. Set the links of the object to be tracked for damage
            # For regular objects, we use a default key to get the damageable links
            links_key = params[cat].get("damageable_links")

            # For robots, we need to use the class name to get the damageable links
            cls_links_key = params[cat].get(
                f"{self.__class__.__name__.lower()}_damageable_links"
            )
            if cat == "agent" and cls_links_key is not None:
                self._set_damageable_links(cls_links_key)
            elif links_key:
                self._set_damageable_links(links_key)
            else:
                # In case config file is missing the damageable_links key, we use all the links of the object
                self._set_damageable_links(self.links.keys())
                    
        else:
            print(f"Warning: no damage params found for {self.category}, using default params. If you would like custom damage params, please add them to damage_params.py")
            self.set_params(params["default"])
            self._set_damageable_links(self.links.keys())

# ═══════════════════════════════════════════════════════════════════════
# Concrete damageable object classes
# ═══════════════════════════════════════════════════════════════════════

class DamageableDatasetObject(OGDamageableMixin, DatasetObject):
    pass


class DamageablePrimitiveObject(OGDamageableMixin, PrimitiveObject):
    pass


class DamageableUSDObject(OGDamageableMixin, USDObject):
    pass


class DamageableControllableObject(OGDamageableMixin, ControllableObject):
    pass


class DamageableLightObject(OGDamageableMixin, LightObject):
    pass


class DamageableStatefulObject(OGDamageableMixin, StatefulObject):
    pass


# ═══════════════════════════════════════════════════════════════════════
# Concrete damageable robot classes
# ═══════════════════════════════════════════════════════════════════════

class DamageableFrankaPanda(OGDamageableMixin, FrankaPanda):
    @property
    def usd_path(self):
        import os
        from omnigibson.utils.asset_utils import get_dataset_path
        return os.path.join(
            get_dataset_path("omnigibson-robot-assets"),
            "models/franka/franka_panda/usd/franka_panda.usda",
        )


class DamageableFrankaMounted(OGDamageableMixin, FrankaMounted):
    @property
    def usd_path(self):
        import os
        from omnigibson.utils.asset_utils import get_dataset_path
        return os.path.join(
            get_dataset_path("omnigibson-robot-assets"),
            "models/franka/franka_mounted/usd/franka_mounted.usda",
        )


class DamageableTiago(OGDamageableMixin, Tiago):
    @property
    def usd_path(self):
        import os
        from omnigibson.macros import gm
        return os.path.join(
            gm.DATA_PATH,
            "omnigibson-robot-assets/models/tiago/usd/tiago.usda",
        )


class DamageableR1Pro(OGDamageableMixin, R1Pro):
    @property
    def usd_path(self):
        import os
        from omnigibson.macros import gm
        return os.path.join(
            gm.DATA_PATH,
            "omnigibson-robot-assets/models/r1pro/usd/r1pro.usda",
        )

    @property
    def model_name(self):
        return "R1Pro"

