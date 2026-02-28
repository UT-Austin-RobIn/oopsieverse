"""
Robosuite-specific DamageableMixin and concrete damageable object / robot /
fixture classes.

Provides:
* ``RSDamageableMixin`` — Robosuite-aware core mixin
* ``DamageableRobotMixin`` / specific robot classes
* ``DamageableFixtureMixin`` / ``create_damageable_from_fixture``
* ``DamageableMJCFObject``, ``DamageableXMLObject``
* Primitive objects: ``DamageableBox``, ``DamageableBall``, etc.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from damagesim.core.damageable_mixin import DamageableMixin
from damagesim.robosuite.evaluators import DAMAGE_EVALUATORS

# ── Lazy Robosuite / RoboCasa imports ───────────────────────────────────
try:
    from robosuite.robots import (
        FixedBaseRobot,
        MobileRobot,
        WheeledRobot,
        LeggedRobot,
        ROBOT_CLASS_MAPPING,
    )
except ImportError:
    FixedBaseRobot = object
    MobileRobot = object
    WheeledRobot = object
    LeggedRobot = object
    ROBOT_CLASS_MAPPING = {}

try:
    from robocasa.models.fixtures.fixture import Fixture as _RobocasaFixture
except ImportError:
    _RobocasaFixture = object

try:
    from robocasa.models.objects.objects import MJCFObject as _MJCFObject
except ImportError:
    _MJCFObject = object

try:
    from robosuite.models.objects import MujocoXMLObject as _MujocoXMLObject
except ImportError:
    _MujocoXMLObject = object

try:
    from robosuite.models.objects.primitive import (
        BoxObject as _BoxObject,
        BallObject as _BallObject,
        CylinderObject as _CylinderObject,
        CapsuleObject as _CapsuleObject,
    )
except ImportError:
    _BoxObject = object
    _BallObject = object
    _CylinderObject = object
    _CapsuleObject = object


# ═══════════════════════════════════════════════════════════════════════
# RS-specific base mixin
# ═══════════════════════════════════════════════════════════════════════

class RSDamageableMixin(DamageableMixin):
    """
    Robosuite-specific DamageableMixin.

    Differences from the core:
    * Uses ``body_healths`` / ``damageable_bodies`` terminology.
    * Carries ``sim`` and ``control_freq`` references (set by env).
    * Wires up the RS evaluator registry.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # MuJoCo references (populated by DamageableEnvironment)
        if not hasattr(self, "sim"):
            self.sim = None
        if not hasattr(self, "control_freq"):
            self.control_freq = None

    # ── Evaluator registry ──────────────────────────────────────────────

    def _get_evaluator_registry(self) -> dict:
        return DAMAGE_EVALUATORS

    # ── Part enumeration ────────────────────────────────────────────────

    def _get_all_part_names(self) -> List[str]:
        """
        Return body names from the MuJoCo model.

        Falls back to ``root_body`` if available.
        """
        if self.sim is not None and hasattr(self, "root_body"):
            try:
                bid = self.sim.model.body_name2id(self.root_body)
                return [self.sim.model.body_id2name(bid)]
            except Exception:
                pass
        return []

    # ── Bridge to core attributes ──────────────────────────────────────

    @property
    def damageable_parts(self) -> List[str]:
        return self.damageable_links

    @damageable_parts.setter
    def damageable_parts(self, value):
        self.damageable_links = list(value)

    @property
    def part_healths(self) -> Dict[str, float]:
        return self.link_healths

    @part_healths.setter
    def part_healths(self, value: Dict[str, float]):
        self.link_healths = value

    def _initialize_health(self):
        self.initialize_health()

    # ── Aliases ─────────────────────────────────────────────────────────

    @property
    def body_healths(self) -> Dict[str, float]:
        return self.part_healths

    @body_healths.setter
    def body_healths(self, value: Dict[str, float]):
        self.part_healths = value

    @property
    def damageable_bodies(self) -> List[str]:
        return self.damageable_parts

    @damageable_bodies.setter
    def damageable_bodies(self, value):
        self.damageable_parts = list(value)

    def set_damageable_bodies(self, bodies=None):
        """RS-idiomatic alias for ``set_damageable_parts``."""
        if bodies is None:
            # Default: try root body
            if hasattr(self, "root_body") and self.sim is not None:
                try:
                    bid = self.sim.model.body_name2id(self.root_body)
                    name = self.sim.model.body_id2name(bid)
                    self.damageable_parts = [name]
                    return
                except Exception:
                    pass
            self.damageable_parts = []
        else:
            self.damageable_parts = list(bodies)

    # Backward-compat damage_statuses dict
    @property
    def damage_statuses(self) -> dict:
        if not hasattr(self, "_damage_statuses"):
            self._damage_statuses = {}
        return self._damage_statuses

    @damage_statuses.setter
    def damage_statuses(self, val):
        self._damage_statuses = val


# ═══════════════════════════════════════════════════════════════════════
# Robot classes
# ═══════════════════════════════════════════════════════════════════════

class DamageableRobotMixin(RSDamageableMixin):
    """Adds robot-specific auto-detection of damageable bodies."""

    def __init__(self, *args, params=None, **kwargs):
        robot_type = args[0] if args else kwargs.get("robot_type", "default")
        if params is None:
            from damagesim.robosuite.params import OBJECT_PARAMS
            params = OBJECT_PARAMS.get(robot_type, OBJECT_PARAMS.get("default", {})).copy()
        super().__init__(*args, params=params, **kwargs)
        self.track_damage = True

    def _initialize_health(self):
        if self.sim is None:
            return
        if not self.damageable_parts:
            self.damageable_parts = self._get_robot_body_names()
        super()._initialize_health()

    def _get_robot_body_names(self) -> List[str]:
        if self.damageable_parts:
            valid = []
            for bn in self.damageable_parts:
                try:
                    self.sim.model.body_name2id(bn)
                    valid.append(bn)
                except Exception:
                    pass
            return valid
        bodies = []
        prefix = "robot0"
        for i in range(self.sim.model.nbody):
            bn = self.sim.model.body_id2name(i)
            if bn and (bn.startswith(prefix) or bn.startswith("gripper0")):
                bodies.append(bn)
        return bodies

    @property
    def root_body(self):
        if self.damageable_parts:
            return self.damageable_parts[0]
        return f"{self.name}_link0"


class DamageableFixedBaseRobot(DamageableRobotMixin, FixedBaseRobot):
    pass

class DamageableMobileRobot(DamageableRobotMixin, MobileRobot):
    pass

class DamageableWheeledRobot(DamageableRobotMixin, WheeledRobot):
    pass

class DamageableLeggedRobot(DamageableRobotMixin, LeggedRobot):
    pass


DAMAGEABLE_ROBOT_CLASS_MAPPING = {
    FixedBaseRobot: DamageableFixedBaseRobot,
    MobileRobot: DamageableMobileRobot,
    WheeledRobot: DamageableWheeledRobot,
    LeggedRobot: DamageableLeggedRobot,
}

DAMAGEABLE_ROBOT_TYPE_MAPPING = {
    "Baxter": DamageableFixedBaseRobot,
    "IIWA": DamageableFixedBaseRobot,
    "Jaco": DamageableFixedBaseRobot,
    "Kinova3": DamageableFixedBaseRobot,
    "Panda": DamageableFixedBaseRobot,
    "Sawyer": DamageableFixedBaseRobot,
    "UR5e": DamageableFixedBaseRobot,
    "PandaDexRH": DamageableFixedBaseRobot,
    "PandaDexLH": DamageableFixedBaseRobot,
    "PandaOmron": DamageableWheeledRobot,
    "Tiago": DamageableWheeledRobot,
    "SpotWithArm": DamageableLeggedRobot,
    "SpotWithArmFloating": DamageableLeggedRobot,
    "GR1": DamageableLeggedRobot,
    "GR1FixedLowerBody": DamageableLeggedRobot,
    "GR1ArmsOnly": DamageableLeggedRobot,
    "GR1FloatingBody": DamageableLeggedRobot,
}


def get_damageable_robot_class(robot_type: str):
    if robot_type in DAMAGEABLE_ROBOT_TYPE_MAPPING:
        return DAMAGEABLE_ROBOT_TYPE_MAPPING[robot_type]
    if robot_type in ROBOT_CLASS_MAPPING:
        base = ROBOT_CLASS_MAPPING[robot_type]
        if base in DAMAGEABLE_ROBOT_CLASS_MAPPING:
            return DAMAGEABLE_ROBOT_CLASS_MAPPING[base]
    print(f"Warning: unknown robot type '{robot_type}', using DamageableFixedBaseRobot")
    return DamageableFixedBaseRobot


# ═══════════════════════════════════════════════════════════════════════
# Fixture classes
# ═══════════════════════════════════════════════════════════════════════

class DamageableFixtureMixin(RSDamageableMixin, _RobocasaFixture):
    """Adds damage tracking to RoboCasa fixtures."""

    def __init__(self, *args, params=None, **kwargs):
        fixture_name = kwargs.get("name", "default")
        if params is None:
            from damagesim.robosuite.params import OBJECT_PARAMS
            if fixture_name in OBJECT_PARAMS:
                params = OBJECT_PARAMS[fixture_name].copy()
            elif "fixture_default" in OBJECT_PARAMS:
                params = OBJECT_PARAMS["fixture_default"].copy()
            else:
                params = OBJECT_PARAMS.get("default", {}).copy()
        super().__init__(*args, params=params, **kwargs)
        self.track_damage = True

    def _initialize_health(self):
        if self.sim is None:
            return
        if not self.damageable_parts:
            self.damageable_parts = self._get_fixture_body_names()
        super()._initialize_health()

    def _get_fixture_body_names(self) -> List[str]:
        if self.sim is None or not hasattr(self.sim, "model"):
            return []
        if self.damageable_parts:
            valid = []
            for bn in self.damageable_parts:
                try:
                    self.sim.model.body_name2id(bn)
                    valid.append(bn)
                except Exception:
                    pass
            return valid
        bodies = []
        if hasattr(self, "root_body"):
            try:
                self.sim.model.body_name2id(self.root_body)
                bodies.append(self.root_body)
                for i in range(self.sim.model.nbody):
                    bn = self.sim.model.body_id2name(i)
                    if bn and hasattr(self, "naming_prefix") and bn.startswith(self.naming_prefix):
                        if bn not in bodies:
                            bodies.append(bn)
            except Exception:
                pass
        if not bodies and hasattr(self, "name"):
            try:
                for i in range(self.sim.model.nbody):
                    bn = self.sim.model.body_id2name(i)
                    if bn and self.name in bn:
                        bodies.append(bn)
            except Exception:
                pass
        return bodies

    def set_damageable_bodies(self, bodies=None):
        if bodies is not None:
            self.damageable_parts = list(bodies)
        else:
            detected = self._get_fixture_body_names()
            if detected:
                self.damageable_parts = detected
            elif hasattr(self, "root_body") and self.sim is not None:
                try:
                    bid = self.sim.model.body_name2id(self.root_body)
                    self.damageable_parts = [self.sim.model.body_id2name(bid)]
                except Exception:
                    if not self.damageable_parts:
                        self.damageable_parts = []
            elif not self.damageable_parts:
                self.damageable_parts = []


def create_damageable_from_fixture(fixture):
    """
    Dynamically convert an existing fixture instance to a damageable one.
    """
    if isinstance(fixture, DamageableMixin):
        return fixture

    from damagesim.robosuite.params import OBJECT_PARAMS

    fixture_name = getattr(fixture, "name", getattr(fixture, "_name", "default"))
    fixture_class = type(fixture)

    damageable_cls = type(
        f"Damageable{fixture_class.__name__}",
        (DamageableFixtureMixin, fixture_class),
        {},
    )
    fixture.__class__ = damageable_cls

    if not hasattr(fixture, "_name"):
        fixture._name = fixture_name

    # Find best-matching params
    found = None
    if fixture_name in OBJECT_PARAMS:
        found = OBJECT_PARAMS[fixture_name].copy()
    if found is None:
        name_lower = fixture_name.lower()
        mappings = {
            "cab": "cabinet", "drawer": "drawer", "counter": "counter",
            "microwave": "microwave", "stove": "stove", "fridge": "fridge",
            "sink": "sink", "dishwasher": "dishwasher",
            "shelf": "cabinet", "shelves": "cabinet",
        }
        for key, target in mappings.items():
            if key in name_lower and target in OBJECT_PARAMS:
                found = OBJECT_PARAMS[target].copy()
                break
    if found is None:
        found = OBJECT_PARAMS.get("fixture_default", OBJECT_PARAMS.get("default", {})).copy()

    fixture.params = found
    fixture.damage_evaluators = []
    fixture.track_damage = True
    fixture.damageable_parts = []
    thresholds = found.get("health_thresholds", [90.0, 60.0, 30.0])
    fixture.minor_threshold, fixture.major_threshold, fixture.critical_threshold = thresholds
    fixture.part_healths = {}
    fixture._damage_statuses = {}
    fixture.damage_info = {}
    fixture.previous_health = 100.0
    if not hasattr(fixture, "sim"):
        fixture.sim = None
    if not hasattr(fixture, "control_freq"):
        fixture.control_freq = None

    return fixture


# ═══════════════════════════════════════════════════════════════════════
# Object classes
# ═══════════════════════════════════════════════════════════════════════

class DamageableMJCFObject(RSDamageableMixin, _MJCFObject):
    def __init__(self, name, mjcf_path, scale=1.0, solimp=None, solref=None,
                 density=100, friction=None, margin=None, rgba=None,
                 priority=None, params=None):
        if params is None:
            from damagesim.robosuite.params import get_params_for_object
            params = get_params_for_object(name)
        from damagesim.robosuite.params import OBJECT_PARAMS
        if solimp is None:
            solimp = params.get("solimp", OBJECT_PARAMS["default"]["solimp"])
        if solref is None:
            solref = params.get("solref", OBJECT_PARAMS["default"]["solref"])
        if friction is None:
            friction = params.get("friction", OBJECT_PARAMS["default"]["friction"])
        super().__init__(name=name, mjcf_path=mjcf_path, scale=scale,
                         solimp=solimp, solref=solref, density=density,
                         friction=friction, margin=margin, rgba=rgba,
                         priority=priority, params=params)


class DamageableXMLObject(RSDamageableMixin, _MujocoXMLObject):
    def __init__(self, name, fname, joints="default", obj_type="all",
                 duplicate_collision_geoms=True, params=None, rgba=None):
        super().__init__(fname=fname, name=name, joints=joints,
                         obj_type=obj_type,
                         duplicate_collision_geoms=duplicate_collision_geoms,
                         params=params)
        self.rgba = rgba or [0.5, 0.5, 0.5, 1.0]


class DamageableBox(RSDamageableMixin, _BoxObject):
    def __init__(self, name, size=None, rgba=None, density=None, friction=None,
                 solref=None, solimp=None, material=None, joints="default",
                 obj_type="all", duplicate_collision_geoms=True, params=None):
        super().__init__(params=params, name=name, size=size, rgba=rgba,
                         density=density, friction=friction, solref=solref,
                         solimp=solimp, material=material, joints=joints,
                         obj_type=obj_type,
                         duplicate_collision_geoms=duplicate_collision_geoms)


class DamageableBall(RSDamageableMixin, _BallObject):
    def __init__(self, name, size=None, rgba=None, density=None, friction=None,
                 solref=None, solimp=None, material=None, joints="default",
                 obj_type="all", duplicate_collision_geoms=True, params=None):
        super().__init__(params=params, name=name, size=size, rgba=rgba,
                         density=density, friction=friction, solref=solref,
                         solimp=solimp, material=material, joints=joints,
                         obj_type=obj_type,
                         duplicate_collision_geoms=duplicate_collision_geoms)


class DamageableCylinder(RSDamageableMixin, _CylinderObject):
    def __init__(self, name, size=None, rgba=None, density=None, friction=None,
                 solref=None, solimp=None, material=None, joints="default",
                 obj_type="all", duplicate_collision_geoms=True, params=None):
        super().__init__(params=params, name=name, size=size, rgba=rgba,
                         density=density, friction=friction, solref=solref,
                         solimp=solimp, material=material, joints=joints,
                         obj_type=obj_type,
                         duplicate_collision_geoms=duplicate_collision_geoms)


class DamageableCapsule(RSDamageableMixin, _CapsuleObject):
    def __init__(self, name, size=None, rgba=None, density=None, friction=None,
                 solref=None, solimp=None, material=None, joints="default",
                 obj_type="all", duplicate_collision_geoms=True, params=None):
        super().__init__(params=params, name=name, size=size, rgba=rgba,
                         density=density, friction=friction, solref=solref,
                         solimp=solimp, material=material, joints=joints,
                         obj_type=obj_type,
                         duplicate_collision_geoms=duplicate_collision_geoms)


# Object class mapping for the environment factory
DAMAGEABLE_OBJECT_MAPPING = {
    "BoxObject": DamageableBox,
    "BallObject": DamageableBall,
    "CylinderObject": DamageableCylinder,
    "CapsuleObject": DamageableCapsule,
    "MJCFObject": DamageableMJCFObject,
}

