"""
Core (simulator-agnostic) base classes for damage tracking.

Classes:
    DamageableMixin — mixin that adds health tracking to any object
    DamageableEnvironment — base environment wrapper for damage tracking
"""

from damagesim.core.damageable_mixin import DamageableMixin
from damagesim.core.damageable_env import DamageableEnvironment

