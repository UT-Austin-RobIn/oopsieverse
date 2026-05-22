"""
OmniGibson-side damage color manager.

Continuously blends each damageable object's diffuse tint toward red as its
health decreases. Mirrors the API and visual effect of
``DamageColorManager`` in ``scripts/teleop_robocasa.py``, but adapted to
OmniGibson by driving ``MaterialPrim.diffuse_tint`` (a multiplicative albedo
modifier) instead of MuJoCo's ``geom_rgba`` array.

Per-material albedo math (from OmniGibson):

    albedo = diffuse_tint * (albedo + albedo_add)

so ``diffuse_tint = [1, 1, 1]`` is the identity (texture unchanged) and
``diffuse_tint = [1, 0, 0]`` keeps only the red channel of the texture.
The blend is::

    tint = (1 - alpha) * original_tint + alpha * [1, 0, 0]
    alpha = 1 - health / 100

For materials whose backend does not actually support ``diffuse_tint`` (the
base ``MaterialPrim`` stub), the setter is a no-op, so iterating over every
material on the object is safe.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

try:
    import torch as th
except ImportError:
    th = None


_IDENTITY_TINT = [1.0, 1.0, 1.0]
_DAMAGE_TINT_TARGET = [1.0, 0.0, 0.0]


def _as_tint_tensor(value):
    """Convert *value* to a length-3 float tensor, falling back to identity."""
    if th is None:
        return None
    if value is None:
        return th.tensor(_IDENTITY_TINT, dtype=th.float32)
    if isinstance(value, th.Tensor):
        if value.numel() != 3:
            return th.tensor(_IDENTITY_TINT, dtype=th.float32)
        return value.detach().to(dtype=th.float32).clone()
    try:
        tensor = th.as_tensor(value, dtype=th.float32)
    except Exception:
        return th.tensor(_IDENTITY_TINT, dtype=th.float32)
    if tensor.numel() != 3:
        return th.tensor(_IDENTITY_TINT, dtype=th.float32)
    return tensor.clone()


class OGDamageColorManager:
    """Manage continuous diffuse-tint blending toward red for damageable objects.

    Args:
        env: An ``OGDamageableEnvironment`` (anything exposing
            ``get_damageable_objects()`` works).
    """

    def __init__(self, env):
        self.env = env
        # obj_name -> list of (material, original_tint_tensor)
        self._original_tints: Dict[str, List[Tuple[object, "th.Tensor"]]] = {}
        self._initialized: bool = False
        self._red_target = (
            th.tensor(_DAMAGE_TINT_TARGET, dtype=th.float32) if th is not None else None
        )

    def initialize_colors(self) -> None:
        """Snapshot the current ``diffuse_tint`` of every material on every
        damageable object so it can be restored later.

        Idempotent — repeated calls are no-ops until ``restore()`` is called.
        """
        if self._initialized or th is None:
            self._initialized = True
            return
        for obj in self._iter_damageable_objects():
            obj_name = getattr(obj, "name", None)
            if not obj_name:
                continue
            entries: List[Tuple[object, "th.Tensor"]] = []
            for material in self._iter_materials(obj):
                try:
                    current = material.diffuse_tint
                except Exception:
                    continue
                original = _as_tint_tensor(current)
                if original is None:
                    continue
                entries.append((material, original))
            if entries:
                self._original_tints[obj_name] = entries
        self._initialized = True

    def update(self, health_states: Dict[str, float]) -> None:
        """Blend each tracked object's diffuse tint toward red based on its health.

        Args:
            health_states: ``{obj_name: health_pct in [0, 100]}``.
        """
        if th is None:
            return
        if not self._initialized:
            self.initialize_colors()
        for obj_name, health_pct in health_states.items():
            entries = self._original_tints.get(obj_name)
            if not entries:
                continue
            clamped = max(0.0, min(100.0, float(health_pct)))
            alpha = 1.0 - clamped / 100.0
            for material, original in entries:
                blended = (1.0 - alpha) * original + alpha * self._red_target
                try:
                    material.diffuse_tint = blended
                except Exception:
                    pass

    def restore(self) -> None:
        """Restore every captured material back to its original tint."""
        for entries in self._original_tints.values():
            for material, original in entries:
                try:
                    material.diffuse_tint = original
                except Exception:
                    pass
        self._original_tints.clear()
        self._initialized = False

    # Match RoboCasa's API name for parity.
    reset = restore

    def _iter_damageable_objects(self):
        try:
            return list(self.env.get_damageable_objects())
        except Exception:
            return []

    @staticmethod
    def _iter_materials(obj):
        try:
            materials = obj.materials
        except Exception:
            return []
        if materials is None:
            return []
        return list(materials)
