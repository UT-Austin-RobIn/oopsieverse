"""
Output types the model must answer with (pydantic). Field names are the exact
``MechanicalDamageEvaluator`` kwargs; unknown kwargs are silently dropped at runtime.
"""

from __future__ import annotations

from typing import Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field, create_model, model_validator

PARAM_NAMES: tuple[str, ...] = (
    "damage_threshold", "damage_scale", "impact_damage_sensitivity", "qs_damage_sensitivity",
)


class MechanicalParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    damage_threshold: float = Field(gt=0)
    damage_scale: float = Field(gt=0)
    impact_damage_sensitivity: float = Field(ge=0)
    qs_damage_sensitivity: float = Field(ge=0)


class Generation(MechanicalParams):
    reasoning: str  # one sentence; logged, never scored


def selection_model(names: Sequence[str], k: int) -> type[BaseModel]:
    """A model whose ``similar`` list may only hold ``names``, exactly min(k, len(names)) distinct ones."""
    want = min(k, len(names))

    class _Base(BaseModel):
        model_config = ConfigDict(extra="forbid")

        @model_validator(mode="after")
        def _count(self):
            if len(self.similar) != want or len(set(self.similar)) != want:
                raise ValueError(f"pick exactly {want} different names from the list")
            return self

    return create_model("Selection", __base__=_Base, similar=(list[Literal[tuple(names)]], ...))
