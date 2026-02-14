"""
OmniGibson damage evaluator implementations.
"""

from damagesim.omnigibson.evaluators.mechanical import OGMechanicalDamageEvaluator
from damagesim.omnigibson.evaluators.thermal import OGThermalDamageEvaluator
from damagesim.omnigibson.evaluators.electrical import OGElectricalDamageEvaluator

DAMAGE_EVALUATORS = {
    "mechanical": OGMechanicalDamageEvaluator,
    "thermal": OGThermalDamageEvaluator,
    "electrical": OGElectricalDamageEvaluator,
}

