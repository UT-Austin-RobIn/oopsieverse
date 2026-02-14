"""
Robosuite / MuJoCo damage evaluator implementations.
"""

from damagesim.robosuite.evaluators.mechanical import RSMechanicalDamageEvaluator

DAMAGE_EVALUATORS = {
    "mechanical": RSMechanicalDamageEvaluator,
}

