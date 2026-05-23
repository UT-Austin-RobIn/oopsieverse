# DamageSim

**DamageSim** is the damage-aware simulation layer used with **robosuite / RoboCasa** in this repository. It tracks contact-relevant properties and task-specific **damageable objects** so policies can be evaluated under realistic harm constraints—not only task success.

## Where to look in the codebase

| Area | Role |
|------|------|
| [`damagesim/robosuite/params/damage_params.py`](https://github.com/UT-Austin-RobIn/oopsieverse/blob/main/damagesim/robosuite/params/damage_params.py) | Per-task **damageable object** sets, contact parameters, and damage tuning keyed by environment / object names. |
| [`damagesim/robosuite/evaluators/`](https://github.com/UT-Austin-RobIn/oopsieverse/tree/main/damagesim/robosuite/evaluators) | Damage **evaluators** wired into simulation steps. |

!!! info "Relationship to OopsieBench"

    [`oopsiebench`](https://github.com/UT-Austin-RobIn/oopsieverse/tree/main/oopsiebench) environments pair standard task definitions with **damageable** subclasses that use this stack. See [OopsieBench](oopsiebench.md) for the task catalog.
