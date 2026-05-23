import numpy as np
from typing import Optional

def apply_gripper_finger_geom_friction(
    model,
    slide: float,
    spin: Optional[float] = None,
    roll: Optional[float] = None,
) -> int:
    """Raise sliding friction on gripper / finger collision geoms (MuJoCo ``mjModel``).
    Try 2.0-10.0 for PandaOmron; spin/roll are scaled from slide.
    Use after each ``env.reset()`` so values survive episode
    restarts but are reapplied if the model is reloaded from XML.

    Args:
        model: ``mjModel`` (or robosuite's wrapped model).
        slide: Sliding friction (first component of ``geom_friction``).
        spin: Torsional friction; if ``None``, derived from ``slide``.
        roll: Rolling friction; if ``None``, derived from ``slide``.

    Returns:
        Number of geoms updated.
    """
    if slide <= 0:
        raise ValueError("slide friction must be positive")

    if spin is None:
        spin = float(np.clip(slide * 0.04, 0.05, 0.25))
    if roll is None:
        roll = float(np.clip(slide * 0.004, 0.005, 0.03))

    friction = np.array([slide, spin, roll], dtype=np.float64)
    modified = 0

    for gid in range(model.ngeom):
        name = model.geom_id2name(gid)
        if not name:
            continue
        lower = name.lower()
        if "wheel" in lower:
            continue    
        if "finger" not in lower and "gripper" not in lower:
            continue
        if model.geom_contype[gid] == 0 and model.geom_conaffinity[gid] == 0:
            continue

        model.geom_friction[gid] = friction
        modified += 1

    if modified == 0:
        print(
            "Warning: no gripper/finger "
            "collision geoms matched; friction unchanged."
        )
    else:
        print(
            f"Gripper friction: set {modified} geom(s) to slide={slide} "
            "(spin/roll auto-scaled)."
        )

    return modified

