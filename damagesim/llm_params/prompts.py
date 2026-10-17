"""
The two prompts: pick similar objects, then generate the numbers. Prompt changes are
one-file diffs here; keep them short.
"""

from __future__ import annotations

from damagesim.llm_params.schema import PARAM_NAMES

SIMULATOR_NOTE = {
    "robocasa": "These numbers are for the RoboCasa (MuJoCo) simulator, in its own internal units. "
                "Stay on the scale of the examples.",
    "omnigibson": "These numbers are for the BEHAVIOR-1K (OmniGibson) simulator, in its own internal units. "
                  "Stay on the scale of the examples.",
}

# Per step: D = impact_damage_sensitivity * impact force + qs_damage_sensitivity * pushing force;
# health lost = max(0, D - damage_threshold) * damage_scale (damagesim/core/evaluators/mechanical.py).
PARAM_MEANINGS = {
    "damage_threshold": "How hard a hit or push has to be before any damage happens.",
    "damage_scale": "How fast health drops once the threshold is passed.",
    "impact_damage_sensitivity": "How much sudden impacts (drops, collisions) count.",
    "qs_damage_sensitivity": "How much steady pushing or squeezing counts.",
}


def selection_messages(backend: str, target_name: str, pool_names: list[str], k: int) -> list[dict]:
    listing = "\n".join(f"- {n}" for n in pool_names)
    return [
        {"role": "system", "content": "You help set damage parameters for objects in a robot simulator."},
        {"role": "user", "content": (
            f"Object: {target_name}\n\n"
            f"From the list below, pick the {k} objects that would break most like a {target_name} "
            f"(material, fragility, size). Use names from the list only.\n\n{listing}"
        )},
    ]


def _exemplar_line(row: dict) -> str:
    return f"{row['display_name']}: " + ", ".join(f"{p}={float(row[p])!r}" for p in PARAM_NAMES)


def generation_messages(backend: str, target_name: str, exemplars: list[dict]) -> list[dict]:
    meanings = "\n".join(f"- {p}: {PARAM_MEANINGS[p]}" for p in PARAM_NAMES)
    examples = "\n".join(_exemplar_line(r) for r in exemplars)
    return [
        {"role": "system", "content": "You set damage parameters for objects in a robot simulator. "
                                      "Answer with the JSON schema only; never ask a question."},
        {"role": "user", "content": (
            f"{SIMULATOR_NOTE[backend]}\n\nParameters:\n{meanings}\n\n"
            f"Hand-tuned values for similar objects:\n{examples}\n\n"
            f"Give the four parameters for: {target_name}\n"
            "Also give a one-sentence reasoning."
        )},
    ]
