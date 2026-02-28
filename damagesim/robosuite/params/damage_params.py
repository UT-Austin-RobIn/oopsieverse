"""
Robosuite / RoboCasa damage parameters.

Each entry maps an object name (or category) to a parameter dict
containing contact properties and damage tuning constants.
"""

from __future__ import annotations

from damagesim.robosuite.evaluators import DAMAGE_EVALUATORS  # noqa: F401


# ═══════════════════════════════════════════════════════════════════════
# Damageable object tracking config
# ═══════════════════════════════════════════════════════════════════════

DAMAGEABLE_OBJECTS = {
    "default": {
        "categories": ["agent"],
        "names": [],
    },
    "pick_egg": {
        "categories": ["agent"],
        "names": ["egg"],
    },
}


# ═══════════════════════════════════════════════════════════════════════
# Object / fixture parameters
# ═══════════════════════════════════════════════════════════════════════

OBJECT_PARAMS = {
    # ── Default ──
    "default": {
        "solimp": (0.95, 0.99, 0.001),
        "solref": (0.004, 1),
        "friction": (0.95, 0.3, 0.1),
        "health_thresholds": [90.0, 60.0, 30.0],
        "damage_evaluators": ["mechanical"],
        "mechanical": {
            "damage_threshold": 0.0,
            "damage_scale": 0.01,
            "impact_damage_sensitivity": 0.5,
            "qs_damage_sensitivity": 0.25,
        },
    },

    # ── Robots ──
    "PandaOmron": {
        "damage_evaluators": ["mechanical"],
        "damageable_bodies": [
            "robot0_link0", "robot0_link1", "robot0_link2",
            "robot0_link3", "robot0_link4", "robot0_link5",
            "robot0_link6", "robot0_link7", "robot0_right_hand",
            "gripper0_right_right_gripper", "gripper0_right_eef",
            "gripper0_right_leftfinger", "gripper0_right_finger_joint1_tip",
            "gripper0_right_rightfinger", "gripper0_right_finger_joint2_tip",
        ],
        "health_thresholds": [90.0, 60.0, 30.0],
        "mechanical": {
            "impact_damage_sensitivity": 0.005,
            "qs_damage_sensitivity": 0.1,
            "damage_threshold": 50.0,
            "damage_scale": 0.02,
            "link_config_overrides": {
                "gripper": {
                    "impact_damage_sensitivity": 0.01,
                    "qs_damage_sensitivity": 1.0,
                    "damage_threshold": 70.0,
                    "damage_scale": 0.2,
                },
                "link0": {
                    "impact_damage_sensitivity": 0.01,
                    "qs_damage_sensitivity": 1.0,
                    "damage_threshold": 100.0,
                    "damage_scale": 0.2,
                },
                "hand": {
                    "impact_damage_sensitivity": 0.01,
                    "qs_damage_sensitivity": 1.0,
                    "damage_threshold": 70.0,
                    "damage_scale": 0.2,
                },
                "link": {
                    "impact_damage_sensitivity": 0.01,
                    "qs_damage_sensitivity": 1.0,
                    "damage_threshold": 70.0,
                    "damage_scale": 0.2,
                },
            },
        },
    },

    # ── Objects ──
    "cup": {
        "solimp": (0.998, 0.998, 0.001),
        "solref": (0.001, 1),
        "friction": (0.95, 0.3, 0.1),
        "health_thresholds": [95.0, 70.0, 40.0],
        "damage_evaluators": ["mechanical"],
        "mechanical": {
            "damage_threshold": 50.0,
            "damage_scale": 1.0,
            "impact_damage_sensitivity": 1.0,
            "qs_damage_sensitivity": 0.5,
        },
    },
    "sponge": {
        "solimp": (0.9, 0.95, 0.003),
        "solref": (0.02, 1),
        "friction": (0.8, 0.3, 0.1),
        "health_thresholds": [80.0, 50.0, 25.0],
        "damage_evaluators": ["mechanical"],
        "mechanical": {
            "damage_threshold": 80.0,
            "damage_scale": 0.5,
            "impact_damage_sensitivity": 0.5,
            "qs_damage_sensitivity": 0.0,
        },
    },
    "apple": {
        "solimp": (0.95, 0.99, 0.001),
        "solref": (0.005, 1),
        "friction": (0.5, 0.3, 0.1),
        "health_thresholds": [85.0, 60.0, 30.0],
        "damage_evaluators": ["mechanical"],
        "mechanical": {
            "damage_threshold": 20.0,
            "damage_scale": 0.5,
            "impact_damage_sensitivity": 0.8,
            "qs_damage_sensitivity": 0.3,
        },
    },
    "can": {
        "solimp": (0.99, 0.995, 0.001),
        "solref": (0.002, 1),
        "friction": (0.6, 0.1, 0.1),
        "health_thresholds": [90.0, 70.0, 40.0],
        "damage_evaluators": ["mechanical"],
        "mechanical": {
            "damage_threshold": 15.0,
            "damage_scale": 0.8,
            "impact_damage_sensitivity": 0.6,
            "qs_damage_sensitivity": 0.9,
        },
    },
    "egg": {
        "solimp": (0.998, 0.998, 0.001),
        "solref": (0.001, 1),
        "friction": (0.95, 0.3, 0.1),
        "health_thresholds": [50.0, 25.0, 0.0],
        "damage_evaluators": ["mechanical"],
        "mechanical": {
            "damage_threshold": 100.0,
            "damage_scale": 1.0,
            "impact_damage_sensitivity": 1.0,
            "qs_damage_sensitivity": 1.0,
        },
    },
    "plate": {
        "solimp": (0.998, 0.998, 0.001),
        "solref": (0.001, 1),
        "friction": (0.7, 0.2, 0.1),
        "health_thresholds": [95.0, 75.0, 45.0],
        "damage_evaluators": ["mechanical"],
        "mechanical": {
            "damage_threshold": 40.0,
            "damage_scale": 0.8,
            "impact_damage_sensitivity": 0.9,
            "qs_damage_sensitivity": 0.6,
        },
    },
    "bowl": {
        "solimp": (0.998, 0.998, 0.001),
        "solref": (0.001, 1),
        "friction": (0.7, 0.2, 0.1),
        "health_thresholds": [95.0, 75.0, 45.0],
        "damage_evaluators": ["mechanical"],
        "mechanical": {
            "damage_threshold": 40.0,
            "damage_scale": 0.8,
            "impact_damage_sensitivity": 0.9,
            "qs_damage_sensitivity": 0.6,
        },
    },
    "mug": {
        "solimp": (0.998, 0.998, 0.001),
        "solref": (0.001, 1),
        "friction": (0.8, 0.3, 0.1),
        "health_thresholds": [95.0, 70.0, 40.0],
        "damage_evaluators": ["mechanical"],
        "mechanical": {
            "damage_threshold": 45.0,
            "damage_scale": 0.9,
            "impact_damage_sensitivity": 0.95,
            "qs_damage_sensitivity": 0.5,
        },
    },
    "coffee_cup": {
        "solimp": (0.998, 0.998, 0.001),
        "solref": (0.001, 1),
        "friction": (0.7, 0.2, 0.1),
        "health_thresholds": [95.0, 70.0, 40.0],
        "damage_evaluators": ["mechanical"],
        "mechanical": {
            "damage_threshold": 40.0,
            "damage_scale": 1.0,
            "impact_damage_sensitivity": 1.0,
            "qs_damage_sensitivity": 0.5,
        },
    },
    "banana": {
        "solimp": (0.92, 0.97, 0.002),
        "solref": (0.008, 1),
        "friction": (0.6, 0.3, 0.1),
        "health_thresholds": [80.0, 55.0, 25.0],
        "damage_evaluators": ["mechanical"],
        "mechanical": {
            "damage_threshold": 15.0,
            "damage_scale": 0.4,
            "impact_damage_sensitivity": 0.6,
            "qs_damage_sensitivity": 0.7,
        },
    },
    "orange": {
        "solimp": (0.94, 0.98, 0.002),
        "solref": (0.006, 1),
        "friction": (0.5, 0.3, 0.1),
        "health_thresholds": [85.0, 60.0, 30.0],
        "damage_evaluators": ["mechanical"],
        "mechanical": {
            "damage_threshold": 25.0,
            "damage_scale": 0.45,
            "impact_damage_sensitivity": 0.7,
            "qs_damage_sensitivity": 0.5,
        },
    },
    "bread": {
        "solimp": (0.85, 0.92, 0.005),
        "solref": (0.02, 1),
        "friction": (0.7, 0.4, 0.2),
        "health_thresholds": [70.0, 45.0, 20.0],
        "damage_evaluators": ["mechanical"],
        "mechanical": {
            "damage_threshold": 50.0,
            "damage_scale": 0.3,
            "impact_damage_sensitivity": 0.2,
            "qs_damage_sensitivity": 0.8,
        },
    },
    "cereal": {
        "solimp": (0.90, 0.95, 0.002),
        "solref": (0.005, 1),
        "friction": (0.6, 0.3, 0.1),
        "health_thresholds": [80.0, 50.0, 25.0],
        "damage_evaluators": ["mechanical"],
        "mechanical": {
            "damage_threshold": 30.0,
            "damage_scale": 0.4,
            "impact_damage_sensitivity": 0.3,
            "qs_damage_sensitivity": 0.9,
        },
    },
    "jam": {
        "solimp": (0.998, 0.998, 0.001),
        "solref": (0.001, 1),
        "friction": (0.6, 0.2, 0.1),
        "health_thresholds": [95.0, 70.0, 40.0],
        "damage_evaluators": ["mechanical"],
        "mechanical": {
            "damage_threshold": 50.0,
            "damage_scale": 1.2,
            "impact_damage_sensitivity": 1.0,
            "qs_damage_sensitivity": 0.3,
        },
    },
    "milk": {
        "solimp": (0.93, 0.97, 0.002),
        "solref": (0.005, 1),
        "friction": (0.5, 0.2, 0.1),
        "health_thresholds": [85.0, 55.0, 30.0],
        "damage_evaluators": ["mechanical"],
        "mechanical": {
            "damage_threshold": 40.0,
            "damage_scale": 0.35,
            "impact_damage_sensitivity": 0.4,
            "qs_damage_sensitivity": 0.6,
        },
    },
    "table_mat": {
        "solimp": (0.90, 0.95, 0.003),
        "solref": (0.01, 1),
        "friction": (0.9, 0.4, 0.2),
        "health_thresholds": [80.0, 50.0, 25.0],
        "damage_evaluators": ["mechanical"],
        "mechanical": {
            "damage_threshold": 80.0,
            "damage_scale": 0.2,
            "impact_damage_sensitivity": 0.2,
            "qs_damage_sensitivity": 0.3,
        },
    },
    "cake": {
        "solimp": (0.85, 0.92, 0.005),
        "solref": (0.015, 1),
        "friction": (0.6, 0.3, 0.1),
        "health_thresholds": [80.0, 50.0, 20.0],
        "damage_evaluators": ["mechanical"],
        "mechanical": {
            "damage_threshold": 15.0,
            "damage_scale": 0.8,
            "impact_damage_sensitivity": 0.9,
            "qs_damage_sensitivity": 0.7,
        },
    },
    "tray": {
        "solimp": (0.95, 0.98, 0.002),
        "solref": (0.004, 1),
        "friction": (0.7, 0.3, 0.1),
        "health_thresholds": [85.0, 60.0, 35.0],
        "damage_evaluators": ["mechanical"],
        "mechanical": {
            "damage_threshold": 60.0,
            "damage_scale": 0.4,
            "impact_damage_sensitivity": 0.5,
            "qs_damage_sensitivity": 0.3,
        },
    },
    "pastry": {
        "solimp": (0.85, 0.92, 0.005),
        "solref": (0.015, 1),
        "friction": (0.6, 0.3, 0.1),
        "health_thresholds": [75.0, 45.0, 20.0],
        "damage_evaluators": ["mechanical"],
        "mechanical": {
            "damage_threshold": 20.0,
            "damage_scale": 0.7,
            "impact_damage_sensitivity": 0.85,
            "qs_damage_sensitivity": 0.9,
        },
    },
    "wine": {
        "solimp": (0.998, 0.998, 0.001),
        "solref": (0.001, 1),
        "friction": (0.5, 0.2, 0.1),
        "health_thresholds": [95.0, 70.0, 40.0],
        "damage_evaluators": ["mechanical"],
        "mechanical": {
            "damage_threshold": 45.0,
            "damage_scale": 1.1,
            "impact_damage_sensitivity": 1.0,
            "qs_damage_sensitivity": 0.35,
        },
    },
    "liquor": {
        "solimp": (0.998, 0.998, 0.001),
        "solref": (0.001, 1),
        "friction": (0.5, 0.2, 0.1),
        "health_thresholds": [95.0, 70.0, 40.0],
        "damage_evaluators": ["mechanical"],
        "mechanical": {
            "damage_threshold": 50.0,
            "damage_scale": 1.0,
            "impact_damage_sensitivity": 0.95,
            "qs_damage_sensitivity": 0.3,
        },
    },

    # ── Fixtures ──
    "fixture_default": {
        "health_thresholds": [90.0, 60.0, 30.0],
        "damage_evaluators": ["mechanical"],
        "mechanical": {
            "damage_threshold": 100.0,
            "damage_scale": 0.03,
            "impact_damage_sensitivity": 0.3,
            "qs_damage_sensitivity": 0.2,
        },
    },
    "microwave": {
        "health_thresholds": [90.0, 60.0, 30.0],
        "damage_evaluators": ["mechanical"],
        "mechanical": {
            "damage_threshold": 80.0,
            "damage_scale": 0.05,
            "impact_damage_sensitivity": 0.4,
            "qs_damage_sensitivity": 0.3,
        },
    },
    "cabinet": {
        "health_thresholds": [90.0, 60.0, 30.0],
        "damage_evaluators": ["mechanical"],
        "mechanical": {
            "damage_threshold": 70.0,
            "damage_scale": 0.08,
            "impact_damage_sensitivity": 0.5,
            "qs_damage_sensitivity": 0.4,
        },
    },
    "drawer": {
        "health_thresholds": [90.0, 60.0, 30.0],
        "damage_evaluators": ["mechanical"],
        "mechanical": {
            "damage_threshold": 60.0,
            "damage_scale": 0.1,
            "impact_damage_sensitivity": 0.4,
            "qs_damage_sensitivity": 0.6,
        },
    },
    "counter": {
        "health_thresholds": [95.0, 80.0, 50.0],
        "damage_evaluators": ["mechanical"],
        "mechanical": {
            "damage_threshold": 150.0,
            "damage_scale": 0.02,
            "impact_damage_sensitivity": 0.2,
            "qs_damage_sensitivity": 0.1,
        },
    },
    "fridge": {
        "health_thresholds": [90.0, 60.0, 30.0],
        "damage_evaluators": ["mechanical"],
        "mechanical": {
            "damage_threshold": 100.0,
            "damage_scale": 0.04,
            "impact_damage_sensitivity": 0.3,
            "qs_damage_sensitivity": 0.3,
        },
    },
    "stove": {
        "health_thresholds": [90.0, 60.0, 30.0],
        "damage_evaluators": ["mechanical"],
        "mechanical": {
            "damage_threshold": 70.0,
            "damage_scale": 0.06,
            "impact_damage_sensitivity": 0.6,
            "qs_damage_sensitivity": 0.2,
        },
    },
    "sink": {
        "health_thresholds": [95.0, 70.0, 40.0],
        "damage_evaluators": ["mechanical"],
        "mechanical": {
            "damage_threshold": 120.0,
            "damage_scale": 0.03,
            "impact_damage_sensitivity": 0.25,
            "qs_damage_sensitivity": 0.15,
        },
    },
    "dishwasher": {
        "health_thresholds": [90.0, 60.0, 30.0],
        "damage_evaluators": ["mechanical"],
        "mechanical": {
            "damage_threshold": 80.0,
            "damage_scale": 0.05,
            "impact_damage_sensitivity": 0.4,
            "qs_damage_sensitivity": 0.35,
        },
    },
}


# ═══════════════════════════════════════════════════════════════════════
# Helper functions
# ═══════════════════════════════════════════════════════════════════════

def _copy_params(params: dict) -> dict:
    result = {}
    for k, v in params.items():
        if isinstance(v, dict):
            result[k] = v.copy()
        elif isinstance(v, (list, tuple)):
            result[k] = type(v)(v)
        else:
            result[k] = v
    return result


def get_params_for_object(obj_name: str, obj_type: str | None = None) -> dict:
    if obj_name in OBJECT_PARAMS:
        return _copy_params(OBJECT_PARAMS[obj_name])
    if obj_type and obj_type in OBJECT_PARAMS:
        return _copy_params(OBJECT_PARAMS[obj_type])
    return _copy_params(OBJECT_PARAMS["default"])


def get_contact_properties(obj_name_or_type: str) -> dict:
    params = OBJECT_PARAMS.get(obj_name_or_type, OBJECT_PARAMS["default"])
    return {
        "solimp": params.get("solimp", OBJECT_PARAMS["default"]["solimp"]),
        "solref": params.get("solref", OBJECT_PARAMS["default"]["solref"]),
        "friction": params.get("friction", OBJECT_PARAMS["default"]["friction"]),
    }


def get_damage_params_for_object(obj_name: str, obj_type: str | None = None) -> dict:
    params = get_params_for_object(obj_name, obj_type)
    return {k: v for k, v in params.items() if k not in ("solimp", "solref", "friction")}

