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
    "pastry_display": {
        "categories": ["agent"],
        "names": ["pastry", "plate"],
    },
    "open_single_door": {
        "categories": ["agent", "microwave"],
        "names": [],
    },
    "turn_on_faucet": {
        "categories": ["agent", "sink"],
        "names": [],
    },
    "turn_on_microwave": {
        "categories": ["agent", "microwave"],
        "names": [],
    },
    "turn_on_stove": {
        "categories": ["agent", "stove"],
        "names": [],
    },
    "open_drawer": {
        "categories": ["agent", "drawer"],
        "names": [],
    },
    "place_plate": {
        "categories": ["agent", "sink", "counter"],
        "names": ["plate"],
    },
    "counter_to_microwave": {
        "categories": ["agent", "microwave"],
        "names": ["coffee_cup"],
    },
    "prepare_coffee": {
        "categories": ["agent", "coffee_machine"],
        "names": ["mug", "distr_object"],
    },
    "shelve_item": {
        "categories": ["agent"],
        "names": ["table_mat", "cereal", "wine_1", "wine_glass", "wine_2", "flour_bag"],
    },
    "prepare_breakfast": {
        "categories": ["agent"],
        "names": ["tray", "mug", "egg"],
    },
    "dirty_dishes": {
        "categories": ["agent", "sink", "counter"],
        "names": ["bowl", "cup", "plate"],
    },
    "nav_to_counter": {
        "categories": ["agent"],
        "names": ["stool_obstacle", "bowl"],
    },
    "wipe_counter": {
        "categories": ["agent", "sink", "counter"],
        "names": ["sponge"],
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
        "damageable_links": [
            "robot0_link0", "robot0_link1", "robot0_link2",
            "robot0_link3", "robot0_link4", "robot0_link5",
            "robot0_link6", "robot0_link7", "robot0_right_hand",
            "gripper0_right_right_gripper", "gripper0_right_eef",
            "gripper0_right_leftfinger", "gripper0_right_finger_joint1_tip",
            "gripper0_right_rightfinger", "gripper0_right_finger_joint2_tip",
        ],
        "mechanical": {
            "impact_damage_sensitivity": 0.0,
            "qs_damage_sensitivity": 0.0,
            "damage_threshold": 50.0,
            "damage_scale": 0.02,
            "link_config_overrides": {
                "gripper": {
                    "impact_damage_sensitivity": 0.0,
                    "qs_damage_sensitivity": 1.0,
                    "damage_threshold": 70.0,
                    "damage_scale": 0.2,
                },
                "link0": {
                    "impact_damage_sensitivity": 0.0,
                    "qs_damage_sensitivity": 1.0,
                    "damage_threshold": 100.0,
                    "damage_scale": 0.2,
                },
                "hand": {
                    "impact_damage_sensitivity": 0.0,
                    "qs_damage_sensitivity": 1.0,
                    "damage_threshold": 70.0,
                    "damage_scale": 0.2,
                },
                "link": {
                    "impact_damage_sensitivity": 0.0,
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
        "damage_evaluators": ["mechanical"],
        "mechanical": {
            "damage_threshold": 10.0,
            "damage_scale": 1.0,
            "impact_damage_sensitivity": 1.0,
            "qs_damage_sensitivity": 0.5,
        },
    },
    "sponge": {
        "solimp": (0.9, 0.95, 0.003),
        "solref": (0.02, 1),
        "friction": (0.8, 0.3, 0.1),
        "damage_evaluators": ["mechanical"],
        "mechanical": {
            "damage_threshold": 300.0,
            "damage_scale": 0.05,
            "impact_damage_sensitivity": 0.0,
            "qs_damage_sensitivity": 0.0,
        },
    },
    "apple": {
        "solimp": (0.95, 0.99, 0.001),
        "solref": (0.005, 1),
        "friction": (0.5, 0.3, 0.1),
        "damage_evaluators": ["mechanical"],
        "mechanical": {
            "damage_threshold": 34.0,
            "damage_scale": 0.5,
            "impact_damage_sensitivity": 1.0,
            "qs_damage_sensitivity": 0.5,
        },
    },
    "can": {
        "solimp": (0.99, 0.995, 0.001),
        "solref": (0.002, 1),
        "friction": (0.6, 0.1, 0.1),
        "damage_evaluators": ["mechanical"],
        "mechanical": {
            "damage_threshold": 80.0,
            "damage_scale": 0.4,
            "impact_damage_sensitivity": 0.5,
            "qs_damage_sensitivity": 1.0,
        },
    },
    "egg": {
        "solimp": (0.998, 0.998, 0.001),
        "solref": (0.001, 1),
        "friction": (0.95, 0.3, 0.1),
        "damage_evaluators": ["mechanical"],
        "mechanical": {
            "damage_threshold": 5.0,
            "damage_scale": 1.5,
            "impact_damage_sensitivity": 1.0,
            "qs_damage_sensitivity": 1.0,
        },
    },
    "plate": {
        "solimp": (0.998, 0.998, 0.001),
        "solref": (0.001, 1),
        "friction": (0.7, 0.2, 0.1),
        "damage_evaluators": ["mechanical"],
        "mechanical": {
            "damage_threshold": 12.0,
            "damage_scale": 0.8,
            "impact_damage_sensitivity": 1.0,
            "qs_damage_sensitivity": 0.5,
        },
    },
    "bowl": {
        "solimp": (0.998, 0.998, 0.001),
        "solref": (0.001, 1),
        "friction": (0.7, 0.2, 0.1),
        "damage_evaluators": ["mechanical"],
        "mechanical": {
            "damage_threshold": 12.0,
            "damage_scale": 0.8,
            "impact_damage_sensitivity": 1.0,
            "qs_damage_sensitivity": 0.5,
        },
    },
    "mug": {
        "solimp": (0.998, 0.998, 0.001),
        "solref": (0.001, 1),
        "friction": (0.8, 0.3, 0.1),
        "damage_evaluators": ["mechanical"],
        "mechanical": {
            "damage_threshold": 12.0,
            "damage_scale": 0.9,
            "impact_damage_sensitivity": 1.0,
            "qs_damage_sensitivity": 0.5,
        },
    },
    "coffee_cup": {
        "solimp": (0.998, 0.998, 0.001),
        "solref": (0.001, 1),
        "friction": (0.7, 0.2, 0.1),
        "damage_evaluators": ["mechanical"],
        "mechanical": {
            "damage_threshold": 10.0,
            "damage_scale": 1.0,
            "impact_damage_sensitivity": 1.0,
            "qs_damage_sensitivity": 0.5,
        },
    },
    "banana": {
        "solimp": (0.92, 0.97, 0.002),
        "solref": (0.008, 1),
        "friction": (0.6, 0.3, 0.1),
        "damage_evaluators": ["mechanical"],
        "mechanical": {
            "damage_threshold": 12.0,
            "damage_scale": 0.4,
            "impact_damage_sensitivity": 0.5,
            "qs_damage_sensitivity": 0.5,
        },
    },
    "orange": {
        "solimp": (0.94, 0.98, 0.002),
        "solref": (0.006, 1),
        "friction": (0.5, 0.3, 0.1),
        "damage_evaluators": ["mechanical"],
        "mechanical": {
            "damage_threshold": 25.0,
            "damage_scale": 0.45,
            "impact_damage_sensitivity": 0.5,
            "qs_damage_sensitivity": 0.5,
        },
    },
    "bread": {
        "solimp": (0.85, 0.92, 0.005),
        "solref": (0.02, 1),
        "friction": (0.7, 0.4, 0.2),
        "damage_evaluators": ["mechanical"],
        "mechanical": {
            "damage_threshold": 20.0,
            "damage_scale": 0.3,
            "impact_damage_sensitivity": 0.0,
            "qs_damage_sensitivity": 1.0,
        },
    },
    "cereal": {
        "solimp": (0.90, 0.95, 0.002),
        "solref": (0.005, 1),
        "friction": (0.6, 0.3, 0.1),
        "damage_evaluators": ["mechanical"],
        "mechanical": {
            "damage_threshold": 125.0,
            "damage_scale": 0.3,
            "impact_damage_sensitivity": 0.5,
            "qs_damage_sensitivity": 1.0,
        },
    },
    "jam": {
        "solimp": (0.998, 0.998, 0.001),
        "solref": (0.001, 1),
        "friction": (0.6, 0.2, 0.1),
        "damage_evaluators": ["mechanical"],
        "mechanical": {
            "damage_threshold": 20.0,
            "damage_scale": 1.2,
            "impact_damage_sensitivity": 1.0,
            "qs_damage_sensitivity": 0.5,
        },
    },
    "milk": {
        "solimp": (0.93, 0.97, 0.002),
        "solref": (0.005, 1),
        "friction": (0.5, 0.2, 0.1),
        "damage_evaluators": ["mechanical"],
        "mechanical": {
            "damage_threshold": 60.0,
            "damage_scale": 0.35,
            "impact_damage_sensitivity": 0.5,
            "qs_damage_sensitivity": 0.5,
        },
    },
    "table_mat": {
        "solimp": (0.90, 0.95, 0.003),
        "solref": (0.01, 1),
        "friction": (0.9, 0.4, 0.2),
        "damage_evaluators": ["mechanical"],
        "mechanical": {
            "damage_threshold": 200.0,
            "damage_scale": 0.1,
            "impact_damage_sensitivity": 0.0,
            "qs_damage_sensitivity": 0.5,
        },
    },
    "cake": {
        "solimp": (0.85, 0.92, 0.005),
        "solref": (0.015, 1),
        "friction": (0.6, 0.3, 0.1),
        "damage_evaluators": ["mechanical"],
        "mechanical": {
            "damage_threshold": 9.0,
            "damage_scale": 0.8,
            "impact_damage_sensitivity": 1.0,
            "qs_damage_sensitivity": 0.5,
        },
    },
    "tray": {
        "solimp": (0.95, 0.98, 0.002),
        "solref": (0.004, 1),
        "friction": (0.7, 0.3, 0.1),
        "damage_evaluators": ["mechanical"],
        "mechanical": {
            "damage_threshold": 125.0,
            "damage_scale": 0.3,
            "impact_damage_sensitivity": 0.5,
            "qs_damage_sensitivity": 0.5,
        },
    },
    "pastry": {
        "solimp": (0.85, 0.92, 0.005),
        "solref": (0.015, 1),
        "friction": (0.6, 0.3, 0.1),
        "damage_evaluators": ["mechanical"],
        "mechanical": {
            "damage_threshold": 17.0,
            "damage_scale": 0.7,
            "impact_damage_sensitivity": 1.0,
            "qs_damage_sensitivity": 1.0,
        },
    },
    "wine_glass": {
        "solimp": (0.998, 0.998, 0.001),
        "solref": (0.001, 1),
        "friction": (0.4, 0.2, 0.1),
        "damage_evaluators": ["mechanical"],
        "mechanical": {
            "damage_threshold": 6.0,
            "damage_scale": 1.5,
            "impact_damage_sensitivity": 1.0,
            "qs_damage_sensitivity": 0.5,
        },
    },
    "flour_bag": {
        "solimp": (0.88, 0.94, 0.003),
        "solref": (0.01, 1),
        "friction": (0.7, 0.4, 0.2),
        "damage_evaluators": ["mechanical"],
        "mechanical": {
            "damage_threshold": 205.0,
            "damage_scale": 0.2,
            "impact_damage_sensitivity": 0.5,
            "qs_damage_sensitivity": 1.0,
        },
    },
    "wine": {
        "solimp": (0.998, 0.998, 0.001),
        "solref": (0.001, 1),
        "friction": (0.5, 0.2, 0.1),
        "damage_evaluators": ["mechanical"],
        "mechanical": {
            "damage_threshold": 20.0,
            "damage_scale": 1.1,
            "impact_damage_sensitivity": 1.0,
            "qs_damage_sensitivity": 0.5,
        },
    },
    "liquor": {
        "solimp": (0.998, 0.998, 0.001),
        "solref": (0.001, 1),
        "friction": (0.5, 0.2, 0.1),
        "damage_evaluators": ["mechanical"],
        "mechanical": {
            "damage_threshold": 25.0,
            "damage_scale": 1.0,
            "impact_damage_sensitivity": 1.0,
            "qs_damage_sensitivity": 0.5,
        },
    },

    # ── Fixtures ──
    "fixture_default": {
        "damage_evaluators": ["mechanical"],
        "mechanical": {
            "damage_threshold": 100.0,
            "damage_scale": 0.03,
            "impact_damage_sensitivity": 0.5,
            "qs_damage_sensitivity": 0.0,
        },
    },
    "microwave": {
        "damage_evaluators": ["mechanical"],
        "mechanical": {
            "damage_threshold": 143.0,
            "damage_scale": 0.05,
            "impact_damage_sensitivity": 0.5,
            "qs_damage_sensitivity": 0.5,
        },
    },
    "cabinet": {
        "damage_evaluators": ["mechanical"],
        "mechanical": {
            "damage_threshold": 78.0,
            "damage_scale": 0.08,
            "impact_damage_sensitivity": 0.5,
            "qs_damage_sensitivity": 0.5,
        },
    },
    "drawer": {
        "damage_evaluators": ["mechanical"],
        "mechanical": {
            "damage_threshold": 60.0,
            "damage_scale": 0.1,
            "impact_damage_sensitivity": 0.5,
            "qs_damage_sensitivity": 0.5,
        },
    },
    "coffee_machine": {
        "damage_evaluators": ["mechanical"],
        "mechanical": {
            "damage_threshold": 88.0,
            "damage_scale": 0.06,
            "impact_damage_sensitivity": 0.5,
            "qs_damage_sensitivity": 0.5,
        },
    },
    "counter": {
        "damage_evaluators": ["mechanical"],
        "mechanical": {
            "damage_threshold": 200.0,
            "damage_scale": 0.02,
            "impact_damage_sensitivity": 0.0,
            "qs_damage_sensitivity": 0.0,
        },
    },
    "fridge": {
        "damage_evaluators": ["mechanical"],
        "mechanical": {
            "damage_threshold": 167.0,
            "damage_scale": 0.04,
            "impact_damage_sensitivity": 0.5,
            "qs_damage_sensitivity": 0.5,
        },
    },
    "stove": {
        "damage_evaluators": ["mechanical"],
        "mechanical": {
            "damage_threshold": 44.0,
            "damage_scale": 0.06,
            "impact_damage_sensitivity": 0.5,
            "qs_damage_sensitivity": 0.0,
        },
    },
    "sink": {
        "damage_evaluators": ["mechanical"],
        "mechanical": {
            "damage_threshold": 75.0,
            "damage_scale": 0.03,
            "impact_damage_sensitivity": 0.25,
            "qs_damage_sensitivity": 0.0,
        },
    },
    "dishwasher": {
        "damage_evaluators": ["mechanical"],
        "mechanical": {
            "damage_threshold": 107.0,
            "damage_scale": 0.05,
            "impact_damage_sensitivity": 0.5,
            "qs_damage_sensitivity": 0.5,
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
    # Strip trailing _<digits> so instance names ("wine_1") match category keys ("wine").
    base, _, suffix = obj_name.rpartition("_")
    if base and suffix.isdigit() and base in OBJECT_PARAMS:
        return _copy_params(OBJECT_PARAMS[base])
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
