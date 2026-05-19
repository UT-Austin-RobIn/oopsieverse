"""
OmniGibson-specific damage parameters for objects and robots.

Each entry maps an OG object category to a parameter dict that controls
which evaluators are active and their tuning constants.
"""

from damagesim.omnigibson.evaluators import DAMAGE_EVALUATORS  # noqa: F401

DAMAGEABLE_OBJECTS = {
    "default": {
        "skip_categories": ["floors", "walls", "ceilings", "hall_tree", "roof", "countertop", "breakfast_table", 
                            "straight_chair", "ottoman", "dining_table", "fixed_window", "carpet", "shower_stall", "toilet",
                            "pedestal_sink", "furniture_sink", "towel_rack", "bed", "sofa", "coffee_table"],
    },
    "shelve_item": {
        "categories": ["bottle_of_beer", "bottle_of_wine", "wineglass", "bag_of_flour", "box_of_crackers", "agent"],
        "names": [],
    },
    "pour_water": {
        "categories": ["laptop", "coffee_cup"],
        "names": ["water_glass"],
    },
    "add_firewood": {
        "categories": ["agent"],
        "names": [],
    },
    "open_drawer": {
        "categories": ["agent"],
        "names": ["bottom_cabinet_bamfsz_1",],
    },
    "wipe_counter": {
        "categories": ["agent"],
        "names": [],
    },
    "place_plate": {
        "categories": ["agent", "plate"],
        "names": [],
    },
    "place_bowl": {
        "categories": ["agent", "bowl"],
        "names": [],
    },
    "pick_egg": {
        "categories": ["agent", "egg"],
        "names": [],
    },
    "nav_to_table": {
        "categories": ["agent", "swivel_chair", "vase"],
        "names": [],
    },
    "turn_on_faucet": {
        "categories": ["agent"],
        "names": [],
    },
}

PARAMS = {
    # ── Default ─────────────────────────────────────────────────────────
    "default": {
        "damage_evaluators": ["mechanical"],
        "mechanical": {
            "impact_damage_sensitivity": 1.0,
            "qs_damage_sensitivity": 1.0,
            "damage_threshold": 30.0,
            "damage_scale": 0.1,
        },
    },

    # ── Egg (pick_egg, fragile vs impacts) ────────────────────────────────
    "egg": {
        "damage_evaluators": ["mechanical"],
        "mechanical": {
            "impact_damage_sensitivity": 1.0,
            "qs_damage_sensitivity": 1.0,
            "damage_threshold": 10.0,
            "damage_scale": 0.5,
        },
    },

    # ── Robots (OG category = "agent") ──────────────────────────────────
    "agent": {
        # TODO: Add mechanical damage evaluator back, need to debug mech damage params
        "damage_evaluators": ["mechanical", "thermal", "electrical"],
        # dict_keys(['panda_link0', 'panda_link1', 'panda_link2', 'panda_link3', 'panda_link4', 'panda_link5', 'panda_link6', 'panda_link7', 'panda_hand', 'panda_leftfinger', 'panda_rightfinger', 'eef_link'])
        "damageablefrankapanda_damageable_links": [
            "panda_link0", "panda_link1", "panda_link2", "panda_link3", "panda_link4", "panda_link5", "panda_link6", "panda_link7", "panda_hand", "panda_leftfinger", "panda_rightfinger", "eef_link"
        ],
        "damageabletiago_damageable_links": [
            "base_link",
            "arm_right_1_link", "arm_right_2_link", "arm_right_3_link",
            "arm_right_4_link", "arm_right_5_link", "arm_right_6_link",
            "arm_right_7_link",
            "gripper_right_link",
            "gripper_right_left_finger_link", "gripper_right_right_finger_link",
        ],
        "damageabler1pro_damageable_links": [
            "base_link",
            "left_arm_link1", "left_arm_link2", "left_arm_link3",
            "left_arm_link4", "left_arm_link5", "left_arm_link6",
            "left_arm_link7",
            "left_gripper_link", "left_gripper_finger_link1",
            "left_gripper_finger_link2", "left_realsense_link",
            "right_arm_link1", "right_arm_link2", "right_arm_link3",
            "right_arm_link4", "right_arm_link5", "right_arm_link6",
            "right_arm_link7",
            "right_gripper_link", "right_gripper_finger_link1",
            "right_gripper_finger_link2", "right_realsense_link",
        ],
        "mechanical": {
            "impact_damage_sensitivity": 0.01,
            "qs_damage_sensitivity": 1.0,
            "damage_threshold": 70.0,
            "damage_scale": 0.2,
            "part_config_overrides": {
                "gripper": {
                    "impact_damage_sensitivity": 0.01,
                    "qs_damage_sensitivity": 1.0,
                    "damage_threshold": 100.0,
                    "damage_scale": 0.2,
                },
                "base": {
                    "impact_damage_sensitivity": 0.01,
                    "qs_damage_sensitivity": 1.0,
                    "damage_threshold": 200.0,
                    "damage_scale": 0.2,
                },
                "arm": {
                    "impact_damage_sensitivity": 0.01,
                    "qs_damage_sensitivity": 1.0,
                    "damage_threshold": 100.0,
                    "damage_scale": 0.2,
                },
            },
        },
        "thermal": {
            "heating_threshold": 45.0,
            "cooling_threshold": -20.0,
            "scale": 1.0,
        },
        "electrical": {
            "damage_threshold": 10.0,
            "scale": 10.0,
            "water_system_name": "water",
        },
    },

    # ── Used in OopsieVerse paper experiments ───────────────────────────────────────────────
    "microwave": {
        "damage_evaluators": ["mechanical"],
        "damageable_links": ["base_link", "link_0", "glass"],
        "mechanical": {
            "impact_damage_sensitivity": 1.0,
            "qs_damage_sensitivity": 1.0,
            "damage_threshold": 100.0,
            "damage_scale": 1.0,
        },
    },
    "camera_tripod": {
        "damage_evaluators": ["mechanical"],
        "mechanical": {
            "impact_damage_sensitivity": 0.1,
            "qs_damage_sensitivity": 1.0,
            "damage_threshold": 150.0,
            "damage_scale": 1.0,
        },
    },
    "digital_camera": {
        "damage_evaluators": ["mechanical"],
        "mechanical": {
            "impact_damage_sensitivity": 1.0,
            "qs_damage_sensitivity": 0.5,
            "damage_threshold": 60.0,
            "damage_scale": 100.0,
        },
    },
    "scrub_brush": {
        "damage_evaluators": ["mechanical"],
        "mechanical": {
            "impact_damage_sensitivity": 0.01,
            "qs_damage_sensitivity": 0.01,
            "damage_threshold": 300.0,
            "damage_scale": 100.0,
        },
    },
    "bottle_of_wine": {
        "damage_evaluators": ["mechanical"],
        "mechanical": {
            "impact_damage_sensitivity": 1.0,
            "qs_damage_sensitivity": 0.5,
            "damage_threshold": 50.0,
            "damage_scale": 100.0,
        },
    },
    "wineglass": {
        "damage_evaluators": ["mechanical"],
        "mechanical": {
            "impact_damage_sensitivity": 1.0,
            "qs_damage_sensitivity": 0.5,
            "damage_threshold": 15.0,
            "damage_scale": 100.0,
        },
    },
    "bottle_of_beer": {
        "damage_evaluators": ["mechanical"],
        "mechanical": {
            "impact_damage_sensitivity": 1.0,
            "qs_damage_sensitivity": 0.5,
            "damage_threshold": 15.0,
            "damage_scale": 100.0,
        },
    },
    "bag_of_flour": {
        "damage_evaluators": ["mechanical"],
        "mechanical": {
            "impact_damage_sensitivity": 0.1,
            "qs_damage_sensitivity": 0.1,
            "damage_threshold": 150.0,
            "damage_scale": 100.0,
        },
    },
    "box_of_crackers": {
        "damage_evaluators": ["mechanical"],
        "mechanical": {
            "impact_damage_sensitivity": 0.1,
            "qs_damage_sensitivity": 0.8,
            "damage_threshold": 200.0,
            "damage_scale": 1.0,
        },
    },
    "stand": {
        "damage_evaluators": ["mechanical"],
        "mechanical": {
            "impact_damage_sensitivity": 0.001,
            "qs_damage_sensitivity": 0.001,
            "damage_threshold": 500.0,
            "damage_scale": 1.0,
        },
    },
    "laptop": {
        "damage_evaluators": ["mechanical", "electrical"],
        "mechanical": {
            "impact_damage_sensitivity": 1.0,
            "qs_damage_sensitivity": 0.5,
            "damage_threshold": 80.0,
            "damage_scale": 100.0,
        },
        "electrical": {
            "damage_threshold": 20.0,
            "scale": 5.0,
            "water_system_name": "water",
            "link_thresholds": {
                "screen": {"damage_threshold": 50.0, "scale": 10.0},
                "keyboard": {"damage_threshold": 50.0, "scale": 8.0},
            },
        },
    },
    "water_glass": {
        "damage_evaluators": ["mechanical"],
        "mechanical": {
            "impact_damage_sensitivity": 1.0,
            "qs_damage_sensitivity": 0.5,
            "damage_threshold": 50.0,
            "damage_scale": 100.0,
        },
    },
    "coffee_cup": {
        "damage_evaluators": ["mechanical"],
        "mechanical": {
            "impact_damage_sensitivity": 1.0,
            "qs_damage_sensitivity": 0.5,
            "damage_threshold": 150.0,
            "damage_scale": 1.0,
        },
    },
    "plate": {
        "damage_evaluators": ["mechanical"],
        "mechanical": {
            "impact_damage_sensitivity": 1.0,
            "qs_damage_sensitivity": 0.5,
            "damage_threshold": 40.0,
            "damage_scale": 1.5,
        },
    },
    "bowl": {
        "damage_evaluators": ["mechanical"],
        "mechanical": {
            "impact_damage_sensitivity": 1.0,
            "qs_damage_sensitivity": 0.5,
            "damage_threshold": 50.0,
            "damage_scale": 1.0,
        },
    },
    "log": {
        "damage_evaluators": ["mechanical"],
        "mechanical": {
            "impact_damage_sensitivity": 0.5,
            "qs_damage_sensitivity": 0.5,
            "damage_threshold": 100.0,
            "damage_scale": 1.0,
        },
    },
    "wood_fireplace": {
        "damage_evaluators": ["mechanical"],
        "mechanical": {
            "impact_damage_sensitivity": 0.1,
            "qs_damage_sensitivity": 0.1,
            "damage_threshold": 200.0,
            "damage_scale": 1.0,
        },
    },

    # ── Large appliances (sturdy, high threshold) ──────────────────────
    "fridge": {
        "damage_evaluators": ["mechanical"],
        "mechanical": {
            "impact_damage_sensitivity": 0.1,
            "qs_damage_sensitivity": 0.1,
            "damage_threshold": 250.0,
            "damage_scale": 1.0,
        },
    },
    "oven": {
        "damage_evaluators": ["mechanical"],
        "mechanical": {
            "impact_damage_sensitivity": 0.1,
            "qs_damage_sensitivity": 0.5,
            "damage_threshold": 150.0,
            "damage_scale": 1.0,
        },
    },
    "dishwasher": {
        "damage_evaluators": ["mechanical"],
        "mechanical": {
            "impact_damage_sensitivity": 0.1,
            "qs_damage_sensitivity": 0.1,
            "damage_threshold": 200.0,
            "damage_scale": 1.0,
        },
    },

    # ── Heavy furniture (sturdy, high threshold) ─────────────────────
    "bed": {
        "damage_evaluators": ["mechanical"],
        "mechanical": {
            "impact_damage_sensitivity": 0.05,
            "qs_damage_sensitivity": 0.05,
            "damage_threshold": 500.0,
            "damage_scale": 1.0,
        },
    },
    "sofa": {
        "damage_evaluators": ["mechanical"],
        "mechanical": {
            "impact_damage_sensitivity": 0.05,
            "qs_damage_sensitivity": 0.05,
            "damage_threshold": 500.0,
            "damage_scale": 1.0,
        },
    },
    "bookcase": {
        "damage_evaluators": ["mechanical"],
        "mechanical": {
            "impact_damage_sensitivity": 0.1,
            "qs_damage_sensitivity": 0.1,
            "damage_threshold": 200.0,
            "damage_scale": 1.0,
        },
    },
    "top_cabinet": {
        "damage_evaluators": ["mechanical"],
        "mechanical": {
            "impact_damage_sensitivity": 0.1,
            "qs_damage_sensitivity": 1.0,
            "damage_threshold": 100.0,
            "damage_scale": 1.0,
        },
    },
    "bottom_cabinet": {
        "damage_evaluators": ["mechanical"],
        "mechanical": {
            "impact_damage_sensitivity": 0.1,
            "qs_damage_sensitivity": 0.1,
            "damage_threshold": 700.0,
            "damage_scale": 1.0,
            "part_config_overrides": {
                "link_{i}": {
                    "impact_damage_sensitivity": 0.05,
                    "qs_damage_sensitivity": 1.0,
                    "damage_threshold": 300.0,
                    "damage_scale": 1.0,
                },
            },
        },
    },
    "coffee_table": {
        "damage_evaluators": ["mechanical"],
        "mechanical": {
            "impact_damage_sensitivity": 0.1,
            "qs_damage_sensitivity": 0.1,
            "damage_threshold": 500.0,
            "damage_scale": 1.0,
        },
    },
    "swivel_chair": {
        "damage_evaluators": ["mechanical"],
        "mechanical": {
            "impact_damage_sensitivity": 0.1,
            "qs_damage_sensitivity": 0.1,
            "damage_threshold": 500.0,
            "damage_scale": 1.0,
        },
    },

    # ── Electronics (fragile, low threshold) ─────────────────────────
    "standing_tv": {
        "damage_evaluators": ["mechanical"],
        "mechanical": {
            "impact_damage_sensitivity": 1.0,
            "qs_damage_sensitivity": 0.5,
            "damage_threshold": 80.0,
            "damage_scale": 100.0,
        },
    },
    "loudspeaker": {
        "damage_evaluators": ["mechanical"],
        "mechanical": {
            "impact_damage_sensitivity": 1.0,
            "qs_damage_sensitivity": 0.5,
            "damage_threshold": 80.0,
            "damage_scale": 10.0,
        },
    },

    # ── Fragile / light objects (moderate-to-low threshold) ──────────
    "floor_lamp": {
        "damage_evaluators": ["mechanical"],
        "mechanical": {
            "impact_damage_sensitivity": 1.0,
            "qs_damage_sensitivity": 0.5,
            "damage_threshold": 80.0,
            "damage_scale": 10.0,
        },
    },
    "table_lamp": {
        "damage_evaluators": ["mechanical"],
        "mechanical": {
            "impact_damage_sensitivity": 1.0,
            "qs_damage_sensitivity": 0.5,
            "damage_threshold": 80.0,
            "damage_scale": 10.0,
        },
    },
    "pot_plant": {
        "damage_evaluators": ["mechanical"],
        "mechanical": {
            "impact_damage_sensitivity": 1.0,
            "qs_damage_sensitivity": 0.5,
            "damage_threshold": 60.0,
            "damage_scale": 10.0,
        },
    },
    "mirror": {
        "damage_evaluators": ["mechanical"],
        "mechanical": {
            "impact_damage_sensitivity": 1.0,
            "qs_damage_sensitivity": 0.5,
            "damage_threshold": 40.0,
            "damage_scale": 100.0,
        },
    },
    "picture": {
        "damage_evaluators": ["mechanical"],
        "mechanical": {
            "impact_damage_sensitivity": 0.5,
            "qs_damage_sensitivity": 0.3,
            "damage_threshold": 50.0,
            "damage_scale": 10.0,
        },
    },
    "public_trash_can": {
        "damage_evaluators": ["mechanical"],
        "mechanical": {
            "impact_damage_sensitivity": 0.1,
            "qs_damage_sensitivity": 0.1,
            "damage_threshold": 200.0,
            "damage_scale": 1.0,
        },
    },

    # ── Structural fixtures (very sturdy) ────────────────────────────
    "door": {
        "damage_evaluators": ["mechanical"],
        "mechanical": {
            "impact_damage_sensitivity": 0.05,
            "qs_damage_sensitivity": 0.05,
            "damage_threshold": 300.0,
            "damage_scale": 1.0,
        },
    },
    "openable_window": {
        "damage_evaluators": ["mechanical"],
        "mechanical": {
            "impact_damage_sensitivity": 0.5,
            "qs_damage_sensitivity": 0.3,
            "damage_threshold": 100.0,
            "damage_scale": 10.0,
        },
    },
    "electric_switch": {
        "damage_evaluators": ["mechanical"],
        "mechanical": {
            "impact_damage_sensitivity": 0.1,
            "qs_damage_sensitivity": 0.1,
            "damage_threshold": 200.0,
            "damage_scale": 1.0,
        },
    },
}

