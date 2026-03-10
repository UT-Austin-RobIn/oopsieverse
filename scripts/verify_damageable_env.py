import argparse
import os
import yaml



def __main__():

    parser = argparse.ArgumentParser()
    parser.add_argument('--sim_framework', type=str, help='Simulation framework to use', choices=["robosuite", "omnigibson"])
    args = parser.parse_args()

    if args.sim_framework == "robosuite":
        from damagesim.robosuite.damageable_env import RSDamageableEnvironment
        env = RSDamageableEnvironment()
    elif args.sim_framework == "omnigibson":
        from damagesim.omnigibson.damageable_env import OGDamageableEnvironment
        import omnigibson as og
        
        # TODO: 1. Clean this up
        # TODO: 2. Check other ways of loading OG env (bddl and json file)
        # Load the pre-selected configuration and set the online_sampling flag
        config_filename = os.path.join(og.example_config_path, "tiago_primitives.yaml")
        cfg = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)
        # Overwrite any configs here
        cfg["scene"]["scene_model"] = "house_single_floor"
        cfg["scene"]["not_load_object_categories"] = ["ottoman"]
        cfg["scene"]["load_room_instances"] = ["kitchen_0", "dining_room_0", "entryway_0", "living_room_0"]
        TASK_OBJECTS = {
            "box_of_crackers": {
                "type": "DatasetObject",
                "name": "box_of_crackers",
                "category": "box_of_crackers",
                "model": "cmdigf",
                "position": [6.0, 0.2, 2.0],
                "orientation": [0.0, 0.0, 0.70710678, 0.70710678],
            }, 
            "bag_of_flour": {
                "type": "DatasetObject",
                "name": "bag_of_flour",
                "category": "bag_of_flour",
                "model": "rlejxx",
                "position": [6.00, 0.35, 1.35],
                "orientation": [0.0, 0.0, 0.0, 1.0],
                "scale": [1.0, 1.0, 0.9],
            },
            "bottle_of_wine": {
                "type": "DatasetObject",
                "name": "bottle_of_wine",
                "category": "bottle_of_wine",
                "model": "hnkiog",
                "position": [6.00, 0.2, 1.35],
                "orientation": [0.0, 0.0, 0.0, 1.0],
                "scale": [1.0, 1.0, 1.0],
            },
            "apple": {
                "type": "DatasetObject",
                "name": "apple",
                "category": "apple",
                "model": "agveuv",
                "position": [6.00, 0.12, 1.35],
                "orientation": [0.0, 0.0, 0.0, 1.0],
                "scale": [1.0, 1.0, 1.0],
            },
        }
        cfg["objects"] = [TASK_OBJECTS[obj] for obj in TASK_OBJECTS]
        cfg["task"] = {
            "activity_name": "shelve_item",
        }

        env = OGDamageableEnvironment(cfg)

        # Verify scene loading
        print(env.scene.objects)
        breakpoint()
    else:
        raise ValueError(f"Invalid simulation framework: {args.sim_framework}")

    env.reset()

    if args.sim_framework == "robosuite":
        test_passed = True
    elif args.sim_framework == "omnigibson":
        robot_name = env.scene.robots[0].name
        objects = ["box_of_crackers", "bag_of_flour", robot_name]
        for obj_name in objects:
            obj = env.scene.object_registry("name", obj_name)
            assert obj is not None, f"Object {obj_name} not found"
            assert obj.track_damage, f"Object {obj_name} is not trackable"
            assert obj.damageable_links, f"Object {obj_name} has no damageable links"
            assert obj.damage_params, f"Object {obj_name} has no damage parameters"
            assert obj.damage_evaluators, f"Object {obj_name} has no damage evaluators"
            for link_name in obj.damageable_links:
                assert link_name in obj.link_healths, f"Object {obj_name} has no link health for {link_name}"
                assert obj.link_healths[link_name] == 100.0, f"Object {obj_name} link {link_name} health is not 100.0"
        print("Test passed for Omnigibson")
        breakpoint()
        og.shutdown()
    else:
        raise ValueError(f"Invalid simulation framework: {args.sim_framework}")

if __name__ == "__main__":
    __main__()