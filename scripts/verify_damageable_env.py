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

        # TODO: Arnav: currently the rest of the objects are not damageable
        # env.scene.objects = [ ..., <omnigibson.objects.dataset_object.DatasetObject object at 0x7f2f26fd4460>, 
        # <omnigibson.objects.dataset_object.DatasetObject object at 0x7f2f26fd4280>, 
        # <damagesim.omnigibson.damageable_mixin.DamageableTiago object at 0x7f207818bcd0>]
        # Fix this

        env = OGDamageableEnvironment(cfg)
    else:
        raise ValueError(f"Invalid simulation framework: {args.sim_framework}")

    breakpoint()
    env.reset()


if __name__ == "__main__":
    __main__()