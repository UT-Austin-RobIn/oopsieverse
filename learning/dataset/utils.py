import h5py
import numpy as np
import os
import shutil

def remove_idle_actions_from_dataset(dataset_path: str, output_path: str, num_idle_actions: int = None):
    """
    Remove timesteps where the first several actions are all zeros (idle) from the dataset.
    
    Args:
        dataset_path: Path to input HDF5 dataset
        output_path: Path to output HDF5 dataset with idle timesteps removed
        num_idle_actions: Number of initial actions to check for zeros. If None, checks all actions in H dimension.
    """
    with h5py.File(dataset_path, "r") as f:
        with h5py.File(output_path, "w") as f_out:
            demo_gro_out = f_out.create_group("data")
            for demo_key in f["data"].keys():
                demo_grp = f["data"][demo_key]
                print(demo_grp.keys())
                actions = demo_grp["action"][:].astype(np.float32)
                non_idle_indices = np.where(np.sum(np.abs(actions[:, :-1]), axis=1) != 0)[0]
                if non_idle_indices.shape[0] == 0:
                    print(f"{demo_key}: No non-idle timesteps found")
                    continue
                if actions.shape[0] < 100:
                    print(f"{demo_key}: Skipping because it has less than 100 timesteps")
                    continue
                demo_grp_out = demo_gro_out.create_group(demo_key)
                demo_grp_out.create_dataset("action", data=actions[non_idle_indices])
                demo_grp_out.create_dataset("reward", data=demo_grp["reward"][non_idle_indices])
                demo_grp_out.create_group("obs")

                #### Processing info ####
                demo_grp_out.create_group("info")
                for info_key in demo_grp["info"].keys():
                    demo_grp_out["info"].create_dataset(info_key, data=demo_grp["info"][info_key][non_idle_indices])

                #### Processing observations ####
                obs_keys = demo_grp["obs"].keys()
                for obs_key in obs_keys:
                    demo_grp_out['obs'].create_dataset(obs_key, data=demo_grp['obs'][obs_key][non_idle_indices])
                print(f"{demo_key}: Removing {np.sum(np.sum(np.abs(actions[:, :-1]), axis=1) == 0)} idle timesteps out of {actions.shape[0]} total timesteps")
                # timesteped_keys = ["dones", "rewards", "obs"]
                # for key in timesteped_keys:
                #     if key in demo_grp:
                #         demo_grp_out.create_dataset(key, data=demo_grp[key][non_idle_indices])
                #     for obs_key in demo_grp["obs"].keys():
                #         demo_grp_out.create_dataset(obs_key, data=demo_grp["obs"][obs_key][non_idle_indices])
                # for key in demo_grp.keys():
                #     if key not in ["action", "dones", "rewards", "obs"]:
                #         demo_grp_out.create_dataset(key, data=demo_grp[key][:])
            # Copy top-level attributes if any
            # for key in f_in.attrs:
            #     f_out.attrs[key] = f_in.attrs[key]
            
            # data_grp_in = f_in["data"]
            # data_grp_out = f_out.create_group("data")
            
            # # Copy data group attributes if any
            # for key in data_grp_in.attrs:
            #     data_grp_out.attrs[key] = data_grp_in.attrs[key]
            
            # demo_keys = sorted([k for k in data_grp_in.keys() if k.startswith("demo_")])
            
            # for demo_key in demo_keys:
            #     demo_grp_in = data_grp_in[demo_key]
            #     demo_grp_out = data_grp_out.create_group(demo_key)
                
            #     # Load actions: shape [H, D] where H is timesteps
            #     actions = demo_grp_in["action"][:].astype(np.float32)
            #     H, D = actions.shape
                
            #     # Determine how many initial actions to check
            #     if num_idle_actions is None:
            #         num_idle_actions = H
            #     else:
            #         num_idle_actions = min(num_idle_actions, H)
                
            #     # Check if first num_idle_actions are all zeros for each timestep
            #     # Sum over the first num_idle_actions and action dimension
            #     # Shape: [B] - True if all first num_idle_actions are zero for that timestep
            #     idle_mask = np.sum(np.abs(actions[:num_idle_actions, :-1]), axis=1) == 0
                
            #     # Get indices of non-idle timesteps
            #     non_idle_indices = np.where(~idle_mask)[0]
                
            #     print(f"{demo_key}: Removing {np.sum(idle_mask)} idle timesteps out of {H} total timesteps")
                
            #     # Filter action array
            #     demo_grp_out.create_dataset("action", data=actions[non_idle_indices])
                
            #     # Filter other time-indexed arrays if they exist
            #     time_indexed_keys = ["dones", "rewards"]
            #     for key in time_indexed_keys:
            #         if key in demo_grp_in:
            #             data = demo_grp_in[key][:]
            #             # if len(data.shape) > 0 and data.shape[0] == H:
            #             demo_grp_out.create_dataset(key, data=data[non_idle_indices])
                
            #     # Handle observations - they may have time as first dimension
            #     if "obs" in demo_grp_in:
            #         obs_grp_in = demo_grp_in["obs"]
            #         obs_grp_out = demo_grp_out.create_group("obs")
                    
            #         for obs_key in obs_grp_in.keys():
            #             obs_data = obs_grp_in[obs_key][:]
            #             # Check if first dimension matches timesteps
            #             if len(obs_data.shape) > 0 and obs_data.shape[0] == H:
            #                 obs_grp_out.create_dataset(obs_key, data=obs_data[non_idle_indices])
            #             else:
            #                 # If shape doesn't match, copy as-is (might be metadata)
            #                 obs_grp_out.create_dataset(obs_key, data=obs_data)
                
            #     # Copy other groups/attributes that are not time-indexed
            #     for key in demo_grp_in.keys():
            #         if key not in ["action", "dones", "rewards", "obs"]:
            #             # Check if it's a dataset with time dimension
            #             if isinstance(demo_grp_in[key], h5py.Dataset):
            #                 data = demo_grp_in[key][:]
            #                 if len(data.shape) > 0 and data.shape[0] == H:
            #                     demo_grp_out.create_dataset(key, data=data[non_idle_indices])
            #                 else:
            #                     demo_grp_out.create_dataset(key, data=data)
            #             else:
            #                 # It's a group, copy recursively
            #                 def copy_group(src, dst):
            #                     for k in src.keys():
            #                         if isinstance(src[k], h5py.Dataset):
            #                             data = src[k][:]
            #                             if len(data.shape) > 0 and data.shape[0] == H:
            #                                 dst.create_dataset(k, data=data[non_idle_indices])
            #                             else:
            #                                 dst.create_dataset(k, data=data)
            #                         else:
            #                             new_grp = dst.create_group(k)
            #                             copy_group(src[k], new_grp)
                            
            #                 new_group = demo_grp_out.create_group(key)
            #                 copy_group(demo_grp_in[key], new_group)
                
            #     # Copy demo group attributes if any
            #     for key in demo_grp_in.attrs:
            #         demo_grp_out.attrs[key] = demo_grp_in.attrs[key]

if __name__ == "__main__":
    remove_idle_actions_from_dataset(
        dataset_path="../../data/safemanibench/new_data_episode_starts_shelf_playback.hdf5",
        output_path="../../data/safemanibench/new_data_episode_starts_shelf_playback_no_idle.hdf5",
    )