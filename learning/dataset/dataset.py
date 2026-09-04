import os
import h5py
import json
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data
from learning.utils import denormalize_action, normalize_action
from damagesim.utils.playback_schema import (
    PROPRIO_DIM,
    canonical_seg_key_from_obs_info_camera,
)


# Default B1K shelve_item cameras + proprio (unified playback schema).
DEFAULT_OBS_KEYS = [
    "cam/eef/seg",
    "cam/external_sensor0/seg",
    "cam/external_sensor1/seg",
    "proprio",
]


class ResizeSegmentation:
    """
    Transform to resize segmentation images using nearest-neighbor interpolation.
    This preserves integer class IDs (no interpolation artifacts).
    """
    
    def __init__(self, size=(128, 128)):
        """
        Args:
            size: Target size as (H, W) tuple
        """
        self.size = size
    
    def __call__(self, sample):
        """
        Args:
            sample: Dict with sample['obs']['extero'] containing seg images
                    Each seg image has shape [frame_stack, H, W]
        """
        if 'obs' in sample and 'extero' in sample['obs']:
            for key, seg_img in sample['obs']['extero'].items():
                
                # Add channel dim for F.interpolate: [T, 1, H, W]
                seg_img = seg_img.unsqueeze(1)
                
                # Resize with nearest neighbor (preserves integer class IDs)
                seg_img = F.interpolate(
                    seg_img, 
                    size=self.size, 
                    mode='nearest-exact'
                )
                
                # Remove channel dim: [T, H, W]
                sample['obs']['extero'][key] = seg_img.squeeze(1)
        
        return sample


class PlaybackDataset(data.Dataset):
    def __init__(
        self, 
        data_path, 
        frame_stack=2, 
        action_chunk_size=8,
        obs_keys=None,
        seg_img_size=(128, 128),
        transform=None,
        load_seg_mapping=True,
        objects_of_interest=None,
        normalize_action=True
    ):
        """
        Dataset for unified playback HDF5s (B1K and Robocasa).

        Args:
            data_path: Path to HDF5 file
            frame_stack: Number of frames to stack for observations (default: 2)
            action_chunk_size: Number of future actions to predict (default: 8)
            obs_keys: List of observation keys to load under ``obs/``
                (canonical: ``cam/{name}/seg``, ``proprio``). Defaults to
                :data:`DEFAULT_OBS_KEYS` (B1K shelve-style cameras).
            seg_img_size: Target size (H, W) for segmentation images (default: 128x128)
            transform: Optional additional transform to apply to samples
            load_seg_mapping: Whether to load segmentation ID to class name mapping
            objects_of_interest: If objects_of_interest is not None, only index segmentation classes that are indicated in this list.
                Other classes will be of class 0. 
            normalize_action: Whether to normalize the actions
        """
        self.data_path = data_path
        self.frame_stack = frame_stack
        self.action_chunk_size = action_chunk_size
        self.obs_keys = list(obs_keys) if obs_keys is not None else list(DEFAULT_OBS_KEYS)
        self.seg_img_size = seg_img_size
        self.transform = transform
        self.load_seg_mapping = load_seg_mapping
        self.objects_of_interest = objects_of_interest
        self.normalize_action = normalize_action
        self.state_dim = PROPRIO_DIM
        self.num_seg_views = sum(1 for k in self.obs_keys if "seg" in k)
        self.action_dim = None
        self.action_min = None
        self.action_max = None

        # Built-in resize transform for segmentation images
        self.resize_seg = ResizeSegmentation(size=seg_img_size) if seg_img_size else None
        
        # Store trajectory metadata for lazy loading
        self.samples = []  # List of (demo_key, timestep) tuples
        self.traj_data = {}  # Cache for loaded trajectory data
        self.seg_mappings = {}  # Cache for segmentation ID -> class name mappings
        self._index_trajectories()
        
        # Load global class vocabulary (all unique class names across dataset)
        if load_seg_mapping:
            self._build_class_vocabulary()

        self._pre_process_global_class_mapping()
        if normalize_action:
            print("Computing action norm statistics...")
            self.action_min, self.action_max = self.compute_action_norm_stat()

    def _pre_process_global_class_mapping(self):
        """
        Pre-process all segmentation images to global class IDs during initialization.
        Uses vectorized LUT indexing - no iteration over pixels.
        
        Trades memory for speed: all remapped seg images stay in RAM,
        so __getitem__ just slices pre-processed arrays.
        
        Creates:
            - self.remapped_seg_data: {demo_key: {obs_key: np.ndarray[T, H, W]}}
        """
        if not self.load_seg_mapping:
            return
        
        self.remapped_seg_data = {}
        
        with h5py.File(self.data_path, "r") as f:
            data_grp = f["data"]
            demo_keys = sorted([k for k in data_grp.keys() if k.startswith("demo_")])
            
            for demo_key in demo_keys:
                self.remapped_seg_data[demo_key] = {}
                demo_grp = data_grp[demo_key]
                obs_grp = demo_grp["obs"]
                
                for obs_key in self.obs_keys:
                    # Only process segmentation keys
                    if "seg" not in obs_key or obs_key not in obs_grp:
                        continue
                    
                    # Load entire trajectory's seg images: [T+1, H, W] (state-aligned)
                    seg_images = obs_grp[obs_key][:].astype(np.int64)
                    T, H, W = seg_images.shape

                    n_obs_info = len(self.seg_mappings.get(demo_key, {}))
                    if n_obs_info != T:
                        raise ValueError(
                            f"{demo_key}: len(info/obs_info)={n_obs_info} != "
                            f"len(obs/{obs_key})={T}. Playback HDF5 must store "
                            f"state-aligned info at T+1 (re-run playback after the "
                            f"T+1 info unification)."
                        )
                    
                    # Create output array for remapped images
                    remapped = np.zeros((T, H, W), dtype=np.int64)
                    
                    # Build per-timestep LUTs and apply vectorized remapping
                    for t in range(T):
                        seg_mapping = self.seg_mappings.get(demo_key, {}).get(t, {}).get(obs_key, {})
                        
                        if seg_mapping:
                            # Build LUT: lut[seg_id] = global_class_id
                            max_seg_id = max(seg_mapping.keys())
                            lut = np.zeros(max_seg_id + 1, dtype=np.int64)
                            
                            for seg_id, class_name in seg_mapping.items():
                                lut[seg_id] = self.class_to_id.get(class_name, 0)
                            
                            # Vectorized remapping: O(H*W) single operation, no pixel iteration!
                            seg_frame = np.clip(seg_images[t], 0, max_seg_id)
                            remapped[t] = lut[seg_frame]
                        # else: remapped[t] stays zeros (unknown class)
                    
                    self.remapped_seg_data[demo_key][obs_key] = remapped
        
        print(f"Pre-processed {len(self.remapped_seg_data)} demos with global class IDs")

    
    def _build_class_vocabulary(self):
        """
        Build a global vocabulary of all segmentation class names across the dataset.
        Scans ALL timesteps in ALL demos to ensure all possible classes are captured.
        
        The obs_info in HDF5 is stored as: data/demo_X/info/obs_info[timestep] -> JSON string
        JSON structure: {camera_type: {camera_name: {"seg_instance": {seg_id: class_name}}}}
        ``obs_info`` is state-aligned with ``obs`` (length T+1).
        
        Creates:
            - self.class_to_id: dict mapping class name -> consistent integer ID (for policy inference)
            - self.id_to_class: dict mapping integer ID -> class name (for decoding)
            - self.seg_mappings: per-demo, per-timestep mapping of original seg IDs to class names
              Structure: {demo_key: {timestep: {obs_key: {seg_id: class_name}}}}
              where obs_key uses canonical ``cam/{name}/seg`` paths.
        """
        all_classes = set()
        
        with h5py.File(self.data_path, "r") as f:
            data_grp = f["data"]
            demo_keys = sorted([k for k in data_grp.keys() if k.startswith("demo_")])
            
            for demo_key in demo_keys:
                demo_grp = data_grp[demo_key]
                
                if "info" in demo_grp and "obs_info" in demo_grp["info"]:
                    obs_info_data = demo_grp["info"]["obs_info"]
                    num_timesteps = len(obs_info_data)
                    
                    if num_timesteps > 0:
                        # Initialize per-demo, per-timestep mapping
                        self.seg_mappings[demo_key] = {}
                        
                        # Iterate through ALL timesteps to gather all classes
                        for timestep in range(num_timesteps):
                            # Decode the JSON string at this timestep
                            obs_info = json.loads(obs_info_data[timestep].decode("utf-8"))
                            
                            # Store per-timestep seg_id -> class_name mapping
                            self.seg_mappings[demo_key][timestep] = {}
                            
                            # Parse structure: obs_info[camera_type][camera_name]["seg_instance"]
                            for camera_type in obs_info:
                                if isinstance(obs_info[camera_type], dict):
                                    for camera_name in obs_info[camera_type]:
                                        if isinstance(obs_info[camera_type][camera_name], dict):
                                            if "seg_instance" in obs_info[camera_type][camera_name]:
                                                seg_mapping = obs_info[camera_type][camera_name]["seg_instance"]
                                                # seg_mapping: {"1": "object_name", "2": "another_obj", ...}
                                                all_classes.update(seg_mapping.values())

                                                # Match unified playback keys: cam/{name}/seg
                                                obs_key = canonical_seg_key_from_obs_info_camera(
                                                    camera_name
                                                )
                                                if obs_key is None:
                                                    continue
                                                # Store mapping: seg_id (int) -> class_name
                                                self.seg_mappings[demo_key][timestep][obs_key] = {
                                                    int(k): v for k, v in seg_mapping.items()
                                                }
        
        # Create consistent global mapping (sorted for reproducibility)
        # Reserve index 0 for "unknown" class (handles unseen seg IDs during inference)
        sorted_classes = sorted(all_classes)
        self.class_to_id = {"unknown": 0}
        self.id_to_class = {0: "unknown"}
        
        idx = 1
        for cls_name in sorted_classes:
            if self.objects_of_interest is None:
                self.class_to_id[cls_name] = idx
                self.id_to_class[idx] = cls_name
                idx += 1
            else:
                for obj in self.objects_of_interest:
                    # print(f"Checking if {cls_name} in {obj}, result: {cls_name in obj}")
                    if obj in cls_name: 
                        self.class_to_id[cls_name] = idx
                        self.id_to_class[idx] = cls_name
                        idx += 1
                        break
                    
        self.num_seg_classes = len(self.class_to_id)
        
        print(f"Built class vocabulary with {self.num_seg_classes} classes (including 'unknown'):")
        for cls_name, idx in self.class_to_id.items():
            print(f"  {idx}: {cls_name}")
    
    def get_class_name(self, demo_key, timestep, obs_key, seg_id):
        """
        Get the class name for a given segmentation ID at a specific timestep.
        
        Args:
            demo_key: Demo key (e.g., "demo_0")
            timestep: Timestep index (integer)
            obs_key: Observation key (e.g., "cam/external_sensor0/seg")
            seg_id: Segmentation ID from the image (integer)
        
        Returns:
            Class name string, or "unknown" if not found
        """
        if demo_key in self.seg_mappings:
            if timestep in self.seg_mappings[demo_key]:
                if obs_key in self.seg_mappings[demo_key][timestep]:
                    return self.seg_mappings[demo_key][timestep][obs_key].get(int(seg_id), "unknown")
        return "unknown"
    
    def get_class_id(self, demo_key, timestep, obs_key, seg_id):
        """
        Get the global class ID for a given segmentation ID at a specific timestep.
        This is the consistent ID used during policy training/inference.
        
        Args:
            demo_key: Demo key (e.g., "demo_0")
            timestep: Timestep index (integer)
            obs_key: Observation key (e.g., "cam/external_sensor0/seg")
            seg_id: Segmentation ID from the image (integer)
        
        Returns:
            Global class ID (integer), 0 (unknown) if not found
        """
        class_name = self.get_class_name(demo_key, timestep, obs_key, seg_id)
        return self.class_to_id.get(class_name, 0)
    
    def get_seg_mapping(self, demo_key, timestep, obs_key):
        """
        Get the full segmentation ID -> class name mapping for a demo/camera at a specific timestep.
        
        Args:
            demo_key: Demo key (e.g., "demo_0")
            timestep: Timestep index (integer)
            obs_key: Observation key (e.g., "cam/external_sensor0/seg")
        
        Returns:
            Dict mapping seg_id (int) -> class_name (str), or empty dict if not found
        """
        if demo_key in self.seg_mappings:
            if timestep in self.seg_mappings[demo_key]:
                return self.seg_mappings[demo_key][timestep].get(obs_key, {})
        return {}
    
    def remap_seg_image_to_class_ids(self, seg_image, demo_key, timestep, obs_key):
        """
        Remap a segmentation image from original seg IDs to global class IDs.
        This ensures consistent class indices during training and inference.
        
        Args:
            seg_image: Segmentation image tensor with original seg IDs
            demo_key: Demo key (e.g., "demo_0")
            timestep: Timestep index (integer)
            obs_key: Observation key (e.g., "cam/external_sensor0/seg")
        
        Returns:
            Remapped segmentation image with global class IDs
        """
        seg_mapping = self.get_seg_mapping(demo_key, timestep, obs_key)
        remapped = torch.zeros_like(seg_image)
        
        for seg_id, class_name in seg_mapping.items():
            class_id = self.class_to_id.get(class_name, 0)
            remapped[seg_image == seg_id] = class_id
        
        return remapped

    def _index_trajectories(self):
        """Index all valid (demo, timestep) pairs for frame stacking and action chunking."""
        with h5py.File(self.data_path, "r") as f:
            data_grp = f["data"]
            demo_keys = sorted([k for k in data_grp.keys() if k.startswith("demo_")])
            
            for demo_key in demo_keys:
                demo_grp = data_grp[demo_key]
                action_shape = demo_grp["action"].shape
                traj_len = action_shape[0]
                if self.action_dim is None and len(action_shape) > 1:
                    self.action_dim = int(action_shape[-1])

                # Valid timesteps: need (frame_stack - 1) frames before and action_chunk_size frames after
                # Start from (frame_stack - 1) to have enough history
                # End at (traj_len - action_chunk_size) to have enough future actions
                start_idx = self.frame_stack - 1
                end_idx = traj_len - self.action_chunk_size
                
                for t in range(start_idx, end_idx + 1):
                    self.samples.append((demo_key, t))
        
        print(f"Indexed {len(self.samples)} samples from {len(demo_keys)} trajectories")
        print(
            f"Frame stack: {self.frame_stack}, Action chunk: {self.action_chunk_size}, "
            f"action_dim: {self.action_dim}"
        )

    def _load_trajectory(self, demo_key):
        """Load and cache a trajectory."""
        if demo_key in self.traj_data:
            return self.traj_data[demo_key]
        
        with h5py.File(self.data_path, "r") as f:
            demo_grp = f["data"][demo_key]
            
            traj = {
                "action": np.asarray(demo_grp["action"][:], dtype=np.float32),
            }
            # Load observation keys if specified
            obs_grp = demo_grp["obs"]
            if self.obs_keys:
                for key in self.obs_keys:
                    if key in obs_grp:
                        traj[f"obs_{key}"] = obs_grp[key][:]
            
            self.traj_data[demo_key] = traj
        
        return traj

    def compute_action_norm_stat(self):
        """Compute the norm statistics of the actions (float32)."""
        all_actions = []
        with h5py.File(self.data_path, "r") as f:
            data_grp = f["data"]
            demo_keys = sorted([k for k in data_grp.keys() if k.startswith("demo_")])
            for demo_key in demo_keys:
                demo_grp = data_grp[demo_key]
                actions = np.asarray(demo_grp["action"][:], dtype=np.float32)
                if actions.shape[0] == 0:
                    print("No actions found for demo {demo_key}")
                    continue
                demo_action_min = np.min(actions, axis=0)
                demo_action_max = np.max(actions, axis=0)
                all_actions.append(actions)
                print(f"Action norm statistics for demo {demo_key}: Min:{demo_action_min}, Max:{demo_action_max}")
        all_actions = np.concatenate(all_actions, axis=0)
        action_min = np.min(all_actions, axis=0).astype(np.float32)
        action_max = np.max(all_actions, axis=0).astype(np.float32)
        print(f"Overall action norm statistics: Min:{action_min}, Max:{action_max}")
        return action_min, action_max

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        demo_key, t = self.samples[idx]
        traj = self._load_trajectory(demo_key)
        
        # Get stacked observations: [t - frame_stack + 1, ..., t]
        # Shape will be (frame_stack, ...) for each obs key
        sample = {'obs': {'proprio': None, 'extero': {}}}
        
        # Stack frames for each observation key
        if self.obs_keys:
            for key in self.obs_keys:
                obs_key = f"obs_{key}"
                if obs_key in traj:
                    # Get frames from t - frame_stack + 1 to t (inclusive)
                    if key == "proprio" or key.endswith("proprio"):
                        stacked = traj[obs_key][t]
                        sample['obs']['proprio'] = torch.from_numpy(stacked).float()
                    else:
                        # Use pre-processed remapped seg data if available
                        if (self.load_seg_mapping and 
                            demo_key in self.remapped_seg_data and 
                            key in self.remapped_seg_data[demo_key]):
                            # Already remapped to global class IDs - just slice!
                            stacked = self.remapped_seg_data[demo_key][key][t - self.frame_stack + 1 : t + 1]
                            sample['obs']['extero'][key] = torch.from_numpy(stacked).float()
                        else:
                            # Fallback to original data (no remapping)
                            stacked = traj[obs_key][t - self.frame_stack + 1 : t + 1]
                            sample['obs']['extero'][key] = torch.from_numpy(stacked).float()
        
        # Get action chunk: [t, t+1, ..., t + action_chunk_size - 1]
        action_chunk = np.asarray(
            traj["action"][t : t + self.action_chunk_size], dtype=np.float32
        )
        sample["action"] = torch.from_numpy(action_chunk)
        if self.normalize_action:
            sample["action"] = normalize_action(sample["action"], self.action_min, self.action_max)
        
        # Add metadata
        sample["demo_key"] = demo_key
        sample["timestep"] = t
        
        # Apply built-in resize transform for segmentation images
        if self.resize_seg:
            sample = self.resize_seg(sample)
        
        # Apply additional user transforms
        if self.transform:
            sample = self.transform(sample)
        
        return sample

if __name__ == "__main__":
    objects_of_interest = ["box_of_crackers", "book", "bottle_of_wine", "bottle_of_beer", "stand"]
    dataset = PlaybackDataset(
        data_path="demos/behavior1k/playback_data/shelve_cereal_box_1.hdf5",
        frame_stack=2,
        action_chunk_size=8,
        load_seg_mapping=True,
        normalize_action=True,
        objects_of_interest=objects_of_interest,
    )
    print(f"\nDataset size: {len(dataset)}")
    print(f"state_dim={dataset.state_dim}, num_seg_views={dataset.num_seg_views}")

    sample = dataset[0]
    print(f"\nSample keys: {sample.keys()}")
    print(f"Demo key: {sample['demo_key']}, Timestep: {sample['timestep']}")
    print(f"Action shape: {sample['action'].shape}")
    print(f"Proprio shape: {sample['obs']['proprio'].shape}")
    print(f"Extero keys: {list(sample['obs']['extero'].keys())}")

    for obs_key in ["cam/external_sensor0/seg"]:
        if obs_key in sample['obs']['extero']:
            seg_img = sample['obs']['extero'][obs_key]
            unique_class_ids = torch.unique(seg_img).int().tolist()
            print(f"\nUnique global class IDs in {obs_key}: {unique_class_ids}")
            for class_id in unique_class_ids:
                class_name = dataset.id_to_class.get(class_id, "unknown")
                print(f"  global_class_id {class_id} -> class_name '{class_name}'")

    print(f"\nTotal number of segmentation classes: {dataset.num_seg_classes}")
