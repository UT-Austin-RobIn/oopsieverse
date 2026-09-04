import torch
import numpy as np

_NORM_EPS = 1e-8


def _as_float32(x):
    """Cast tensors / arrays to float32 so HDF5 float64 never promotes the batch."""
    if isinstance(x, torch.Tensor):
        return x.float()
    return np.asarray(x, dtype=np.float32)


def normalize_action(action, action_min, action_max):
    """Normalize the action to the range [0, 1]. Constant dims use span eps."""
    action = _as_float32(action)
    action_min = _as_float32(action_min)
    action_max = _as_float32(action_max)
    if isinstance(action, torch.Tensor):
        span = torch.clamp(
            torch.as_tensor(action_max - action_min, dtype=action.dtype, device=action.device),
            min=_NORM_EPS,
        )
        action_min_t = torch.as_tensor(action_min, dtype=action.dtype, device=action.device)
        return (action - action_min_t) / span
    span = np.maximum(action_max - action_min, _NORM_EPS)
    return (action - action_min) / span


def denormalize_action(action, action_min, action_max):
    """Denormalize the action to the original range."""
    action = _as_float32(action)
    action_min = _as_float32(action_min)
    action_max = _as_float32(action_max)
    if isinstance(action, torch.Tensor):
        action_min_t = torch.as_tensor(action_min, dtype=action.dtype, device=action.device)
        action_max_t = torch.as_tensor(action_max, dtype=action.dtype, device=action.device)
        return action * (action_max_t - action_min_t) + action_min_t
    return action * (action_max - action_min) + action_min
