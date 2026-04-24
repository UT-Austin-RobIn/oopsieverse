import torch
import numpy as np

def normalize_action(action, action_min, action_max):
    """Normalize the action to the range [0, 1]."""
    return (action - action_min) / (action_max - action_min)

def denormalize_action(action, action_min, action_max):
    """Denormalize the action to the original range."""
    return action * (action_max - action_min) + action_min