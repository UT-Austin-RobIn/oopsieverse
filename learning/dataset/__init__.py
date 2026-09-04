from learning.dataset.dataset import ResizeSegmentation, PlaybackDataset, DEFAULT_OBS_KEYS

# Backward-compatible alias
B1KDataset = PlaybackDataset

__all__ = ["ResizeSegmentation", "PlaybackDataset", "B1KDataset", "DEFAULT_OBS_KEYS"]
