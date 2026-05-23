# Quickstart

Minimal commands to exercise **Behavior-1K** workflows after [Install](install.md).

## Teleoperation (Behavior-1K)

```bash
python scripts/teleop_b1k.py --task_name shelve_item --live_feedback --save_video
```

## Playback and metrics (Behavior-1K)

```bash
python scripts/playback_b1k.py \
  --task_name shelve_item \
  --collect_hdf5_path demos/behavior1k/teleop_data/shelve_item.hdf5 \
  --playback_hdf5_path demos/behavior1k/playback_data/shelve_item.hdf5 \
  --playback --visualize --compute_metrics
```

!!! note "RoboCasa / Robosuite"

    RoboCasa-oriented scripts live under `scripts/` (for example `playback_robocasa.py`). Task names align with the [OopsieBench](oopsiebench.md) RoboCasa registry.
