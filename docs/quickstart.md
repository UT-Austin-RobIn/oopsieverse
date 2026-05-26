# Quickstart

## Teleoperation (Robocasa)
```bash
```

## Playback (Robocasa)
```bash
```

## Teleoperation (Behavior-1K)

```bash
python scripts/teleop_b1k.py 
  --task_name shelve_item 
  --teleop_device keyboard
  --live_feedback 
  --save_video
```

## Playback (Behavior-1K)

```bash
python scripts/playback_b1k.py \
  --task_name shelve_item \
  --source_hdf5_path oopsiebench/test_data/behavior1k/teleop_data/shelve_item_unsafe.hdf5 \
  --playback_hdf5_path demos/behavior1k/playback_data/shelve_item.hdf5 \
  --playback --visualize --compute_metrics
```