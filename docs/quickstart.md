# Quickstart

## Teleoperation (Robocasa)
```bash
python scripts/teleop_robocasa.py \
  --env open_single_door \
  --device keyboard \
  --health-hud
```

## Playback (Robocasa)
```bash
python scripts/playback_robocasa.py \
  --input oopsiebench/test_data/robocasa/open_single_door_unsafe.hdf5 \
  --output demos/robocasa/playback_data/open_single_door_unsafe.hdf5 \
  --env open_single_door \
  --playback --visualize --metrics
```

## Teleoperation (Behavior-1K)

```bash
python scripts/teleop_b1k.py \
  --task_name shelve_item \
  --teleop_device keyboard \
  --live_feedback \
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