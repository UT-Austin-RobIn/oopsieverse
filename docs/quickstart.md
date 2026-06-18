# Quickstart

## Download the demos

Pre-collected **safe/unsafe** teleop demonstrations are hosted on Hugging Face. Download them into `oopsiebench/demos/`:

```bash
python scripts/download_demos.py                  # both simulators
python scripts/download_demos.py --sim robocasa   # or just one (robocasa / behavior1k)
```

## Interactive visualizer (easiest)

Pick a task, pick safe/unsafe, and replay it into a health-overlay video — no flags to remember:

```bash
python scripts/quickstart_b1k.py        # BEHAVIOR-1K (conda env: oopsieverse)
python scripts/quickstart_robocasa.py   # RoboCasa    (conda env: oopsieverse_robocasa)
```

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
  --input oopsiebench/demos/robocasa/teleop/open_single_door_unsafe.hdf5 \
  --output oopsiebench/demos/robocasa/playback/open_single_door_unsafe.hdf5 \
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
  --source_hdf5_path oopsiebench/demos/behavior1k/teleop/shelve_item_unsafe.hdf5 \
  --playback_hdf5_path oopsiebench/demos/behavior1k/playback/shelve_item_unsafe.hdf5 \
  --playback --visualize --compute_metrics
```
