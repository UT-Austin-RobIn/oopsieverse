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
Collect a demonstration with live health feedback
```bash
python scripts/teleop_robocasa.py \
  --env open_single_door \
  --device keyboard \
  --health-hud
```
> On **macOS**, use `mjpython` instead of `python` when teleoperating with a SpaceMouse.

**Available environments:** `pick_egg`, `serve_pastry`, `place_plate`, `shelve_item`, `wipe_counter`, `open_drawer`, `close_drawer`, `open_single_door`, `turn_on_faucet`, `turn_on_microwave`, `turn_on_stove`, `counter_to_microwave`, `prepare_coffee`, `prepare_breakfast`, `dishes_to_sink`, `nav_lift_bowl`.

## Playback (Robocasa)
Playback, render videos, and compute health metrics:
```bash
python scripts/playback_robocasa.py \
  --input oopsiebench/demos/robocasa/teleop/open_single_door_unsafe.hdf5 \
  --output oopsiebench/demos/robocasa/playback/open_single_door_unsafe.hdf5 \
  --env open_single_door \
  --playback --visualize --metrics
```

## Teleoperation (Behavior-1K)
Collect a demonstration with live health feedback
```bash
python scripts/teleop_b1k.py \
  --task_name shelve_item \
  --teleop_device keyboard \
  --live_feedback \
  --save_video
```

**Available tasks:** `pick_egg`, `place_plate`, `fill_bowl`, `shelve_item`, `pour_water`, `wipe_counter`, `nav_to_table`, `open_drawer`, `open_single_door`, `turn_on_faucet`, `heat_saucepot`, `add_firewood`, `food_in_microwave`.

## Playback (Behavior-1K)
Playback, render videos, and compute health metrics:
```bash
python scripts/playback_b1k.py \
  --task_name shelve_item \
  --source_hdf5_path oopsiebench/demos/behavior1k/teleop/shelve_item_unsafe.hdf5 \
  --playback_hdf5_path oopsiebench/demos/behavior1k/playback/shelve_item_unsafe.hdf5 \
  --playback --visualize --compute_metrics
```
