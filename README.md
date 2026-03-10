# OopsieVerse

OopsieVerse provides a damage-aware, simulator-agnostic framework and benchmark for evaluating and learning safer policies

### Installation Steps

1. Clone and enter the repository:
   - **Linux / macOS**: `git clone https://github.com/UT-Austin-RobIn/oopsieverse.git && cd oopsieverse`
   - **Windows**: `git clone https://github.com/UT-Austin-RobIn/oopsieverse.git; cd oopsieverse`
2. Run the installer with `python install.py --new_env` and the simulators you need `--robocasa` or `--behavior1k`
3. `conda activate oopsieverse`
4. `pip install -e .`
5. test OG installation by: `python -m omnigibson.examples.robots.all_robots_visualizer`
6. test Robocasa installation by: `python -m robocasa.demos.demo_kitchen_scenes`

# Commands

1. teleop (b1k)
```bash
python scripts/teleop_b1k.py --task_name shelve_item --live_feedback --save_video
```

2. playback (b1k)

```bash
python scripts/playback_b1k.py --task_name shelve_item --collect_hdf5_path demos/behavior1k/teleop_data/shelve_item.hdf5      --playback_hdf5_path demos/behavior1k/playback_data/shelve_item.hdf5 --playback --visualize --compute_metrics
```
