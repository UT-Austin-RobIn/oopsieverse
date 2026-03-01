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

1. playback

```bash
python scripts/playback.py --task_name shelve_item --playback --collect_hdf5_path /mnt/ssd/safe-manipulation-benchmark/resources/teleop_data/shelve_item/trial_1.hdf5 --playback_hdf5_path resources/playback_data/temp.hdf5 --demo_ids 0
```
