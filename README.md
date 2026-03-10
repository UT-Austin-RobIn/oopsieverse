# OopsieVerse

OopsieVerse provides a damage-aware, simulator-agnostic framework and benchmark for evaluating and learning safer policies

### Installation Steps

1. Clone and enter the repository:
   - **Linux / macOS**: `git clone https://github.com/oopsiverse-anon/oopsiverse-anon.github.io && cd oopsiverse-anon.github.io`
2. Run the installer with `python install.py --new_env` and the simulators you need `--robocasa`
3. `conda activate oopsieverse`
4. `pip install -e .`
5. test Robocasa installation by: `python -m robocasa.demos.demo_kitchen_scenes`

# Commands

1. teleop (robocasa)
```bash
MUJOCO_GL=egl python scripts/teleop_robocasa.py --env pick_egg --device keyboard --health-hud --health-color --video --continuous-gripper
```

2. playback (robocasa)

```bash
python scripts/playback_robocasa.py --input demos/robocasa/teleop_data/pick_egg.hdf5 --output demos/robocasa/playback_data/pick_egg.hdf5 --env pick_egg --visualize --metrics
``` 