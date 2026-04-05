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

## Common issues
```bash
The NVIDIA driver on your system is too old (found version 12080). Please update your GPU driver by downloading and installing a new version from the URL: http://www.nvidia.com/Download/index.aspx Alternatively, go to: https://pytorch.org to install a PyTorch version that has been compiled with your version of the CUDA driver.
```
Uninstall torch and re-install a version compatible with your CUDA. We do:
`pip uninstall torch`
`pip install torch==2.9.1`

For using spacemouse:
Add this for spacemouse compact to /etc/udev/rules.d/99-spacemouse.rules
KERNEL=="hidraw*", ATTRS{idVendor}=="256f", ATTRS{idProduct}=="c635", MODE="0666", GROUP="plugdev"
SUBSYSTEM=="usb", ATTRS{idVendor}=="256f", ATTRS{idProduct}=="c635", MODE="0666", GROUP="plugdev"
then run `sudo udevadm control --reload-rules`

2. Works with pyspacemouse==1.1.4 (did not work with pyspacemouse==2.0.0)

# TODO:
- install telemoma for b1k
- force use mediapipe==0.10.21 (newer version gives an error related to "solutions")
