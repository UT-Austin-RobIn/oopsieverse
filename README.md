<p align="center">
  <img src="docs/assets/images/oopsieverse_logo.png" alt="OopsieVerse" width="340"/>
</p>

<h1 align="center">OopsieVerse</h1>

<p align="center">
  <b>A damage-aware, simulator-agnostic framework and benchmark for learning and evaluating safer robot manipulation.</b>
</p>

<p align="center">
  <a href="https://robin-lab.cs.utexas.edu/oopsieverse/"><b>🌐 Website</b></a> &nbsp;•&nbsp;
  <a href="https://robin-lab.cs.utexas.edu/oopsieverse/static/pdfs/oopsieverse.pdf"><b>📄 Paper</b></a> &nbsp;•&nbsp;
  <a href="https://ut-austin-robin.github.io/oopsieverse/documentation/"><b>📚 Documentation</b></a>
</p>

<p align="center">
  <img src="docs/assets/readme/lift_egg.gif" width="32%"/>
  &nbsp;
  <img src="docs/assets/readme/place_log.gif" width="32%"/>
  &nbsp;
  <img src="docs/assets/readme/pour_water.gif" width="32%"/>
</p>

---

## Overview

**What if your home robot finishes the job — but breaks your kitchen in the process?**

Robots are getting better at manipulating everyday objects, but task success alone is not enough. A robot that picks up an egg while cracking it, or pours water while spilling half of it, is not ready for real homes. The missing piece is safety — and today's simulators barely measure it.

OopsieVerse tackles this head on: a unified, damage-aware simulation framework for household manipulation. Instead of only rewarding task completion, it augments the standard decision-making setup with explicit damage signals and user-defined safety preferences. At its heart is **DamageSim**, a simulator-agnostic layer that converts physical signals like contact forces, temperature changes, and liquid interactions into measurable mechanical, thermal, or fluid damage. Paired with a benchmark suite of household tasks, OopsieVerse cleanly separates "did the robot succeed?" from "did it do so safely?"

Integrated into both **BEHAVIOR-1K** (Omniverse) and **RoboCasa** (MuJoCo), OopsieVerse supports safer data collection, damage-aware imitation and reinforcement learning, and safety benchmarking of Vision-Language-Action policies — for improved sim-to-real transfer of safer behaviors.

## Key Features

- **Unified damage signal** — a single continuous *health* value per object/link, aggregating mechanical, thermal, and electrical damage.
- **Simulator-agnostic** — one consistent damage API across BEHAVIOR-1K (OmniGibson) and RoboCasa (MuJoCo).
- **Real-time safety feedback** — live health bars and damage-based object coloring during teleoperation, so collectors can gather safer demonstrations.
- **Benchmark tasks** — a growing set of contact-, heat-, and fluid-rich manipulation tasks in realistic kitchens and homes.
- **End-to-end pipeline** — teleoperate → playback → render health-overlay videos and compute per-object safety metrics.

## Installation

### Prerequisites

- **Linux** is recommended. BEHAVIOR-1K (OmniGibson) requires an **NVIDIA GPU** with a recent driver; RoboCasa runs on CPU/GPU and also supports macOS.
- [**Conda / Miniconda**](https://docs.conda.io/en/latest/miniconda.html)

### Steps

1. **Clone the repository:**

   ```bash
   git clone https://github.com/UT-Austin-RobIn/oopsieverse.git
   cd oopsieverse
   ```

2. **Download test HDF5 demos** (not stored in git; hosted on Hugging Face):

   ```bash
   pip install huggingface_hub
   python scripts/download_test_data.py
   ```

3. **Create the environment(s) and install the simulator(s) you need.** Each simulator gets its own conda environment (`oopsieverse_b1k` and/or `oopsieverse_robocasa`):

   ```bash
   # BEHAVIOR-1K (OmniGibson)
   python install.py --new_env --behavior1k

   # RoboCasa (RoboSuite / MuJoCo)
   python install.py --new_env --robocasa

   # ...or both at once
   python install.py --new_env --behavior1k --robocasa
   ```

4. **Activate the environment and install OopsieVerse:**

   ```bash
   conda activate oopsieverse_b1k          # or: oopsieverse_robocasa
   pip install -e .
   ```

5. **Verify the simulator install:**

   ```bash
   # BEHAVIOR-1K
   python -m omnigibson.examples.robots.all_robots_visualizer

   # RoboCasa
   python -m robocasa.demos.demo_kitchen_scenes
   ```

## Usage

### Download demonstrations

Pre-collected **safe/unsafe** teleop demonstrations are hosted on Hugging Face. Download them into `oopsiebench/demos/`:

```bash
python scripts/download_demos.py                  # both simulators
python scripts/download_demos.py --sim robocasa   # or just one (robocasa / behavior1k)
```

### Quickstart: browse & visualize demos

The quickstart scripts interactively list the downloaded tasks, let you pick one (safe/unsafe), and replay it into a health-overlay video — the fastest way to see OopsieVerse in action:

```bash
conda activate oopsieverse            # BEHAVIOR-1K
python scripts/quickstart_b1k.py

conda activate oopsieverse_robocasa   # RoboCasa
python scripts/quickstart_robocasa.py
```

### Manual workflow

OopsieVerse uses a two-stage workflow:

1. **Teleoperate** to collect demonstrations, optionally with live damage feedback. Trajectories are saved to an HDF5 file.
2. **Playback** re-simulates the recorded trajectory to render images and health-overlay videos, and to compute per-object safety metrics.

> **Teleop controls:** drive the robot with the standard simulator key bindings, and press **`K`** to end the current episode (both simulators). In BEHAVIOR-1K, **`ESC`** quits and **`BACKSPACE`** discards the current episode; in RoboCasa, **`Ctrl+Q`** discards & quits and **`=`** toggles a free-camera pause.

### BEHAVIOR-1K (OmniGibson)

**Collect** a demonstration with live health feedback and a saved video:

```bash
python scripts/teleop_b1k.py --task_name shelve_item --live_feedback --save_video
```

**Playback**, render videos, and compute metrics:

```bash
python scripts/playback_b1k.py --task_name shelve_item \
  --source_hdf5_path   demos/behavior1k/teleop_data/shelve_item.hdf5 \
  --playback_hdf5_path demos/behavior1k/playback_data/shelve_item.hdf5 \
  --video_dir          demos/behavior1k/playback_videos/shelve_item \
  --playback --visualize --compute_metrics
```

Useful flags: `--n_episodes N`, `--collect_hdf5_path PATH`, `--overlay_links`, `--teleop_device {keyboard,spacemouse}`.

**Available tasks:** `pick_egg`, `place_plate`, `fill_bowl`, `shelve_item`, `pour_water`, `wipe_counter`, `nav_to_table`, `open_drawer`, `open_single_door`, `turn_on_faucet`, `heat_saucepot`, `add_firewood`, `food_in_microwave`.

### RoboCasa (RoboSuite / MuJoCo)

**Collect** a demonstration with the live health HUD:

```bash
python scripts/teleop_robocasa.py --env shelve_item --device keyboard --health-hud
```

> On **macOS**, use `mjpython` instead of `python` when teleoperating with a SpaceMouse.

**Playback**, render videos, and compute metrics:

```bash
python scripts/playback_robocasa.py \
  --input  demos/robocasa/teleop_data/shelve_item.hdf5 \
  --output demos/robocasa/playback_data/shelve_item.hdf5 \
  --env    shelve_item \
  --playback --visualize --metrics
```

Useful flags: `--device {keyboard,spacemouse}`, `--n-episodes N`, `--video`, `--health-color`, `--save-cameras`, `--save-health`.

**Available environments:** `pick_egg`, `serve_pastry`, `place_plate`, `shelve_item`, `wipe_counter`, `open_drawer`, `close_drawer`, `open_single_door`, `turn_on_faucet`, `turn_on_microwave`, `turn_on_stove`, `counter_to_microwave`, `prepare_coffee`, `prepare_breakfast`, `dishes_to_sink`, `nav_lift_bowl`.

## Troubleshooting

<details>
<summary><b>PyTorch / NVIDIA driver mismatch</b></summary>

If you see an error like:

```
The NVIDIA driver on your system is too old (found version 12080). Please update your GPU
driver ... Alternatively, go to https://pytorch.org to install a PyTorch version that has
been compiled with your version of the CUDA driver.
```

Reinstall a PyTorch build that matches your CUDA driver:

```bash
pip uninstall torch
pip install torch==2.9.1   # pick the version compatible with your CUDA
```
</details>

<details>
<summary><b>SpaceMouse teleoperation (Linux)</b></summary>

1. Create `/etc/udev/rules.d/99-spacemouse.rules` with:

   ```
   KERNEL=="hidraw*", ATTRS{idVendor}=="256f", ATTRS{idProduct}=="c635", MODE="0666", GROUP="plugdev"
   SUBSYSTEM=="usb", ATTRS{idVendor}=="256f", ATTRS{idProduct}=="c635", MODE="0666", GROUP="plugdev"
   ```

2. Reload the udev rules:

   ```bash
   sudo udevadm control --reload-rules
   ```

3. Use **`pyspacemouse==1.1.4`** — version `2.0.0` is known not to work.
</details>

## Citation

If you use OopsieVerse in your research, please cite:

```bibtex
@inproceedings{balaji2026oopsieverse,
  title={OopsieVerse: A Safety Benchmark with Damage-Aware Simulation for Robot Manipulation},
  author={Balaji, Arnav and Bahety, Arpit and Ambatipudi, Sriniket and Lam, Daniel and Xu, Junhong and Mart{\'\i}n-Mart{\'\i}n, Roberto},
  booktitle={Robotics: Science and Systems (RSS), 2026},
  year={2026}
}
```

## Acknowledgements

OopsieVerse builds on [BEHAVIOR-1K / OmniGibson](https://behavior.stanford.edu/), [RoboCasa](https://robocasa.ai/), and [RoboSuite](https://robosuite.ai/).
