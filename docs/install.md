# Installation


## Prerequisites
- Conda / Miniconda 
- For simulators specific prerequisites, refer to [BEHAVIOR-1K](https://behavior.stanford.edu/index.html) and [RoboCasa](https://robocasa.ai/) prerequisites respectively

## 1. Clone the repository

=== "Linux / macOS"

    ```bash
    git clone https://github.com/UT-Austin-RobIn/oopsieverse.git
    cd oopsieverse
    ```

=== "Windows (PowerShell)"

    ```powershell
    git clone https://github.com/UT-Austin-RobIn/oopsieverse.git
    cd oopsieverse
    ```

## 2. Install the simulator
Create the environment(s) and install the simulator(s) you need. Each simulator gets its own conda environment (`oopsieverse_b1k` and/or `oopsieverse_robocasa`):
   ```bash
   # BEHAVIOR-1K (OmniGibson)
   python install.py --new_env --behavior1k

   # RoboCasa (RoboSuite / MuJoCo)
   python install.py --new_env --robocasa

   # ...or both at once
   python install.py --new_env --behavior1k --robocasa
   ```

OopsieVerse has been validated on specific tagged versions of the underlying simulator. While it may work with other versions, compatibility is not guaranteed. If you encounter any issues with a particular simulator version, please open an issue so we can investigate and improve support.

## 3. Install OopsieVerse

From the repository root:

```bash
conda activate oopsieverse_b1k      # or: oopsieverse_robocasa
pip install -e .
```

## 4. Verify simulator installs (optional)

**OmniGibson:**

```bash
python -m omnigibson.examples.robots.all_robots_visualizer
```

**RoboCasa:**

```bash
python -m robocasa.demos.demo_kitchen_scenes
```

## 5. Download demonstrations

Pull the pre-collected safe/unsafe teleop demos from Hugging Face into `oopsiebench/demos/`:

```bash
python scripts/download_demos.py                  # both simulators
python scripts/download_demos.py --sim robocasa   # or just one (robocasa / behavior1k)
```

Then head to the [Quickstart](quickstart.md) to browse and replay them.

<!-- 
!!! tip "Docs-only dependencies"

    To build this documentation locally:

    ```bash
    pip install -e ".[docs]"
    mkdocs serve
    ``` -->
