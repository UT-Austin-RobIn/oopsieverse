# Install

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

## 2. Create the environment and install simulators

Run the installer with a fresh conda environment and the simulators you need:

```bash
python install.py --new_env --robocasa   # and/or --behavior1k
```

Activate the environment:

```bash
conda activate oopsieverse_robocasa     # or --oopsieverse_b1k
```

## 3. Install the Python package

From the repository root:

```bash
pip install -e .
```

## 4. Verify simulator installs (optional)

**OmniGibson (example smoke test):**

```bash
python -m omnigibson.examples.robots.all_robots_visualizer
```

**RoboCasa (example demo):**

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
