# OopsieVerse

OopsieVerse provides a damage-aware, simulator-agnostic framework and benchmark for evaluating and learning safer policies

### Installation Steps

1. Clone and enter the repository:
   - **Linux / macOS**: `git clone https://github.com/UT-Austin-RobIn/oopsieverse.git && cd oopsieverse`
   - **Windows (PowerShell)**: `git clone https://github.com/UT-Austin-RobIn/oopsieverse.git; cd oopsieverse`
2. Add --behavior1k and/or --robocasa: `python install.py --new_env`
3. `conda activate oopsieverse`
4. `pip install -e .`
5. Test OG installation: `python -m omnigibson.examples.robots.all_robots_visualizer`
6. Test RoboCasa installation: `python -m robocasa.demos.demo_kitchen_scenes`
