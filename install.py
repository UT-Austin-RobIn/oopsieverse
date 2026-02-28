import subprocess
from pathlib import Path
import argparse

ROOT = Path("externals")
ROOT.mkdir(exist_ok=True)

ENV_NAME = "oopsieverse"
PYTHON_VERSION = "3.10"

def run(cmd, cwd=None):
    print(f"[RUN] {cmd}")
    subprocess.check_call(cmd, shell=True, cwd=cwd)

def create_conda_env():
    """Create a new conda environment if it doesn't exist yet"""
    try:
        envs = subprocess.check_output("conda env list", shell=True, text=True)
        if ENV_NAME in envs:
            print(f"[INFO] Conda env '{ENV_NAME}' already exists.")
            return
    except Exception:
        pass

    print(f"[INFO] Creating new conda env '{ENV_NAME}' with Python {PYTHON_VERSION}...")
    run(f"conda create -y -n {ENV_NAME} python={PYTHON_VERSION}")
    print(f"[INFO] Done! Activate it with: conda activate {ENV_NAME}")

def install_behavior1k():
    repo = ROOT / "behavior1k"
    branch = "proj/safemanibench"
    url = "https://github.com/UT-Austin-RobIn/BEHAVIOR-1K.git"

    if not repo.exists():
        print(f"[INFO] Cloning Behavior1k from branch '{branch}'...")
        run(f"git clone --branch {branch} {url} {repo}")
    else:
        print(f"[INFO] Repository already exists at {repo}, skipping clone.")
    print("[INFO] Installing behavior1k...")
    run(f"conda run --no-capture-output -n {ENV_NAME} bash setup.sh --omnigibson --bddl --dataset", cwd=repo)

def install_robocasa():
    rc = ROOT / "robocasa"
    rs = ROOT / "robosuite"

    if not rs.exists():
        print("[INFO] Cloning RoboSuite repository...")
        run(f"git clone https://github.com/ARISE-Initiative/robosuite {rs}") 

    if not rc.exists():
        print("[INFO] Cloning RoboCasa repository...")
        run(f"git clone https://github.com/robocasa/robocasa {rc}")

    print("[INFO] Installing RoboSuite...")
    run(f"conda run --no-capture-output -n {ENV_NAME} pip install -e .", cwd=rs)
    print("[INFO] Installing RoboCasa...")
    run(f"conda run --no-capture-output -n {ENV_NAME} pip install -e .", cwd=rc)

    run(f"conda run --no-capture-output -n {ENV_NAME} python robocasa/scripts/download_kitchen_assets.py", cwd=rc)
    run(f"conda run --no-capture-output -n {ENV_NAME} python robocasa/scripts/setup_macros.py", cwd=rc)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Install OopsieVerse submodules")
    parser.add_argument("--new_env", action="store_true",
                        help="Create a new conda env named 'oopsieverse' with Python 3.10")
    parser.add_argument("--behavior1k", action="store_true", help="Install Behavior1k (OmniGibson)")
    parser.add_argument("--robocasa", action="store_true", help="Install RoboCasa and RoboSuite")

    args = parser.parse_args()

    if args.new_env:
        create_conda_env()

    if not args.behavior1k and not args.robocasa:
        print("[WARNING] No submodule selected. Use --behavior1k, --robocasa, or both.")
        exit(1)

    if args.behavior1k:
        install_behavior1k()

    if args.robocasa:
        install_robocasa()

    print("[SUCCESS] All requested submodules installed!")
