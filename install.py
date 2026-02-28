import subprocess
import sys
from pathlib import Path
import argparse

ROOT = Path("externals")
ROOT.mkdir(exist_ok=True)

ENV_NAME = "oopsieverse"
PYTHON_VERSION = "3.10"
IS_WINDOWS = sys.platform == "win32"

def run(cmd, cwd=None):
    print(f"[RUN] {cmd}")
    subprocess.check_call(cmd, shell=True, cwd=cwd)

def conda_run(cmd, cwd=None):
    """Run a command inside the conda environment, cross-platform."""
    if IS_WINDOWS:
        full_cmd = f"conda run -n {ENV_NAME} --no-capture-output {cmd}"
    else:
        full_cmd = f"bash -c 'source activate {ENV_NAME} && {cmd}'"
    run(full_cmd, cwd=cwd)

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
    if IS_WINDOWS:
        conda_run("bash setup.sh --omnigibson --bddl --dataset", cwd=repo)
    else:
        conda_run("./setup.sh --omnigibson --bddl --dataset", cwd=repo)

def patch_robocasa_windows(rc):
    """Patch robocasa source files for Windows compatibility."""
    if not IS_WINDOWS:
        return

    demo = rc / "robocasa" / "demos" / "demo_kitchen_scenes.py"
    if not demo.exists():
        return

    text = demo.read_text()
    if "import msvcrt" in text:
        return

    print("[INFO] Patching robocasa for Windows compatibility...")

    text = text.replace(
        "import termios",
        'if sys.platform == "win32":\n    import msvcrt\nelse:\n    import termios',
    )

    text = text.replace(
        "termios.tcflush(sys.stdin, termios.TCIFLUSH)",
        'if sys.platform == "win32":\n'
        '            while msvcrt.kbhit():\n'
        '                msvcrt.getch()\n'
        '        else:\n'
        '            termios.tcflush(sys.stdin, termios.TCIFLUSH)',
    )

    demo.write_text(text)
    print("[INFO] Patched demo_kitchen_scenes.py for Windows.")

def install_robocasa():
    rc = ROOT / "robocasa"
    rs = ROOT / "robosuite"

    if not rs.exists():
        print("[INFO] Cloning RoboSuite repository...")
        run(f"git clone https://github.com/ARISE-Initiative/robosuite {rs}")

    if not rc.exists():
        print("[INFO] Cloning RoboCasa repository...")
        run(f"git clone https://github.com/robocasa/robocasa {rc}")

    patch_robocasa_windows(rc)

    print("[INFO] Installing RoboSuite...")
    conda_run("pip install -e .", cwd=rs)
    print("[INFO] Installing RoboCasa...")
    conda_run("pip install -e .", cwd=rc)

    conda_run("python robocasa/scripts/download_kitchen_assets.py", cwd=rc)
    conda_run("python robocasa/scripts/setup_macros.py", cwd=rc)

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
