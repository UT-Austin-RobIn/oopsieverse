import subprocess
import platform
import shutil
from pathlib import Path
import argparse

ROOT = Path("externals")
ROOT.mkdir(exist_ok=True)

B1K_ENV_NAME = "oopsieverse_b1k"
ROBOCASA_ENV_NAME = "oopsieverse_robocasa"
PYTHON_VERSION = "3.10"
IS_WINDOWS = platform.system() == "Windows"

def run(cmd, cwd=None):
    print(f"[RUN] {cmd}")
    subprocess.check_call(cmd, shell=True, cwd=cwd)

def conda_run(env_name, cmd, cwd=None):
    """Run a command inside the conda environment."""
    run(f"conda run --no-capture-output -n {env_name} {cmd}", cwd=cwd)

def _find_bash():
    """Locate a bash executable on Windows."""
    bash = shutil.which("bash")
    if bash:
        return bash
    for candidate in [
        Path(r"C:\Program Files\Git\bin\bash.exe"),
        Path(r"C:\Program Files (x86)\Git\bin\bash.exe"),
    ]:
        if candidate.exists():
            return str(candidate)
    return None

def _conda_env_exists(env_name):
    """Return True if the named conda environment already exists."""
    try:
        envs = subprocess.check_output("conda env list", shell=True, text=True)
        for line in envs.splitlines():
            if line.split() and line.split()[0] == env_name:
                return True
    except Exception:
        pass
    return False

def create_conda_env(env_name):
    """Create a named conda environment if it doesn't exist yet."""
    if _conda_env_exists(env_name):
        print(f"[INFO] Conda env '{env_name}' already exists.")
        return

    print(f"[INFO] Creating new conda env '{env_name}' with Python {PYTHON_VERSION}...")
    run(f"conda create -y -n {env_name} python={PYTHON_VERSION}")
    print(f"[INFO] Done! Activate it with: conda activate {env_name}")

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
        bash = _find_bash()
        if bash is None:
            print("[ERROR] bash is required to install behavior1k but was not found.")
            print("[ERROR] Install Git for Windows (https://git-scm.com) which includes bash,")
            print("[ERROR] then re-run this script.")
            exit(1)
        print(f"[INFO] Using bash at: {bash}")
        conda_run(B1K_ENV_NAME, f'"{bash}" setup.sh --omnigibson --bddl --dataset', cwd=repo)
    else:
        conda_run(B1K_ENV_NAME, "bash setup.sh --omnigibson --bddl --dataset", cwd=repo)

def patch_robocasa_numba_pin(rc):
    """Relax robocasa's strict numba pin, no conflict with the version required by oopsieverse."""
    setup_py = rc / "setup.py"
    text = setup_py.read_text()
    old = '"numba==0.61.2"'
    new = '"numba>=0.61.2"'
    if old in text:
        print("[PATCH] Relaxing robocasa numba pin from ==0.61.2 to >=0.61.2")
        setup_py.write_text(text.replace(old, new))

def patch_robocasa_for_windows(rc):
    """Patch robocasa files that use Unix-only modules so they work on Windows."""
    if not IS_WINDOWS:
        return

    demo = rc / "robocasa" / "demos" / "demo_kitchen_scenes.py"
    if not demo.exists():
        return

    text = demo.read_text()
    changed = False

    if "import termios\n" in text and "try:" not in text.split("import termios")[0][-10:]:
        print("[PATCH] Making termios import conditional for Windows")
        text = text.replace(
            "import termios",
            "try:\n    import termios\nexcept ImportError:\n    termios = None",
        )
        changed = True

    if "termios.tcflush(sys.stdin, termios.TCIFLUSH)" in text and "if termios" not in text:
        print("[PATCH] Adding Windows fallback for termios.tcflush")
        text = text.replace(
            "        termios.tcflush(sys.stdin, termios.TCIFLUSH)",
            "        if termios is not None:\n"
            "            termios.tcflush(sys.stdin, termios.TCIFLUSH)\n"
            "        else:\n"
            "            import msvcrt\n"
            "            while msvcrt.kbhit():\n"
            "                msvcrt.getch()",
        )
        changed = True

    if changed:
        demo.write_text(text)

def install_robocasa():
    rc = ROOT / "robocasa"
    rs = ROOT / "robosuite"

    if not rs.exists():
        print("[INFO] Cloning RoboSuite repository...")
        run(f"git clone https://github.com/ARISE-Initiative/robosuite {rs}")
        run(f"cd {rs} && git checkout aaa8b9b")

    if not rc.exists():
        print("[INFO] Cloning RoboCasa repository...")
        run(f"git clone https://github.com/robocasa/robocasa {rc}")
        run(f"cd {rc} && git checkout 97a4060")

    patch_robocasa_numba_pin(rc)
    patch_robocasa_for_windows(rc)

    print("[INFO] Installing RoboSuite...")
    conda_run(ROBOCASA_ENV_NAME, "pip install -e .", cwd=rs)
    print("[INFO] Installing RoboCasa...")
    conda_run(ROBOCASA_ENV_NAME, "pip install -e .", cwd=rc)

    conda_run(ROBOCASA_ENV_NAME, "python robocasa/scripts/download_kitchen_assets.py", cwd=rc)
    conda_run(ROBOCASA_ENV_NAME, "python robocasa/scripts/setup_macros.py", cwd=rc)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Install OopsieVerse submodules")
    parser.add_argument("--new_env", action="store_true",
                        help="Create the needed conda env(s) for selected installs")
    parser.add_argument("--behavior1k", action="store_true", help="Install Behavior1k (OmniGibson)")
    parser.add_argument("--robocasa", action="store_true", help="Install RoboCasa and RoboSuite")

    args = parser.parse_args()

    if args.new_env:
        if args.behavior1k:
            create_conda_env(B1K_ENV_NAME)
        if args.robocasa:
            create_conda_env(ROBOCASA_ENV_NAME)
        if not args.behavior1k and not args.robocasa:
            create_conda_env(B1K_ENV_NAME)
            create_conda_env(ROBOCASA_ENV_NAME)
            print("[SUCCESS] Both conda environments are ready.")
            exit(0)

    if not args.behavior1k and not args.robocasa:
        print("[WARNING] No submodule selected. Use --behavior1k, --robocasa, or both.")
        exit(1)

    # Ensure required conda env(s) exist before installing
    if args.behavior1k and not _conda_env_exists(B1K_ENV_NAME):
        print(f"[ERROR] Conda environment '{B1K_ENV_NAME}' does not exist.")
        print(f"[ERROR] Re-run with --new_env to create it:")
        print(f"[ERROR]   python install.py --new_env --behavior1k")
        exit(1)
    if args.robocasa and not _conda_env_exists(ROBOCASA_ENV_NAME):
        print(f"[ERROR] Conda environment '{ROBOCASA_ENV_NAME}' does not exist.")
        print(f"[ERROR] Re-run with --new_env to create it:")
        print(f"[ERROR]   python install.py --new_env --robocasa")
        exit(1)

    if args.behavior1k:
        install_behavior1k()

    if args.robocasa:
        install_robocasa()

    print("[SUCCESS] All requested submodules installed!")
