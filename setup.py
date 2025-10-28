import subprocess
import sys
import platform
from pathlib import Path

def install_requirements(venv_path=".venv"):
    pip_path = Path(venv_path) / ("Scripts" if platform.system() == "Windows" else "bin") / "pip"
    subprocess.run([str(pip_path), "install", "--upgrade", "pip"], check=True)
    if Path("requirements.txt").exists():
        subprocess.run([str(pip_path), "install", "-r", "requirements.txt"], check=True)

def create_venv(venv_path=".venv"):
    print(f"Creating new virtual environment at {venv_path}...")
    subprocess.run([sys.executable, "-m", "venv", venv_path], check=True)

def run_shell_script(script_path):
    print(f"Running {script_path}...")
    subprocess.run(["bash", script_path], check=True)

def run_python_script(script_name, args=None, venv_path=".venv"):
    python_exe = Path(venv_path) / ("Scripts" if platform.system() == "Windows" else "bin") / "python"
    command = [str(python_exe), script_name]
    if args:
        command.extend(args)
    print(f"Running: {' '.join(command)}")
    subprocess.run(command, check=True)

# Run in root project directory

# Setup artifact folders
dirs = [
    Path("intermediary/model"),
    Path("data/csvs")
]
for d in dirs:
    d.mkdir(parents=True, exist_ok=True)
    print(f"📁 Created: {d}")

# Setup .venv
print("Setting up .venv")
create_venv()
install_requirements()

## Curl all the contract csvs
#print("Curling Contract CSVS")
#run_shell_script("scripts/curl_csvs.sh")
#
## Combine csvs into one csv
#run_python_script("scripts/csv_combine.py")
