#!/usr/bin/env python3

import os
import sys
import subprocess
from pathlib import Path

def ensure_repo(repo_dir: Path, repo_url: str, branch: str = None):
    """
    Clone a git repository into repo_dir if it doesn't already exist.
    """
    if not repo_dir.exists():
        print(f"Cloning {repo_url} into {repo_dir}")
        cmd = ["git", "clone", repo_url, str(repo_dir)]
        if branch:
            cmd.extend(["-b", branch])
        subprocess.check_call(cmd, stdout=subprocess.DEVNULL)
    else:
        print(f"Repository already present at {repo_dir}")

# Path to Blender executable (adjust if needed)
BLENDER_PATH = os.environ.get("BLENDER_PATH", "/Applications/Blender.app/Contents/MacOS/Blender")

def ensure_phoneme_env(env_dir: Path):
    """
    Create a venv for phoneme extraction and install requirements if it doesn't exist.
    """
    if not env_dir.exists():
        print(f"Creating phoneme venv") # at {env_dir}")
        # Create virtual environment
        subprocess.check_call([sys.executable, "-m", "venv", str(env_dir)])
        # Upgrade pip in the venv and install phoneme requirements
        pip_bin = env_dir / "bin" / "pip"
        subprocess.check_call([str(pip_bin), "install", "--upgrade", "pip"], stdout=subprocess.DEVNULL)
        subprocess.check_call([str(pip_bin), "install", "-r", "requirements/phoneme.txt"], stdout=subprocess.DEVNULL)
    else:
        print(f"Phoneme venv already exists") # at {env_dir}")

def run_phoneme(env_dir: Path, wav_path: str, csv_path: str):
    """
    Invoke the phoneme extraction script inside its venv.
    """
    python_bin = env_dir / "bin" / "python"
    cmd = [
        str(python_bin),
        "scripts/phoneme.py",
        "--wav", wav_path,
        "--output", csv_path
    ]
    print("Running phoneme extraction") #:", " ".join(cmd))
    subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def ensure_nfr_env(env_dir: Path):
    """
    Create a venv for NFR rigging and install its requirements if missing.
    """
    if not env_dir.exists():
        print(f"Creating NFR venv at {env_dir}")
        subprocess.check_call([sys.executable, "-m", "venv", str(env_dir)])
        pip_bin = env_dir / "bin" / "pip"
        subprocess.check_call([str(pip_bin), "install", "--upgrade", "pip"])
        subprocess.check_call([str(pip_bin), "install", "-r", "requirements/nfr.txt"])
    else:
        print(f"NFR venv already exists at {env_dir}")

def run_nfr(
    env_dir: Path,
    script_path: Path,
    nfr_repo: str,
    neutral: str,
    target: str,
    poses_dir: str,
    output_dir: str,
    device: str = "cpu",
    project: bool = False
):
    """
    Invoke nfr.py inside its venv to batch-rig faces.
    """
    python_bin = env_dir / "bin" / "python"
    print(f"Using Python: {python_bin}")  # Debug line to ensure correct Python is being used

    # Check the Python version and packages in the subprocess
    check_python_cmd = [
        str(python_bin),
        "-c", 
        "import sys; print(sys.executable); import torch; print(torch.__version__)"
    ]
    subprocess.check_call(check_python_cmd)

    # Construct the command for running nfr.py
    cmd = [
        str(python_bin),
        str(script_path),
        "--nfr_repo", nfr_repo,
        "--neutral", neutral,
        "--target", target,
        "--poses_dir", poses_dir,
        "--output_dir", output_dir,
        "--device", device
    ]
    if project:
        cmd.append("--project")

    print("Running NFR rigging:", " ".join(cmd))
    subprocess.check_call(cmd)
    
import subprocess
from pathlib import Path

import subprocess
import time
from pathlib import Path

def run_nfr_pip(env_dir: Path):
    python_bin = env_dir / "bin" / "python"
    cmds = [
        f"{python_bin} -m pip uninstall torch-scatter -y",
        f"{python_bin} -m pip uninstall torch -y",
        f"{python_bin} -m pip cache purge",
        f"{python_bin} -m pip install torch==2.7.0"
    ]

    for cmd in cmds:
        print("Running:", cmd)
        subprocess.check_call(cmd, shell=True)

    # Verify torch installed before proceeding
    try:
        subprocess.check_call([str(python_bin), "-c", "import torch; print(torch.__version__)"])
    except subprocess.CalledProcessError:
        raise RuntimeError("torch install failed")

    # Now install torch-scatter
    scatter_cmd = f"{python_bin} -m pip install torch-scatter -f https://data.pyg.org/whl/torch-2.7.0+cpu.html"
    print("Running:", scatter_cmd)
    subprocess.check_call(scatter_cmd, shell=True)

def main():
    project_root = Path(__file__).parent
    # Define paths
    envs_dir      = project_root / "envs"
    data_dir      = project_root / "data"

    # Detect input audio and FBX by base name in data/input
    input_dir = data_dir / "input"
    wav_files = list(input_dir.glob("*.wav"))
    if not wav_files:
        raise FileNotFoundError(f"No WAV file found in {input_dir}")
    audio_wav = wav_files[0]
    stem = audio_wav.stem
    model_fbx = input_dir / f"{stem}.fbx"
    phoneme_env = envs_dir / "phoneme_env"
    phoneme_csv = data_dir / "phonemes" / f"{stem}_timeline.csv"
    faces_out = data_dir / "faces"

    # Ensure necessary directories exist
    for d in (
        envs_dir,
        data_dir / "input",
        data_dir / "phonemes",
        data_dir / "faces",
        data_dir / "faces" / "rigged_faces"
    ):
        os.makedirs(d, exist_ok=True)

    # Stage 1: Phoneme extraction
    ensure_phoneme_env(phoneme_env)
    run_phoneme(phoneme_env, str(audio_wav), str(phoneme_csv))

    print("✅ Phoneme extraction complete.")

    # Stage 2: Prepare model for Neural Face Rigging via Blender
    # Paths for FBX -> prepared OBJ conversion
    print("▶ Preparing model for NFR via Blender")
    script_path = project_root / "scripts" / "prepare_nfr.py"
    if not script_path.exists():
        raise FileNotFoundError(f"NFR preparer script not found: {script_path}")
    # Run preparer with CLI args
    subprocess.check_call([
        BLENDER_PATH,
        "--background",
        "--python", str(script_path),
        "--",
        "--fbx_path", str(model_fbx),
        "--output_dir", str(faces_out)
    ])
    print("✅ NFR model preparation complete.")

    # Ensure NFR_pytorch repo is available under src/
    src_dir = project_root / "src"
    os.makedirs(src_dir, exist_ok=True)
    nfr_repo_dir = src_dir / "NFR_pytorch"
    ensure_repo(
        nfr_repo_dir,
        "https://github.com/dafei-qin/NFR_pytorch.git",  # replace with the real upstream URL
    )

    # Stage 3: Run NFR rigging
    print("▶ Running NFR rigging")
    neutral_template  = nfr_repo_dir / "test-mesh" / "neutral" / "neutral.obj"
    prepared_head_obj = data_dir / "faces" / "your_head.obj"
    poses_dir         = nfr_repo_dir / "test-mesh" / "neutral" / "poses"
    rigged_faces_dir  = data_dir / "faces" / "rigged_faces"

    nfr_env = envs_dir / "nfr_env"
    nfr_script = project_root / "scripts" / "nfr.py"

    ensure_nfr_env(nfr_env)
    run_nfr(
        nfr_env,
        nfr_script,
        str(nfr_repo_dir),
        str(neutral_template),
        str(prepared_head_obj),
        str(poses_dir),
        str(rigged_faces_dir),
        device="cpu",
        project=False
    )
    print("✅ NFR rigging complete.")

if __name__ == "__main__":
    main()