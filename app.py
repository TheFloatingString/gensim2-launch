import modal
import os
import subprocess
from pathlib import Path

# Define the Modal image with micromamba and CUDA support
image = (
    modal.Image.debian_slim()
    .apt_install([
        "git",
        "build-essential",
        "libzmq3-dev",
        "libffi-dev",
        "libssl-dev",
        "wget",
        "curl",
        "ca-certificates",
    ])
    .run_commands(
        "mkdir -p /tmp/mamba && cd /tmp/mamba && curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj",
        "mv /tmp/mamba/bin/micromamba /usr/local/bin/micromamba && chmod +x /usr/local/bin/micromamba",
        "micromamba create -y -n gensim2 -c conda-forge python=3.11 pip",
    )
)

app = modal.App(
    name="gensim2-pipeline",
    image=image,
)


@app.function(
    gpu="T4",
    timeout=3600,
    secrets=[modal.Secret.from_name("gensim2-secrets")],
)
def run_pipeline(
    prompt_folder: str = "keypoint_pipeline_articulated_3stage",
    prompt_data_folder: str = "data_articulated/",
    num_tasks: int = 1,
    output_folder: str = "logs/",
) -> dict:
    """
    Run the GenSim2 pipeline on Modal with CUDA support using micromamba.

    Args:
        prompt_folder: Name of the prompt folder in prompts/
        prompt_data_folder: Name of the data folder in prompts/
        num_tasks: Number of tasks to generate
        output_folder: Output folder for results

    Returns:
        dict with status and output paths
    """
    import shutil

    # Set OpenAI API key from Modal secret
    openai_key = os.environ.get("OPENAI_KEY")
    if not openai_key:
        raise ValueError(
            "OPENAI_KEY not found in environment. "
            "Set it via Modal secret 'gensim2-secrets'"
        )
    os.environ["OPENAI_KEY"] = openai_key

    work_dir = Path("/tmp/gensim2-work")
    work_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(work_dir)

    print("[*] Cloning GenSim2-fork repository...")
    repo_dir = work_dir / "GenSim2-fork"
    if repo_dir.exists():
        shutil.rmtree(repo_dir)

    subprocess.run(
        ["git", "clone", "--recursive",
         "https://github.com/TheFloatingString/GenSim2-fork.git"],
        cwd=work_dir,
        check=True,
        capture_output=True,
    )

    os.chdir(repo_dir)
    print(f"[+] Repository cloned to {repo_dir}")

    print("[*] Installing PyTorch with CUDA support...")

    # Helper function to run commands in micromamba environment
    def run_cmd(cmd, check=True, capture_output=False):
        """Run command in gensim2 micromamba environment"""
        full_cmd = f"micromamba run -n gensim2 {cmd}"
        return subprocess.run(
            full_cmd,
            shell=True,
            check=check,
            capture_output=capture_output,
        )

    # Install PyTorch with CUDA 12.1
    run_cmd(
        "pip install -q torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 "
        "--index-url https://download.pytorch.org/whl/cu121"
    )

    print("[*] Installing core dependencies...")

    # Install iopath
    run_cmd("pip install -q iopath")

    # Try to install pytorch3d, but continue if it fails
    print("[*] Attempting pytorch3d installation (optional)...")
    try:
        run_cmd(
            "pip install pytorch3d -f "
            "https://dl.fbaipublicfiles.com/pytorch3d/get_install_url.html",
            check=False,
        )
        print("[+] pytorch3d installed")
    except Exception as e:
        print(f"[!] pytorch3d installation skipped: {e}")

    # Install requirements
    run_cmd("pip install -q -r requirements.txt")

    # Install the package
    run_cmd("pip install -q -e .")

    print("[*] Building C++ extensions...")

    # Build C++ extensions
    extensions = [
        "gensim2/agent/third_party/openpoints/cpp/pointnet2_batch",
        "gensim2/agent/third_party/openpoints/cpp/subsampling",
        "gensim2/agent/third_party/openpoints/cpp/pointops",
        "gensim2/agent/third_party/openpoints/cpp/chamfer_dist",
        "gensim2/agent/third_party/openpoints/cpp/emd",
    ]

    for ext in extensions:
        ext_path = repo_dir / ext
        if ext_path.exists():
            print(f"  [*] Building {ext_path.name}...")
            try:
                if "chamfer_dist" in ext or "emd" in ext:
                    run_cmd(
                        f"cd {ext_path} && python setup.py install --user",
                        capture_output=True,
                    )
                elif "subsampling" in ext:
                    run_cmd(
                        f"cd {ext_path} && python setup.py build_ext --inplace",
                        capture_output=True,
                    )
                else:
                    run_cmd(
                        f"cd {ext_path} && python setup.py install",
                        capture_output=True,
                    )
                print(f"    [+] {ext_path.name} built successfully")
            except subprocess.CalledProcessError as e:
                print(f"    [!] Warning: Failed to build {ext_path.name}")
                print(f"    Error: {e}")
        else:
            print(f"    [!] {ext} not found")

    print("[*] Running GenSim2 pipeline...")

    # Run the pipeline in conda environment
    cmd = (
        f'cd {repo_dir} && '
        f'python gensim2/pipeline/run_pipeline.py '
        f'prompt_folder={prompt_folder} '
        f'prompt_data_folder={prompt_data_folder} '
        f'output_folder={output_folder} '
        f'num_tasks={num_tasks}'
    )

    print(f"  Running: {cmd}")

    result = run_cmd(cmd, check=False)

    if result.returncode == 0:
        print("[+] Pipeline completed successfully!")

        output_path = repo_dir / output_folder
        if output_path.exists():
            output_files = list(output_path.rglob("*"))
            return {
                "status": "success",
                "output_folder": str(output_path),
                "num_output_files": len(output_files),
            }
        else:
            return {
                "status": "success",
                "message": "Pipeline ran but output folder not found",
            }
    else:
        return {
            "status": "failed",
            "return_code": result.returncode,
        }


@app.local_entrypoint()
def main(
    prompt_folder: str = "keypoint_pipeline_articulated_3stage",
    prompt_data_folder: str = "data_articulated/",
    num_tasks: int = 1,
):
    """
    Local entrypoint to trigger the pipeline remotely on Modal.

    Usage:
        modal run modal.py --prompt-folder keypoint_pipeline_articulated_3stage
    """
    print("[*] Starting GenSim2 pipeline on Modal...")
    result = run_pipeline.remote(
        prompt_folder=prompt_folder,
        prompt_data_folder=prompt_data_folder,
        num_tasks=num_tasks,
    )
    print("\n[*] Result:")
    print(result)
