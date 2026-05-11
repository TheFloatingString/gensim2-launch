import modal
import os

REPO_URL = "https://github.com/TheFloatingString/GenSim2-fork.git"
REPO_DIR = "/opt/GenSim2-fork"

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install(
        [
            "git",
            "build-essential",
            "libvulkan-dev",
            "libvulkan1",
            "wget",
            "libgl1",
            "libglib2.0-0",
            "libglfw3",
            "libffi-dev",
            "libssl-dev",
            "curl",
            "ca-certificates",
            "clang",
        ]
    )
    .run_commands(
        "mkdir -p /etc/vulkan/icd.d",
        'echo \'{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0","api_version":"1.3.277"}}\''
        " > /etc/vulkan/icd.d/nvidia_icd.json",
    )
    .run_commands(
        "pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0"
        " --index-url https://download.pytorch.org/whl/cu121",
    )
    .run_commands(
        f"git clone --recursive {REPO_URL} {REPO_DIR}",
        f"cd {REPO_DIR} && pip install -e . || true",
    )
    .pip_install(
        [
            "numpy<2",
            "hydra-core==1.3.2",
            "omegaconf",
            "openai==0.28",
            "pyyaml==6.0",
            "pillow",
            "opencv-python",
            "transforms3d",
            "tqdm",
            "scikit-learn",
            "sapien",
            "trimesh",
            "einops",
            "sentence-transformers",
            "wandb",
        ]
    )
    .pip_install(
        [
            "ipython",
            "ipdb",
        ]
    )
    .pip_install(
        [
            "webcolors",
        ]
    )
    .pip_install(
        [
            "drake==1.22.0",
        ]
    )
    .pip_install(
        [
            "icecream",
            "numpy",
            "drake==1.22.0",
            "matplotlib",
            "opencv-python",
            "colored",
            "transforms3d",
            "gym",
            "pillow",
            "meshcat",
            "openai==0.28",
            "ipdb",
            # "open3d==0.16.0", # 0.16.0 is incompatible with the python version and other dependencies
            "open3d",
            "einops",
            "dm-reverb",
            "meshcat",
            "tabulate",
            "GPUtil",
            "psutil",
            "dotmap",
            "zarr",
            "transformers",
            # "sim-web-visualizer",
            "sapien==2.2.2",
            "mplib==0.1.1",
            "webcolors",
            "chardet",
            "tensorboard",
            "imageio",
            "hydra-core==1.3.2",
            "hydra-submitit-launcher==1.1.5",
            "shortuuid",
            "multimethod",
            # "chamferdist", # installing chamferdist causes some issues
            "sentence_transformers",
            "wandb",
            "easydict",
            "timm",
            "tqdm",
            # "scikit-learn==1.0.2",
            "pickleshare==0.7.5",
            "ninja==1.10.2.3",
            "gdown",
            "PyYAML==6.0",
            "protobuf==3.19.4",
            "termcolor==1.1.0",
            # "h5py==3.6.0",
            "pyvista",
            "setuptools==63.1.0",
            "Cython",
            "pandas",
            "deepspeed",
            "numba",
            "ray",
            "diffusers",
            "noise",
            "webcolors",
            "ftfy",
            "imageio[ffmpeg]",
            "gymnasium",
            "trimesh",
        ]
    )
    .pip_install(["numpy<2"])
    .pip_install(["open3d==0.18.0"])
    .run_commands("pip install --force-reinstall --no-binary :all: noise")
)

app = modal.App("gensim2-launch", image=image)


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
    import subprocess
    from pathlib import Path

    openai_key = os.environ.get("OPENAI_KEY")
    if not openai_key:
        raise ValueError(
            "OPENAI_KEY not found. Set it via Modal secret 'gensim2-secrets'."
        )

    # Write a launcher that runs in a single process so VK_ICD_FILENAMES stays
    # in effect after sapien's _vulkan_tricks.py has had its chance to override it.
    launcher = f"""\
import os, sys
sys.path.insert(0, "{REPO_DIR}")
os.chdir("{REPO_DIR}")

# Let sapien's _vulkan_tricks.py run on import, then take back VK_ICD_FILENAMES.
try:
    import sapien
except Exception:
    pass
os.environ["VK_ICD_FILENAMES"] = "/etc/vulkan/icd.d/nvidia_icd.json"
print("[vulkan] VK_ICD_FILENAMES ->", os.environ["VK_ICD_FILENAMES"])

# Python 3.11 requires random.sample to receive a sequence; dict.items() no longer qualifies.
import random as _random
_orig_sample = _random.sample
def _compat_sample(population, k, *a, **kw):
    if not isinstance(population, (list, tuple, range, str, bytes, bytearray)):
        population = list(population)
    return _orig_sample(population, k, *a, **kw)
_random.sample = _compat_sample

# Force headless: patch the reference in sim.py (where it's called), not constructor.py.
# Importing gensim2.env.sapien also imports sim.py via __init__.py → create_sapien → env → sim.
import gensim2.env.sapien.sim as _sim_mod
import gensim2.env.sapien.constructor as _ctor_mod

_orig_get_engine = _sim_mod.get_engine_and_renderer
def _headless_get_engine(*args, **kwargs):
    kwargs["use_gui"] = False
    return _orig_get_engine(*args, **kwargs)
_sim_mod.get_engine_and_renderer = _headless_get_engine

# Also patch Viewer in both modules to a no-op so render() / create_viewer() don't crash.
class _NoopViewer:
    def __init__(self, *a, **kw): pass
    def __call__(self, *a, **kw): return self
    def __getattr__(self, n): return self  # chains like viewer.window.set_camera_parameters(...) all resolve here
_sim_mod.Viewer = _NoopViewer
_ctor_mod.Viewer = _NoopViewer

# Run the pipeline script as __main__ so Hydra resolves config_path relative
# to the script file rather than the calling module.
sys.argv = [
    "{REPO_DIR}/gensim2/pipeline/run_pipeline.py",
    "prompt_folder={prompt_folder}",
    "prompt_data_folder={prompt_data_folder}",
    "num_tasks={num_tasks}",
    "output_folder={output_folder}",
    "++use_gui=false",
]
with open("{REPO_DIR}/gensim2/pipeline/run_pipeline.py") as _f:
    _src = _f.read()
exec(compile(_src, "{REPO_DIR}/gensim2/pipeline/run_pipeline.py", "exec"),
     {{"__name__": "__main__", "__file__": "{REPO_DIR}/gensim2/pipeline/run_pipeline.py"}})
"""
    launcher_path = "/tmp/gensim2_launcher.py"
    Path(launcher_path).write_text(launcher)

    result = subprocess.run(
        ["python", launcher_path],
        cwd=REPO_DIR,
        env={**os.environ, "PYTHONPATH": REPO_DIR},
    )

    output_path = Path(REPO_DIR) / output_folder
    if result.returncode == 0:
        files = list(output_path.rglob("*")) if output_path.exists() else []
        return {
            "status": "success",
            "output_folder": str(output_path),
            "num_output_files": len(files),
        }
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
    print("[*] Starting GenSim2 pipeline on Modal...")
    result = run_pipeline.remote(
        prompt_folder=prompt_folder,
        prompt_data_folder=prompt_data_folder,
        num_tasks=num_tasks,
    )
    print("\n[*] Result:")
    print(result)
