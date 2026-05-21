# GenSim2-fork Deployment on Modal

Modal serverless app to run [GenSim2-fork](https://github.com/TheFloatingString/GenSim2-fork) with CUDA T4 GPU.

## Prerequisites

- Modal CLI installed: `pip install modal`
- Modal account authenticated: `modal setup`
- OpenAI API key

## Setup

### Create Modal Secret

```bash
modal secret create gensim2-secrets
```

When prompted, create:
```
Key: OPENAI_KEY
Value: sk-...your-key...
```

## Usage

### Basic Run (default: articulated task)

```bash
modal run launch.py
```

### Run Specific Task

```bash
modal run launch.py --task longhorizon
```

### Available Tasks

- `articulated` (default): `keypoint_pipeline_articulated_3stage` + `data_articulated/`
- `longhorizon`: `keypoint_pipeline_longhorizon_topdown` + `data_longhorizon/`

### With Custom Parameters

```bash
modal run launch.py --task longhorizon --num-tasks 5
```

Or override task folders explicitly:

```bash
modal run launch.py --task custom \
  --prompt-folder my_pipeline \
  --prompt-data-folder my_data/ \
  --num-tasks 1
```

### Parameters

- `--task`: Task preset (articulated, longhorizon). Default: `articulated`
- `--prompt-folder`: Prompt folder name in GenSim2 prompts/ (overrides task default)
- `--prompt-data-folder`: Data folder name in GenSim2 prompts/ (overrides task default)
- `--num-tasks`: Number of tasks to generate (default: 1)

## What Happens

1. Modal creates isolated environment with:
   - micromamba (instead of conda)
   - Python 3.11
   - PyTorch with CUDA 12.1
   - T4 GPU

2. GenSim2-fork cloned from GitHub with all submodules

3. Dependencies installed via micromamba/pip:
   - Core packages (iopath, pytorch3d, etc.)
   - All requirements.txt packages
   - C++ extensions built (pointnet2, chamfer_dist, pointops, etc.)

4. Pipeline executed:
   ```
   python gensim2/pipeline/run_pipeline.py prompt_folder=... prompt_data_folder=... num_tasks=...
   ```

5. Results returned with status and output paths

## Files

- `launch.py`: Main Modal app with T4 GPU function and task presets
- `README.md`: This file
- `pyproject.toml`: Project metadata

## Notes

- T4 GPU is adequate for GenSim2 pipeline tasks
- First run builds Docker image (~10 min), subsequent runs use cache
- C++ extension build failures are non-fatal
- Environment is ephemeral, exists only during function execution
- Data directories must be in GenSim2-fork repo or downloaded at runtime
