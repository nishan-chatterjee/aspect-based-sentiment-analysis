# HPC Tasks For Comparison Baselines

This directory adds a split-level parallel execution layer without changing the
current `reviews/` training scripts. It is intended for sharing with a colleague
who can run many one-GPU jobs on SLURM or inside Apptainer.

## Container

Build the image from the repository root:

```bash
bash hpc-tasks/build_apptainer.sh
```

The default image path is:

```text
hpc-tasks/absa-comparisons.sif
```

If the cluster does not allow building images, build it on a workstation with
Apptainer and copy the `.sif` to the cluster.

Run a quick GPU check:

```bash
apptainer exec --nv \
  --bind "$PWD:/workspace" \
  hpc-tasks/absa-comparisons.sif \
  python - <<'PY'
import torch
print(torch.__version__, torch.version.cuda)
print(torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
PY
```

## Task Matrix

Create all tasks:

```bash
python hpc-tasks/make_task_matrix.py --task-set all --output hpc-tasks/tasks_all.tsv
```

`all` creates 48 one-GPU tasks:

- 4 approaches: `longformer`, `mdeberta`, `mt5`, `slavic_specific`
- 2 languages
- 2 masking variants
- 3 train/validation split runs

For only the additional-comparison three main approaches, use:

```bash
python hpc-tasks/make_task_matrix.py --task-set core --output hpc-tasks/tasks_core.tsv
```

## Run One Task

With Apptainer:

```bash
TASK_FILE=hpc-tasks/tasks_all.tsv bash hpc-tasks/run_one_task_apptainer.sh 0
```

Without Apptainer, using a local conda env:

```bash
conda activate absa
PYTHON_BIN="$(which python)" TASK_FILE=hpc-tasks/tasks_all.tsv bash hpc-tasks/run_one_task.sh 0
```

Each task trains exactly one `(approach, language, variant, run_index)` and
writes disjoint files such as `best_model_0.pt`, `training_metrics_0.json`, and
`test_predictions_0.json`.

## Run A Local Interactive GPU Queue

For an interactive allocation with several visible GPUs, use the queue launcher:

```bash
CONDA_ENV=absa GPU_IDS=0,1,2,3 TASK_SET=all bash hpc-tasks/run_interactive_gpu_queue.sh
```

This keeps all GPUs busy by scheduling one split-run at a time. When a short
Slovenian task finishes, that GPU immediately receives the next pending task
instead of waiting for longer Serbian tasks.

Resume behavior is conservative. A split-run is skipped only when both files
exist and the training metrics contain the requested number of train/eval
epochs:

```text
reviews/<approach>/<variant>/<language>/best_model_<run>.pt
reviews/<approach>/<variant>/<language>/training_metrics_<run>.json
```

So a stray checkpoint from an interrupted run is not treated as complete.

## Submit To SLURM

Submit all tasks, allowing at most 8 concurrent GPU jobs:

```bash
MAX_PARALLEL=8 TASK_SET=all USE_APPTAINER=1 bash hpc-tasks/submit_array.sh
```

Use `MAX_PARALLEL=N` to scale to any number of available GPUs. The scheduler
maps each array task to one GPU via `#SBATCH --gres=gpu:1`.

If you do not want Apptainer:

```bash
MAX_PARALLEL=8 TASK_SET=all USE_APPTAINER=0 CONDA_ENV=absa bash hpc-tasks/submit_array.sh
```

## Merge Results

After all array tasks finish, rebuild notebook-compatible summaries:

```bash
python hpc-tasks/merge_results.py --output-root reviews
```

This recomputes `test_metrics_summary.json` from `test_predictions_*.json`, so
it is robust to concurrent single-run tasks overwriting temporary summaries.

## Progress And Logs

By default, `run_one_task.sh` sets:

```bash
TQDM_DISABLE=1
```

This keeps SLURM logs compact: you still get epoch summaries with train loss,
validation macro-F1, and QWK, but not thousands of batch progress updates.

To re-enable live tqdm bars:

```bash
TQDM_DISABLE=0 bash hpc-tasks/run_one_task_apptainer.sh 0
```

To summarize progress from metrics files:

```bash
python hpc-tasks/progress_report.py --output-root reviews
```

## Directory Assumptions

The container/task scripts assume the shared project directory contains:

```text
data/
models/
reviews/
hpc-tasks/
```

When using Apptainer, the project root is bound to `/workspace` inside the
container.
