# Lab 06 Report: Reproducible, Observable, and Fault-Tolerant ML Training Pipeline

## 1. Executive Summary & System Overview
This lab transitions a Fashion-MNIST image classification model (`CNNModel`, with support for `MLPModel` and `AttentionModel` from Lab 3) from ad-hoc notebook/script execution into a structured, reproducible, and observable machine learning training pipeline[cite: 1, 2]. Grounded in ML systems engineering principles (Reddi, *Introduction to Machine Learning Systems*, Chapters 5, 8, and 13), the repository organizes configuration, training, profiling, experimentation, and visualization across a clean layout:

- `src/config.yaml`: Centralizes experiment metadata, random seeds, dataset splits, DataLoader settings, model architecture, optimizer/scheduler hyperparameters, gradient accumulation steps, and checkpointing rules.
- `src/pipeline.py`: Implements deterministic seeding, Fashion-MNIST loading/splitting, model construction, optimizer/scheduler setup, gradient accumulation, epoch checkpointing/resuming, MLflow tracking, and test evaluation.
- `src/profile_study.py`: Runs single-epoch execution profiling (`torch.profiler`), DataLoader `num_workers` throughput sweeps, and batch-size scaling/memory studies.
- `src/run_experiments.py`: Automates multi-run experiments across optimizers/schedulers, gradient accumulation configurations, and checkpoint interrupt/resume workflows.
- `src/plot_results.py`: Reads logged run histories (`history.json`) and generates visual plots (`training_curves.png`) for analysis and reporting[cite: 1].
- `proposal.md`: Contains the Task 8 one-page project proposal[cite: 1, 2].
- `requirements.txt`: Pins compatible package versions (including `numpy==1.26.4`, `torch==2.3.1`, `torchvision==0.18.1`, `mlflow==2.14.1`, `matplotlib==3.9.0`, and `psutil==5.9.8`)[cite: 3].

---

## 2. Dataset & Environment Setup
- **Dataset:** Fashion-MNIST ($28 \times 28$ single-channel grayscale images across 10 clothing classes), normalized using mean `(0.2860,)` and standard deviation `(0.3530,)`.
- **Train / Validation / Test Split:** The 60,000 training images are split deterministically (`split_seed: 42`) into **55,000 training** and **5,000 validation** samples, evaluated against the standard **10,000-sample test set**.
- **Environment & Stability Fixes:**
  - Pinned `numpy==1.26.4` alongside `torch==2.3.1` and `torchvision==0.18.1` to prevent NumPy 2.x ABI incompatibilities[cite: 3].
  - Enabled local filesystem tracking via `os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")` prior to importing `mlflow` and resolved paths relative to `Path(__file__).resolve().parent` so scripts execute reliably from any working directory.

---

## 3. Task-by-Task Breakdown: Goals & Implementation

### Task 1: YAML Configuration & Core Training Pipeline (`src/config.yaml`, `src/pipeline.py`)
- **Goal:** Eliminate hardcoded hyperparameters and manual run bookkeeping by driving training through a declarative YAML configuration file with experiment tracking and model selection.
- **What the Task Does:**
  - Loads settings from `src/config.yaml` (`experiment`, `data`, `model`, `training`, `checkpointing`) and validates that the configuration dictionary is present.
  - Sets seeds across Python's `random`, `numpy`, and `torch` (`set_seed`) for repeatable initialization and data shuffling.
  - Builds the requested architecture (`cnn`, `mlp`, or `attention`), configures `CrossEntropyLoss`, instantiates `Adam` or `SGD` (with momentum and weight decay), and attaches an optional learning-rate scheduler (`none`, `step`, `cosine`, or `plateau`).
  - Logs hyperparameters and per-epoch metrics (`train_loss`, `val_loss`, `val_acc`, `lr`, `throughput_samples_per_sec`) plus final `test_loss` and `test_acc` to MLflow, while persisting `history.json` to disk.

### Task 2: Single-Epoch Execution Profiling (`profile_one_epoch` in `src/profile_study.py`)
- **Goal:** Measure where wall-clock time is spent during a training epoch—specifically distinguishing data loading/transfer overhead from forward compute, backward gradient computation, and optimizer parameter updates.
- **What the Task Does:**
  - Wraps a full training epoch in `torch.profiler.profile` (with `ProfilerActivity.CPU` and `ProfilerActivity.CUDA` when available, plus `record_shapes=True`).
  - Times four distinct phases across batches: **data loading** (time waiting for the next batch from `train_loader`), **forward pass** (`model(x)` + loss calculation), **backward pass** (`loss.backward()`), and **optimizer step** (`optimizer.step()`), synchronizing CUDA when active.
  - Computes both seconds and percentage share for each phase and saves `one_epoch_profile.json` and `profiler_table.txt` under `results/lab6/profiling/`.

### Task 3: DataLoader Worker Scaling Study (`run_num_workers_study` in `src/profile_study.py`)
- **Goal:** Identify whether input pipeline parallelism (`num_workers`) speeds up training throughput or hits diminishing returns / worker-spawning overhead on the host machine.
- **What the Task Does:**
  - Sweeps `num_workers` across `[0, 1, 2, 4, 8]` over 100 training steps per worker setting.
  - Measures elapsed wall time and throughput (`samples / sec`), writes `num_workers_study.json`, and generates `results/lab6/profiling/num_workers_throughput.png`.

### Task 4: Batch Size Throughput, Memory & Accuracy Study (`run_batch_size_study` in `src/profile_study.py`)
- **Goal:** Evaluate the trade-off between training throughput (samples/sec), memory footprint, and generalization accuracy across mini-batch sizes.
- **What the Task Does:**
  - Tests batch sizes `[32, 64, 128, 256, 512]`, training for 5 epochs per configuration with a reset seed and fresh model/optimizer state so comparisons are fair.
  - Records throughput (`samples / sec`), peak memory usage (CUDA max allocated memory in MB when on GPU, or process RSS memory in MB via `psutil` on CPU)[cite: 3], and 5-epoch validation accuracy (`val_acc_5ep`).
  - Saves results to `results/lab6/profiling/batch_size_study.json`.

### Task 5: Optimizer & Learning-Rate Scheduler Grid (`run_task5_schedulers_and_optimizers` in `src/run_experiments.py`)
- **Goal:** Compare how different optimization algorithms and learning-rate schedules affect convergence speed, stability, and final accuracy over 10 epochs.
- **What the Task Does:**
  - Systematically runs a 10-epoch grid pairing optimizers (`sgd` with momentum `0.9` at `lr=0.01`, and `adam` at `lr=0.001`) with schedulers (`step`, `cosine`, and `plateau`).
  - Logs each run under a distinct MLflow run name (`task5_<optimizer>_<scheduler>`) with its own `history.json` and metrics.

### Task 6: Gradient Accumulation Equivalence Study (`run_task6_gradient_accumulation` in `src/run_experiments.py`)
- **Goal:** Determine whether accumulating gradients over smaller micro-batches achieves a larger effective batch size without paying the full single-step memory cost, and how its dynamics compare to a direct large batch.
- **What the Task Does:**
  - Scales loss by `1 / grad_accum_steps` (`loss = criterion(logits, y) / grad_accum`), calls `loss.backward()` on each micro-batch, and steps the optimizer every `grad_accum_steps` batches (or at the end of the epoch).
  - Compares a 5-epoch run with `batch_size=64` and `grad_accum_steps=4` (effective batch size `256`, run name `task6_bs64_accum4`) against a 5-epoch run with `batch_size=256` and `grad_accum_steps=1` (`task6_bs256_accum1`).

### Task 7: Fault Tolerance — Interrupt & Resume From Checkpoint (`run_task7_interrupt_and_resume` in `src/run_experiments.py` & `src/pipeline.py`)
- **Goal:** Ensure long-running training jobs can survive interruptions (preemption, power loss, or manual stops) and resume cleanly without losing progress or corrupting optimizer/scheduler state.
- **What the Task Does:**
  - **Interrupt Phase:** Runs a 10-epoch job configured with `stop_after_epoch: 3`, saving `ckpt_epoch_3.pt` containing `epoch`, `model_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`, and `history`.
  - **Resume Phase:** Reloads `ckpt_epoch_3.pt` via `resume_from`, restores model, optimizer, scheduler, and `history`, starts at `epoch = 4`, finishes through `epoch = 10`, and evaluates on the test set.
  - **Observed Run Results:** After resuming at Epoch 4, validation accuracy progressed from **88.96%** (Epoch 3) to **90.70%** (Epoch 9) and **90.52%** (Epoch 10), achieving **90.32%** final test accuracy[cite: 1]. Plots are generated via `src/plot_results.py` into `results/lab6/task7_interrupt_resume/training_curves.png`.

### Task 8: One-Page Project Proposal (`proposal.md`)
- **Goal:** Articulate the problem motivation, ML systems architecture, and concrete milestones in a concise standalone proposal aligned with ML systems design principles[cite: 1, 2].
- **What the Task Does:** Documents the system scope, architectural choices across configuration/profiling/training/fault-tolerance, and deliverables in `lab06/proposal.md`[cite: 1, 2].

---

## 4. Results, Visualizations & Observations

### Recorded Task 7 Progression (Epochs 3–10)
| Epoch | Val Accuracy (%) | Notes |
| :---: | :---: | :--- |
| 3 | 88.96% | Checkpoint `ckpt_epoch_3.pt` saved (`stop_after_epoch=3`)[cite: 1] |
| 4 | 89.44% | Resumed cleanly from `ckpt_epoch_3.pt`[cite: 1] |
| 5 | 89.40% | — |
| 6 | 89.98% | — |
| 7 | 90.20% | — |
| 8 | 90.40% | — |
| 9 | 90.70% | Peak validation accuracy in observed run |
| 10 | 90.52% | **Final Test Accuracy: 90.32%**[cite: 1] |

### Plots & Artifacts Checklist
- [ ] `results/lab6/profiling/one_epoch_profile.json` & `profiler_table.txt` (Task 2 breakdown)
- [ ] `results/lab6/profiling/num_workers_throughput.png` (Task 3 worker scaling curve)
- [ ] `results/lab6/profiling/batch_size_study.json` (Task 4 batch-size throughput/memory/accuracy table)
- [ ] Task 5 & Task 6 MLflow runs and `history.json` comparisons
- [x] `results/lab6/task7_interrupt_resume/training_curves.png` (Task 7 training curves)

### Key Takeaways & Observations
- **Fault-Tolerant Continuity:** Saving full training state (`model`, `optimizer`, `scheduler`, and `history`) allowed seamless resumption at Epoch 4 and strong test accuracy (**90.32%**)[cite: 1].
- **Systems Bottleneck Analysis:** Comparing `one_epoch_profile.json`, `num_workers_throughput.png`, and `batch_size_study.json` highlights where throughput plateaus (data loading vs. compute) and how larger batch sizes trade memory for step efficiency.
