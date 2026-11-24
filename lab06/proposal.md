# Project Proposal: Reproducible & Fault-Tolerant 100m Dash Laser & Frame Speed-Tracking Pipeline

## 1. Problem Statement & Motivation
Measuring a sprinter's speed, split times, and acceleration across a 100-meter dash using laser gates/sensors and captured image frames requires both accurate estimation and dependable systems engineering. In practice, training and evaluation pipelines often suffer from unlogged parameter changes, non-reproducible random splits or initialization, hidden bottlenecks between frame loading and model compute, memory limits on high-resolution capture batches, and lost progress when long training jobs are interrupted. Grounded in Reddi's *Introduction to Machine Learning Systems* (Chapters 5, 8, and 13), this project defines a reproducible, observable, and fault-tolerant pipeline for sprint frame/laser speed analysis.

## 2. System Architecture & ML Systems Alignment
- **Configuration & Reproducibility (Ch. 5 & 8):** All experiment parameters—random seeds, train/validation/test split sizes, model architecture (`CNNModel`, `MLPModel`, `AttentionModel` adapted to sprint frame/sensor inputs), batch size, learning rate, optimizer, learning-rate scheduler, gradient accumulation steps, and checkpoint settings—are centralized in `src/config.yaml` and tracked with MLflow alongside pinned dependencies in `requirements.txt`.
- **Execution Profiling & Throughput Bottlenecks (Ch. 5 & 13):** Uses `torch.profiler` to break down execution time across data/frame loading, forward pass, backward pass, and optimizer updates, paired with sweeps over `num_workers` (`[0, 1, 2, 4, 8]`) and batch sizes (`[32, 64, 128, 256, 512]`) to identify whether latency comes from disk/sensor I/O or model compute.
- **Optimization Dynamics & Fault Tolerance (Ch. 8 & 13):** Compares `SGD + momentum` and `Adam` across `StepLR`, `CosineAnnealingLR`, and `ReduceLROnPlateau` over 10 epochs, evaluates gradient accumulation (`batch_size=64`, `accum=4` vs. `batch_size=256`) to maintain larger effective batch updates under memory constraints, and verifies checkpoint save/resume behavior after interrupting training at epoch 3.

## 3. Milestones & Deliverables
1. **Core Pipeline & Experiment Tracking:** YAML-driven `src/pipeline.py` supporting reproducible seeding, MLflow run logging, and epoch checkpointing.
2. **Profiling & Scaling Study:** `src/profile_study.py` outputs detailing stage-level timing breakdowns, `num_workers` throughput scaling, and batch-size trade-offs.
3. **Training & Fault-Tolerance Experiments:** `src/run_experiments.py` results across optimizers, schedulers, gradient accumulation configurations, and epoch-3 interrupt/resume recovery.
4. **Final Documentation & Environment:** Completed `report.md` with charts and systems analysis, `proposal.md`, and pinned `requirements.txt`.
