# Project Proposal: Reproducible & Resource-Aware Fashion-MNIST Training Pipeline

## 1. Problem Statement & Motivation
Training deep neural networks on image classification benchmarks like Fashion-MNIST ($28 \times 28$ grayscale images across 10 classes, split into 55,000 train, 5,000 validation, and 10,000 test samples) often suffers from unlogged hyperparameters, irreproducible runs, and hardware bottlenecks across data loading, forward/backward passes, and optimizer updates. Following Reddi's *Introduction to Machine Learning Systems* (Chapters 5, 8, and 13), this project establishes a configurable, observable, and fault-tolerant training pipeline.

## 2. System Architecture & ML Systems Alignment
- **Configuration & Reproducibility (Ch. 5 & 8):** All experiment settings—seeds, dataset splits, model architecture (`CNNModel`, `MLPModel`, `AttentionModel` from Lab 3), batch size, learning rate, optimizer, scheduler, gradient accumulation, and checkpoint controls—are externalized in `src/config.yaml` and tracked via MLflow.
- **Execution Profiling & Throughput Bottlenecks (Ch. 5 & 13):** Uses `torch.profiler` for single-epoch time breakdown across data loading, forward pass, backward pass, and optimizer steps, paired with systematic sweeps over `num_workers` (`[0, 1, 2, 4, 8]`) and batch sizes (`[32, 64, 128, 256, 512]`) to identify compute vs. I/O bottlenecks and throughput/memory trade-offs.
- **Optimization Dynamics & Fault Tolerance (Ch. 8 & 13):** Evaluates `SGD + momentum` and `Adam` across `StepLR`, `CosineAnnealingLR`, and `ReduceLROnPlateau` over 10 epochs, tests gradient accumulation (`batch_size=64`, `accum=4` vs. `batch_size=256`), and verifies checkpoint save/resume continuity after interrupting at epoch 3.

## 3. Milestones & Deliverables
1. **Pipeline & Tracking:** YAML-driven `src/pipeline.py` with MLflow logging and epoch checkpointing.
2. **Profiling & Scaling Studies:** `src/profile_study.py` outputs for profiler breakdown, `num_workers` throughput plot, and batch-size metrics.
3. **Training & Fault-Tolerance Experiments:** `src/run_experiments.py` results for optimizer/scheduler pairs, gradient accumulation, and epoch-3 interrupt/resume.
4. **Documentation & Dependencies:** Final `report.md` with visual plots, `proposal.md`, and pinned `requirements.txt` committed in `lab06/`.
