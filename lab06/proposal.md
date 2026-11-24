# Project Proposal: Reproducible & Resource-Aware CIFAR-10 Training Pipeline

## 1. Problem Statement & Motivation
Training deep neural networks on multi-class image benchmarks such as CIFAR-10 (60,000 $32 \times 32$ color RGB images across 10 object classes, split into 45,000 train, 5,000 validation, and 10,000 test samples) frequently encounters unlogged experiment runs, irreproducible random initialization or splits, and I/O or compute bottlenecks across data loading, forward/backward passes, and parameter updates. Grounded in Reddi's *Introduction to Machine Learning Systems* (Chapters 5, 8, and 13), this project defines a modular, observable, and fault-tolerant training pipeline.

## 2. System Architecture & ML Systems Alignment
- **Configuration & Reproducibility (Ch. 5 & 8):** All hyperparameters, seeds, dataset split sizes, model choices (`CNNModel`, `MLPModel`, `AttentionModel` adapted for 3-channel $32 \times 32$ inputs), batch size, learning rate, optimizer, learning-rate scheduler, gradient accumulation steps, and checkpoint options are centralized in `src/config.yaml` and tracked via MLflow.
- **Execution Profiling & Throughput Bottlenecks (Ch. 5 & 13):** Uses `torch.profiler` to measure per-epoch time across data loading, forward pass, backward pass, and optimizer updates, combined with sweeps over `num_workers` (`[0, 1, 2, 4, 8]`) and batch sizes (`[32, 64, 128, 256, 512]`) to separate input pipeline delays from compute or memory limits.
- **Optimization Dynamics & Fault Tolerance (Ch. 8 & 13):** Compares `SGD + momentum` and `Adam` across `StepLR`, `CosineAnnealingLR`, and `ReduceLROnPlateau` over 10 epochs, evaluates gradient accumulation (`batch_size=64`, `accum=4` vs. `batch_size=256`), and validates checkpoint save/resume behavior after interrupting at epoch 3.

## 3. Milestones & Deliverables
1. **Pipeline & Tracking:** YAML-driven `src/pipeline.py` with MLflow experiment logging and epoch checkpointing.
2. **Profiling & Scaling Studies:** `src/profile_study.py` outputs covering profiler stage breakdowns, `num_workers` throughput curves, and batch-size trade-offs.
3. **Training & Fault-Tolerance Experiments:** `src/run_experiments.py` results for optimizer/scheduler combinations, gradient accumulation, and epoch-3 interrupt/resume.
4. **Documentation & Dependencies:** Final `report.md` with plots, `proposal.md`, and pinned `requirements.txt` in `lab06/`.
