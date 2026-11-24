# Project Proposal: Reproducible & Fault-Tolerant 100m Dash Laser & Image Speed-Tracking Pipeline

## 1. Problem Statement
Measuring a sprinter's speed, split times, and acceleration across a 100-meter dash using laser gates/sensors and captured image frames requires both reliable velocity estimation and disciplined ML systems engineering. In practice, training and evaluation pipelines often fail due to unlogged configuration changes, non-reproducible dataset splits or random seeds, hidden bottlenecks between frame loading and model compute, memory limits on high-resolution capture batches, and lost progress when long training runs are interrupted. Grounded in Reddi's *Introduction to Machine Learning Systems* (Chapters 5, 8, and 13), this project defines a reproducible, observable, and fault-tolerant pipeline for sprint frame and laser speed analysis.

## 2. Dataset & Preprocessing Pipeline
- **Data Sources & Splits:** Combines synchronized laser split/position timestamps with sequential camera frames captured along the 100m track (alongside benchmark validation fixtures during development), partitioned deterministically into training, validation, and test splits using fixed seeds configured in `src/config.yaml`.
- **Preprocessing & Augmentation:** Synchronizes laser trigger timestamps with frame indices, normalizes pixel intensities and spatial dimensions, applies controlled frame cropping/jitter/brightness adjustments without distorting track distance markers, and feeds batches through a configurable `DataLoader` (`num_workers`, `batch_size`) tracked via MLflow.
- **Data Integrity & Reproducibility:** Pins environment dependencies in `requirements.txt`, logs dataset split counts and config hashes per run, and enforces deterministic seeding across Python, NumPy, and PyTorch.

## 3. Success Metrics
- **Accuracy & Estimation Quality:** Track validation and test accuracy/error metrics across training epochs (targeting strong convergence past epoch 3 up through epoch 10 without overfitting), comparing `CNNModel`, `MLPModel`, and `AttentionModel` variants.
- **Throughput & Stage Latency:** Measure samples/sec and per-epoch wall time using `torch.profiler` broken down across data loading, forward pass, backward pass, and optimizer step stages across `num_workers` (`[0, 1, 2, 4, 8]`) and batch sizes (`[32, 64, 128, 256, 512]`).
- **Optimization & Fault-Tolerance Reliability:** Evaluate convergence across `SGD + momentum` and `Adam` paired with `StepLR`, `CosineAnnealingLR`, and `ReduceLROnPlateau`, verify gradient accumulation (`batch_size=64`, `accum=4` vs. `batch_size=256`), and confirm zero-loss resume after interrupting training at epoch 3.

## 4. Deployment Constraints & Compliance
- **Timing & Hardware Constraints:** A 100m dash concludes in roughly 10–15 seconds, requiring low-latency inference and predictable memory usage so post-sprint feedback is timely and local/edge hardware does not run out of memory on high-resolution frame batches.
- **Operational Reliability & Fault Tolerance:** Training and inference jobs must tolerate interruptions, process/worker overhead on desktop/edge OS environments, and recover cleanly from saved checkpoints.
- **Safety, Privacy & Track Compliance:** Adhere to athletic event rules, physical laser/camera safety boundaries on the track, and runner privacy/consent guidelines when storing or reviewing captured sprint imagery and performance telemetry.

## 5. Planned Optimization Strategy
- **Bottleneck-Driven Profiling First (Ch. 5 & 13):** Profile stage timings and sweep `num_workers` and batch sizes before tuning architectures so optimization targets real bottlenecks (frame I/O vs. forward/backward compute).
- **Optimizer, Scheduler & Batch Scaling (Ch. 8):** Benchmark `SGD + momentum` and `Adam` across `StepLR`, `CosineAnnealingLR`, and `ReduceLROnPlateau` over 10 epochs, and use gradient accumulation (`batch_size=64`, `accum=4` vs. `batch_size=256`) to achieve larger effective batch stability within tight memory budgets.
- **Checkpoint-Backed Fault Recovery (Ch. 13):** Save model weights, optimizer/scheduler states, epoch index, and RNG states per epoch so an interrupt at epoch 3 resumes seamlessly with identical trajectory and logged MLflow lineage.
