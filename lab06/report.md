# Lab 06 Report

This report documents the implementation and findings for Lab 06, covering YAML-driven pipeline orchestration, PyTorch profiling, hardware throughput studies, optimizer/scheduler evaluations, gradient accumulation, checkpoint resumption, and the final individual project proposal outline.

## Summary
1. YAML-Driven Training Pipeline: Restructured the best model from Lab 03 into an automated pipeline driven by a centralized YAML configuration file, complete with deterministic seeding, checkpoint saving/resuming, and comprehensive MLflow experiment tracking (logging parameters, metrics, and artifacts).
2. PyTorch Profiler Integration: Profiled a full training epoch using torch.profiler to break down time distribution across data loading, forward passes, backward passes, and optimizer steps.
3. Worker Count Bottleneck Study: Benchmarked varying data loader worker configurations (num_workers in [0, 1, 2, 4, 8]) to identify I/O bottlenecks and throughput limits.
4. Batch Size & Hardware Scaling Study: Evaluated batch sizes in [32, 64, 128, 256, 512], recording throughput (samples/sec), peak memory consumption, and convergence accuracy after 5 epochs.
5. Optimizer & Scheduler Evaluations: Compared learning rate schedules (StepLR, CosineAnnealingLR, ReduceLROnPlateau) alongside optimizers (SGD + momentum vs. Adam) under a standardized 10-epoch budget.
6. Gradient Accumulation Verification: Implemented gradient accumulation to simulate an effective batch size of 256 using physical mini-batches of size 64 with 4 accumulation steps, validating alignment with direct large-batch training.
7. Checkpoint Resumption & Continuity: Interrupted training cleanly at epoch 3, reloaded from the saved checkpoint file, and verified that training curves and optimizer states resumed smoothly without discontinuity.
8. Individual Project Proposal: Prepared the one-page project proposal covering the problem definition, target dataset, success metrics, deployment constraints ($\le 50\text{ ms}$ latency, $\le 5\text{ MB}$ footprint), and planned optimizations (quantization/pruning).   
