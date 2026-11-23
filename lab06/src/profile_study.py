import copy
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.profiler import ProfilerActivity, profile, record_function
import yaml

from pipeline import build_model, build_optimizer, evaluate, get_dataloaders, set_seed


def profile_one_epoch(cfg: dict) -> Dict[str, float]:
    """Task 2: Share of time on data loading, forward, backward, optimizer step."""
    set_seed(int(cfg["experiment"]["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, _, _ = get_dataloaders(cfg)
    model = build_model(cfg, device)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(cfg, model)

    timings = {"data_loading": 0.0, "forward": 0.0, "backward": 0.0, "optimizer_step": 0.0}
    activities = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if torch.cuda.is_available() else [])

    model.train()
    with profile(activities=activities, record_shapes=True) as prof:
        t_iter_start = time.perf_counter()
        for inputs, targets in train_loader:
            timings["data_loading"] += time.perf_counter() - t_iter_start

            with record_function("ForwardPass"):
                t0 = time.perf_counter()
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(inputs), targets)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                timings["forward"] += time.perf_counter() - t0

            with record_function("BackwardPass"):
                t0 = time.perf_counter()
                loss.backward()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                timings["backward"] += time.perf_counter() - t0

            with record_function("OptimizerStep"):
                t0 = time.perf_counter()
                optimizer.step()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                timings["optimizer_step"] += time.perf_counter() - t0

            t_iter_start = time.perf_counter()

    total = sum(timings.values()) or 1.0
    shares = {k: (v / total) * 100.0 for k, v in timings.items()}
    print("\n=== Task 2: One-Epoch Stage Breakdown ===")
    for k, pct in shares.items():
        print(f"  {k:16s}: {pct:6.2f}% ({timings[k]:.3f}s)")
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))
    return shares


def run_num_workers_study(base_cfg: dict, workers_list: List[int] = None) -> List[Tuple[int, float]]:
    """Task 3: Vary num_workers [0, 1, 2, 4, 8], find bottleneck, and plot."""
    if workers_list is None:
        workers_list = [0, 1, 2, 4, 8]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results: List[Tuple[int, float]] = []

    for w in workers_list:
        cfg = copy.deepcopy(base_cfg)
        cfg["data"]["num_workers"] = w
        set_seed(int(cfg["experiment"]["seed"]))
        train_loader, _, _ = get_dataloaders(cfg)
        model = build_model(cfg, device)
        criterion = nn.CrossEntropyLoss()
        optimizer = build_optimizer(cfg, model)

        model.train()
        seen, t0 = 0, time.perf_counter()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()
            seen += targets.size(0)

        throughput = seen / max(time.perf_counter() - t0, 1e-6)
        results.append((w, throughput))
        print(f"num_workers={w:2d} -> {throughput:8.1f} samples/s")

    out_dir = Path(base_cfg["experiment"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 4.5))
    plt.plot([r[0] for r in results], [r[1] for r in results], marker="o")
    plt.title(f"Fashion-MNIST Throughput vs. num_workers ({base_cfg['model']['arch'].upper()})")
    plt.xlabel("num_workers")
    plt.ylabel("Throughput (samples/s)")
    plt.grid(True)
    plot_path = out_dir / "num_workers_throughput.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved plot: {plot_path}")
    return results


def run_batch_size_study(base_cfg: dict, batch_sizes: List[int] = None) -> List[dict]:
    """Task 4: Batch sizes [32, 64, 128, 256, 512] over 5 epochs."""
    if batch_sizes is None:
        batch_sizes = [32, 64, 128, 256, 512]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    rows = []

    for bs in batch_sizes:
        cfg = copy.deepcopy(base_cfg)
        cfg["data"]["batch_size"] = bs
        cfg["training"]["epochs"] = 5
        cfg["experiment"]["run_name"] = f"bs_{bs}"

        set_seed(int(cfg["experiment"]["seed"]))
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)

        train_loader, val_loader, _ = get_dataloaders(cfg)
        model = build_model(cfg, device)
        optimizer = build_optimizer(cfg, model)

        total_samples, total_time, val_acc = 0, 0.0, 0.0
        for _ in range(5):
            model.train()
            t0 = time.perf_counter()
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(inputs), targets)
                loss.backward()
                optimizer.step()
                total_samples += targets.size(0)
            total_time += time.perf_counter() - t0
            _, val_acc = evaluate(model, val_loader, criterion, device)

        peak_mem_mb = (
            torch.cuda.max_memory_allocated(device) / (1024**2) if torch.cuda.is_available() else 0.0
        )
        row = {
            "batch_size": bs,
            "throughput_samples_per_sec": round(total_samples / max(total_time, 1e-6), 2),
            "peak_memory_mb": round(peak_mem_mb, 2),
            "final_val_acc_ep5": round(val_acc, 2),
        }
        rows.append(row)
        print(row)

    out_dir = Path(base_cfg["experiment"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "batch_size_study.json").write_text(json.dumps(rows, indent=2))
    return rows

if __name__ == "__main__":
    cfg_path = Path(__file__).resolve().parent / "config.yaml"
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    profile_one_epoch(cfg)
    run_num_workers_study(cfg, [0, 1, 2, 4, 8])
    run_batch_size_study(cfg, [32, 64, 128, 256, 512])
    