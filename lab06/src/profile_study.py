import copy
import json
import time
from pathlib import Path
from typing import Dict, List
import os

import matplotlib.pyplot as plt
import psutil
import torch
import torch.nn as nn
import yaml
from torch.profiler import ProfilerActivity, profile, record_function

from pipeline import build_model, build_optimizer, evaluate, get_dataloaders, resolve_path, set_seed


def profile_one_epoch(cfg: Dict, script_dir: Path) -> Dict:
    set_seed(int(cfg["experiment"]["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, _, _ = get_dataloaders(cfg, script_dir)
    model = build_model(cfg).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, cfg)

    out_dir = resolve_path(script_dir, cfg["experiment"]["output_dir"]) / "profiling"
    out_dir.mkdir(parents=True, exist_ok=True)

    data_time = 0.0
    fwd_time = 0.0
    bwd_time = 0.0
    opt_time = 0.0

    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    model.train()
    with profile(activities=activities, record_shapes=True) as prof:
        t_prev = time.perf_counter()
        for x, y in train_loader:
            t_data = time.perf_counter()
            data_time += t_data - t_prev

            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            with record_function("forward_pass"):
                t0 = time.perf_counter()
                logits = model(x)
                loss = criterion(logits, y)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                fwd_time += time.perf_counter() - t0

            with record_function("backward_pass"):
                t1 = time.perf_counter()
                loss.backward()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                bwd_time += time.perf_counter() - t1

            with record_function("optimizer_step"):
                t2 = time.perf_counter()
                optimizer.step()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                opt_time += time.perf_counter() - t2

            t_prev = time.perf_counter()

    total_measured = max(1e-6, data_time + fwd_time + bwd_time + opt_time)
    breakdown = {
        "data_loading_pct": round(100.0 * data_time / total_measured, 2),
        "forward_pass_pct": round(100.0 * fwd_time / total_measured, 2),
        "backward_pass_pct": round(100.0 * bwd_time / total_measured, 2),
        "optimizer_step_pct": round(100.0 * opt_time / total_measured, 2),
        "data_loading_sec": round(data_time, 4),
        "forward_pass_sec": round(fwd_time, 4),
        "backward_pass_sec": round(bwd_time, 4),
        "optimizer_step_sec": round(opt_time, 4),
    }

    (out_dir / "one_epoch_profile.json").write_text(json.dumps(breakdown, indent=2), encoding="utf-8")
    (out_dir / "profiler_table.txt").write_text(
        prof.key_averages().table(sort_by="cpu_time_total", row_limit=15), encoding="utf-8"
    )
    print("Task 2 Profiler Breakdown:", breakdown)
    return breakdown


def run_num_workers_study(cfg: Dict, script_dir: Path, workers: List[int] = None) -> List[Dict]:
    if workers is None:
        workers = [0, 1, 2, 4, 8]

    set_seed(int(cfg["experiment"]["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, cfg)

    out_dir = resolve_path(script_dir, cfg["experiment"]["output_dir"]) / "profiling"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for w in workers:
        train_loader, _, _ = get_dataloaders(cfg, script_dir, num_workers_override=w)
        model.train()
        seen = 0
        t0 = time.perf_counter()
        for step, (x, y) in enumerate(train_loader):
            if step >= 100:
                break
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            seen += x.size(0)
        elapsed = max(1e-6, time.perf_counter() - t0)
        throughput = seen / elapsed
        row = {"num_workers": w, "throughput_samples_per_sec": round(throughput, 2), "elapsed_sec": round(elapsed, 3)}
        rows.append(row)
        print("Task 3 Worker Sweep:", row)

    (out_dir / "num_workers_study.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    plt.figure(figsize=(7, 4))
    plt.plot([r["num_workers"] for r in rows], [r["throughput_samples_per_sec"] for r in rows], marker="o")
    plt.title("Throughput vs. num_workers")
    plt.xlabel("num_workers")
    plt.ylabel("Samples / sec")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "num_workers_throughput.png", dpi=150)
    plt.close()

    return rows


def run_batch_size_study(cfg: Dict, script_dir: Path, batch_sizes: List[int] = None) -> List[Dict]:
    if batch_sizes is None:
        batch_sizes = [32, 64, 128, 256, 512]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    process = psutil.Process(os.getpid()) if "os" in globals() else psutil.Process()

    out_dir = resolve_path(script_dir, cfg["experiment"]["output_dir"]) / "profiling"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for bs in batch_sizes:
        set_seed(int(cfg["experiment"]["seed"]))
        train_loader, val_loader, _ = get_dataloaders(cfg, script_dir, batch_size_override=bs)
        model = build_model(cfg).to(device)
        optimizer = build_optimizer(model, cfg)

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)

        seen = 0
        t0 = time.perf_counter()
        for _ in range(5):
            model.train()
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                loss = criterion(model(x), y)
                loss.backward()
                optimizer.step()
                seen += x.size(0)
        elapsed = max(1e-6, time.perf_counter() - t0)
        throughput = seen / elapsed
        _, val_acc = evaluate(model, val_loader, criterion, device)

        if torch.cuda.is_available():
            peak_mem_mb = round(torch.cuda.max_memory_allocated(device) / (1024 ** 2), 2)
        else:
            peak_mem_mb = round(process.memory_info().rss / (1024 ** 2), 2)

        row = {
            "batch_size": bs,
            "throughput_samples_per_sec": round(throughput, 2),
            "peak_memory_mb": peak_mem_mb,
            "val_acc_5ep": round(val_acc, 2),
        }
        rows.append(row)
        print("Task 4 Batch Size Study:", row)

    (out_dir / "batch_size_study.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return rows


if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    cfg_path = script_dir / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict) or "experiment" not in cfg:
        raise ValueError(f"Config at {cfg_path} is empty or invalid.")

    profile_one_epoch(cfg, script_dir)
    run_num_workers_study(cfg, script_dir, [0, 1, 2, 4, 8])
    run_batch_size_study(cfg, script_dir, [32, 64, 128, 256, 512])