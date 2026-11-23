import copy
from pathlib import Path
from typing import Dict

import yaml

from pipeline import train_pipeline


def run_task5_schedulers_and_optimizers(base_cfg: Dict, script_dir: Path) -> None:
    optimizers = ["sgd", "adam"]
    schedulers = ["step", "cosine", "plateau"]

    for opt in optimizers:
        for sched in schedulers:
            cfg = copy.deepcopy(base_cfg)
            cfg["training"]["epochs"] = 10
            cfg["training"]["optimizer"] = opt
            cfg["training"]["scheduler"] = sched
            cfg["training"]["lr"] = 0.01 if opt == "sgd" else 0.001
            cfg["experiment"]["run_name"] = f"task5_{opt}_{sched}"
            train_pipeline(cfg, script_dir=script_dir)


def run_task6_gradient_accumulation(base_cfg: Dict, script_dir: Path) -> None:
    # Micro-batch 64 with accum=4 (effective batch 256)
    cfg_accum = copy.deepcopy(base_cfg)
    cfg_accum["data"]["batch_size"] = 64
    cfg_accum["training"]["grad_accum_steps"] = 4
    cfg_accum["training"]["epochs"] = 5
    cfg_accum["experiment"]["run_name"] = "task6_bs64_accum4"
    train_pipeline(cfg_accum, script_dir=script_dir)

    # Direct batch 256 with accum=1
    cfg_direct = copy.deepcopy(base_cfg)
    cfg_direct["data"]["batch_size"] = 256
    cfg_direct["training"]["grad_accum_steps"] = 1
    cfg_direct["training"]["epochs"] = 5
    cfg_direct["experiment"]["run_name"] = "task6_bs256_accum1"
    train_pipeline(cfg_direct, script_dir=script_dir)


def run_task7_interrupt_and_resume(base_cfg: Dict, script_dir: Path) -> None:
    run_name = "task7_interrupt_resume"

    cfg_stop = copy.deepcopy(base_cfg)
    cfg_stop["training"]["epochs"] = 10
    cfg_stop["checkpointing"]["stop_after_epoch"] = 3
    cfg_stop["checkpointing"]["resume_from"] = None
    cfg_stop["experiment"]["run_name"] = run_name
    train_pipeline(cfg_stop, script_dir=script_dir)

    cfg_resume = copy.deepcopy(base_cfg)
    cfg_resume["training"]["epochs"] = 10
    cfg_resume["checkpointing"]["stop_after_epoch"] = None
    cfg_resume["checkpointing"]["resume_from"] = f"../results/lab6/{run_name}/checkpoints/ckpt_epoch_3.pt"
    cfg_resume["experiment"]["run_name"] = run_name
    train_pipeline(cfg_resume, script_dir=script_dir)


if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    cfg_path = script_dir / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict) or "experiment" not in cfg:
        raise ValueError(f"Config at {cfg_path} is empty or invalid.")

    # Uncomment whichever experiment you want to run:
    # run_task5_schedulers_and_optimizers(cfg, script_dir)
    # run_task6_gradient_accumulation(cfg, script_dir)
    run_task7_interrupt_and_resume(cfg, script_dir)