import copy
from pathlib import Path
import yaml

from pipeline import train_pipeline


def run_task5_schedulers_and_optimizers(base_cfg: dict) -> None:
    """Task 5: StepLR, CosineAnnealingLR, ReduceLROnPlateau with SGD+momentum and Adam (10 epochs)."""
    for opt_name, lr in [("sgd", 0.05), ("adam", 0.001)]:
        for sched in ["step", "cosine", "plateau"]:
            cfg = copy.deepcopy(base_cfg)
            cfg["training"]["optimizer"] = opt_name
            cfg["training"]["lr"] = lr
            cfg["training"]["scheduler"] = sched
            cfg["training"]["epochs"] = 10
            cfg["experiment"]["run_name"] = f"task5_{opt_name}_{sched}"
            train_pipeline(cfg)


def run_task6_gradient_accumulation(base_cfg: dict) -> None:
    """Task 6: Emulate effective batch 256 (batch_size=64, grad_accum_steps=4) vs direct batch 256."""
    cfg_direct = copy.deepcopy(base_cfg)
    cfg_direct["data"]["batch_size"] = 256
    cfg_direct["training"]["grad_accum_steps"] = 1
    cfg_direct["experiment"]["run_name"] = "task6_direct_bs256"
    train_pipeline(cfg_direct)

    cfg_accum = copy.deepcopy(base_cfg)
    cfg_accum["data"]["batch_size"] = 64
    cfg_accum["training"]["grad_accum_steps"] = 4
    cfg_accum["experiment"]["run_name"] = "task6_accum4_bs64"
    train_pipeline(cfg_accum)


def run_task7_interrupt_and_resume(base_cfg: dict) -> None:
    """Task 7: Stop at epoch 3 and resume from checkpoint through epoch 10."""
    cfg_stop = copy.deepcopy(base_cfg)
    cfg_stop["training"]["epochs"] = 10
    cfg_stop["checkpointing"]["stop_after_epoch"] = 3
    cfg_stop["experiment"]["run_name"] = "task7_interrupt_resume"
    train_pipeline(cfg_stop)

    ckpt_ep3 = str(
        Path(base_cfg["experiment"]["output_dir"])
        / "task7_interrupt_resume/checkpoints/ckpt_epoch_3.pt"
    )
    cfg_resume = copy.deepcopy(base_cfg)
    cfg_resume["training"]["epochs"] = 10
    cfg_resume["checkpointing"]["stop_after_epoch"] = None
    cfg_resume["checkpointing"]["resume_from"] = ckpt_ep3
    cfg_resume["experiment"]["run_name"] = "task7_interrupt_resume"
    train_pipeline(cfg_resume)


if __name__ == "__main__":
    cfg_path = Path(__file__).resolve().parent / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict) or "experiment" not in cfg:
        raise ValueError(f"Config at {cfg_path} is empty or invalid. Paste the YAML config and save (Ctrl+S).")

    # Uncomment whichever experiment you want to run in VS Code:
    # run_task5_schedulers_and_optimizers(cfg)
    # run_task6_gradient_accumulation(cfg)
    run_task7_interrupt_and_resume(cfg)