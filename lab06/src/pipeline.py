import json
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms



def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --- Lab 3 Model Definitions ---
class MLPModel(nn.Module):
    def __init__(self):
        super(MLPModel, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(784, 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.fc(x)
        return x


class AttentionModel(nn.Module):
    def __init__(self):
        super(AttentionModel, self).__init__()
        self.patch_size = 7
        self.d_model = 64
        self.linear_proj = nn.Linear(49, self.d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, nhead=4, dim_feedforward=128, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(self.d_model, 10)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()
        x = x.view(B, 16, C * self.patch_size * self.patch_size)

        x = self.linear_proj(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        x = self.fc(x)
        return x


def build_model(cfg: dict, device: torch.device) -> nn.Module:
    arch = cfg["model"].get("arch", "cnn").lower()
    if arch == "cnn":
        model: nn.Module = CNNModel()
    elif arch == "mlp":
        model = MLPModel()
    elif arch in ("attention", "transformer"):
        model = AttentionModel()
    else:
        raise ValueError(f"Unsupported architecture '{arch}'. Choose from: cnn, mlp, attention.")
    return model.to(device)


def get_dataloaders(cfg: dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ])

    data_dir = cfg["data"].get("data_dir", "./data")
    full_train_dataset = torchvision.datasets.FashionMNIST(
        root=data_dir, train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.FashionMNIST(
        root=data_dir, train=False, download=True, transform=transform
    )

    train_size = int(cfg["data"].get("train_size", 55000))
    val_size = int(cfg["data"].get("val_size", 5000))
    split_seed = int(cfg["data"].get("split_seed", 42))

    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(split_seed),
    )

    batch_size = int(cfg["data"]["batch_size"])
    num_workers = int(cfg["data"]["num_workers"])
    pin_memory = bool(cfg["data"].get("pin_memory", True)) and torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )
    return train_loader, val_loader, test_loader


def build_optimizer(cfg: dict, model: nn.Module) -> optim.Optimizer:
    opt_name = cfg["training"]["optimizer"].lower()
    lr = float(cfg["training"]["lr"])
    weight_decay = float(cfg["training"].get("weight_decay", 0.0))

    if opt_name == "sgd":
        momentum = float(cfg["training"].get("momentum", 0.9))
        return optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    if opt_name == "adam":
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {opt_name}")


def build_scheduler(cfg: dict, optimizer: optim.Optimizer):
    sched_name = cfg["training"].get("scheduler", "none").lower()
    epochs = int(cfg["training"]["epochs"])

    if sched_name == "none":
        return None
    if sched_name == "step":
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(cfg["training"].get("step_size", 4)),
            gamma=float(cfg["training"].get("gamma", 0.5)),
        )
    if sched_name == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    if sched_name == "plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=float(cfg["training"].get("plateau_factor", 0.5)),
            patience=int(cfg["training"].get("plateau_patience", 2)),
        )
    raise ValueError(f"Unsupported scheduler: {sched_name}")


def save_checkpoint(
    path: Path,
    epoch: int,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler,
    history: Dict[str, List[float]],
    seed: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "history": history,
            "rng_state": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
            "seed": seed,
        },
        path,
    )


def load_checkpoint(
    ckpt_path: str,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler,
    device: torch.device,
) -> Tuple[int, Dict[str, List[float]]]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler is not None and ckpt.get("scheduler_state") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state"])
    if "rng_state" in ckpt:
        random.setstate(ckpt["rng_state"]["python"])
        np.random.set_state(ckpt["rng_state"]["numpy"])
        torch.set_rng_state(ckpt["rng_state"]["torch"])
        if torch.cuda.is_available() and ckpt["rng_state"]["cuda"] is not None:
            torch.cuda.set_rng_state_all(ckpt["rng_state"]["cuda"])
    return int(ckpt["epoch"]) + 1, ckpt.get(
        "history", {"train_loss": [], "val_loss": [], "val_acc": [], "lr": []}
    )


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        total_loss += loss.item() * targets.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)
    return total_loss / max(total, 1), 100.0 * correct / max(total, 1)


def flatten_dict(d: dict, parent_key: str = "", sep: str = ".") -> Dict[str, str]:
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = str(v)
    return items


def train_pipeline(cfg: dict) -> Dict[str, List[float]]:
    set_seed(int(cfg["experiment"]["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path(cfg["experiment"]["output_dir"]) / cfg["experiment"]["run_name"]
    ckpt_dir = out_dir / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader = get_dataloaders(cfg)
    model = build_model(cfg, device)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)

    start_epoch = 1
    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": [], "val_acc": [], "lr": []}

    resume_path = cfg["checkpointing"].get("resume_from")
    if resume_path:
        start_epoch, history = load_checkpoint(resume_path, model, optimizer, scheduler, device)

    mlflow.set_tracking_uri(cfg["experiment"]["mlflow_tracking_uri"])
    mlflow.set_experiment(cfg["experiment"]["name"])

    accum_steps = max(1, int(cfg["training"].get("grad_accum_steps", 1)))
    epochs = int(cfg["training"]["epochs"])
    stop_after = cfg["checkpointing"].get("stop_after_epoch")

    with mlflow.start_run(run_name=cfg["experiment"]["run_name"]):
        mlflow.log_params(flatten_dict(cfg))

        for epoch in range(start_epoch, epochs + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            running_loss, seen_samples = 0.0, 0
            t0 = time.perf_counter()

            for step, (inputs, targets) in enumerate(train_loader, start=1):
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                outputs = model(inputs)
                loss = criterion(outputs, targets) / accum_steps
                loss.backward()

                if step % accum_steps == 0 or step == len(train_loader):
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                running_loss += loss.item() * accum_steps * targets.size(0)
                seen_samples += targets.size(0)

            throughput = seen_samples / max(time.perf_counter() - t0, 1e-6)
            train_loss = running_loss / max(seen_samples, 1)
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            current_lr = optimizer.param_groups[0]["lr"]

            if scheduler is not None:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_acc)
                else:
                    scheduler.step()

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            history["lr"].append(current_lr)

            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "lr": current_lr,
                    "throughput_samples_per_sec": throughput,
                },
                step=epoch,
            )

            ckpt_file = ckpt_dir / f"ckpt_epoch_{epoch}.pt"
            if epoch % int(cfg["checkpointing"].get("save_every_epochs", 1)) == 0 or (
                stop_after is not None and epoch == int(stop_after)
            ):
                save_checkpoint(
                    ckpt_file, epoch, model, optimizer, scheduler, history, int(cfg["experiment"]["seed"])
                )

            print(
                f"Epoch {epoch:02d}/{epochs:02d} | LR: {current_lr:.6f} | "
                f"Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.2f}% | Throughput: {throughput:.1f} img/s"
            )

            if stop_after is not None and epoch >= int(stop_after):
                print(f"[Stopped at epoch {epoch}] Checkpoint saved at {ckpt_file}")
                break

        history_path = out_dir / "history.json"
        history_path.write_text(json.dumps(history, indent=2))
        mlflow.log_artifact(str(history_path))

        if len(history["val_acc"]) == epochs:
            _, test_acc = evaluate(model, test_loader, criterion, device)
            mlflow.log_metric("test_acc", test_acc)
            print(f"Final Test Accuracy: {test_acc:.2f}%")

    return history