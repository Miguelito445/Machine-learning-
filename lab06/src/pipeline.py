import os

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import json
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import yaml
from torch.utils.data import DataLoader, random_split


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_path(base_dir: Path, raw_path: str) -> Path:
    p = Path(raw_path)
    return p if p.is_absolute() else (base_dir / p).resolve()


class MLPModel(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CNNModel(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


class AttentionModel(nn.Module):
    def __init__(self, num_classes: int = 10, d_model: int = 64, nhead: int = 4) -> None:
        super().__init__()
        self.proj = nn.Linear(28, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, 28, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=128,
            dropout=0.1,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq = x.squeeze(1)  # (B, 28, 28)
        tokens = self.proj(seq) + self.pos_emb
        encoded = self.encoder(tokens)
        pooled = encoded.mean(dim=1)
        return self.head(pooled)


def build_model(cfg: Dict) -> nn.Module:
    arch = cfg["model"]["arch"].lower()
    num_classes = int(cfg["model"].get("num_classes", 10))
    if arch == "mlp":
        return MLPModel(num_classes=num_classes)
    if arch == "cnn":
        return CNNModel(num_classes=num_classes)
    if arch == "attention":
        return AttentionModel(num_classes=num_classes)
    raise ValueError(f"Unsupported architecture '{arch}'. Choose from: mlp, cnn, attention.")


def get_dataloaders(
    cfg: Dict,
    script_dir: Path,
    batch_size_override: int = None,
    num_workers_override: int = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    data_cfg = cfg["data"]
    data_dir = resolve_path(script_dir, data_cfg.get("data_dir", "../data"))
    batch_size = int(batch_size_override if batch_size_override is not None else data_cfg["batch_size"])
    num_workers = int(num_workers_override if num_workers_override is not None else data_cfg["num_workers"])
    pin_memory = bool(data_cfg.get("pin_memory", True)) and torch.cuda.is_available()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ])

    full_train = torchvision.datasets.FashionMNIST(
        root=str(data_dir), train=True, download=True, transform=transform
    )
    test_ds = torchvision.datasets.FashionMNIST(
        root=str(data_dir), train=False, download=True, transform=transform
    )

    train_size = int(data_cfg.get("train_size", 55000))
    val_size = int(data_cfg.get("val_size", 5000))
    split_seed = int(data_cfg.get("split_seed", 42))
    generator = torch.Generator().manual_seed(split_seed)
    train_ds, val_ds = random_split(full_train, [train_size, val_size], generator=generator)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, test_loader


def build_optimizer(model: nn.Module, cfg: Dict) -> optim.Optimizer:
    t_cfg = cfg["training"]
    opt_name = t_cfg["optimizer"].lower()
    lr = float(t_cfg["lr"])
    weight_decay = float(t_cfg.get("weight_decay", 0.0))

    if opt_name == "adam":
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if opt_name == "sgd":
        momentum = float(t_cfg.get("momentum", 0.9))
        return optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer '{opt_name}'. Choose from: adam, sgd.")


def build_scheduler(optimizer: optim.Optimizer, cfg: Dict):
    t_cfg = cfg["training"]
    sched_name = str(t_cfg.get("scheduler", "none")).lower()
    epochs = int(t_cfg["epochs"])

    if sched_name == "none":
        return None
    if sched_name == "step":
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(t_cfg.get("step_size", 4)),
            gamma=float(t_cfg.get("gamma", 0.5)),
        )
    if sched_name == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    if sched_name == "plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            patience=int(t_cfg.get("plateau_patience", 2)),
            factor=float(t_cfg.get("plateau_factor", 0.5)),
        )
    raise ValueError(f"Unsupported scheduler '{sched_name}'. Choose from: none, step, cosine, plateau.")


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    avg_loss = total_loss / max(1, total)
    acc = 100.0 * correct / max(1, total)
    return avg_loss, acc


def train_pipeline(cfg: Dict, script_dir: Path = None) -> Dict:
    if script_dir is None:
        script_dir = Path(__file__).resolve().parent

    set_seed(int(cfg["experiment"]["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_name = cfg["experiment"]["run_name"]
    base_out_dir = resolve_path(script_dir, cfg["experiment"]["output_dir"])
    run_dir = base_out_dir / run_name
    ckpt_dir = run_dir / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    tracking_uri = resolve_path(script_dir, cfg["experiment"]["mlflow_tracking_uri"]).as_uri()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(cfg["experiment"]["name"])

    train_loader, val_loader, test_loader = get_dataloaders(cfg, script_dir)
    model = build_model(cfg).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    epochs = int(cfg["training"]["epochs"])
    grad_accum = max(1, int(cfg["training"].get("grad_accum_steps", 1)))
    save_every = max(1, int(cfg["checkpointing"].get("save_every_epochs", 1)))
    stop_after = cfg["checkpointing"].get("stop_after_epoch", None)
    resume_from = cfg["checkpointing"].get("resume_from", None)

    start_epoch = 1
    history: Dict[str, List] = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "lr": [],
        "throughput_samples_per_sec": [],
    }
    history_path = run_dir / "history.json"

    if resume_from:
        ckpt_path = resolve_path(script_dir, str(resume_from))
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = int(ckpt["epoch"]) + 1
        if ckpt.get("history"):
            history = ckpt["history"]
        elif history_path.exists():
            history = json.loads(history_path.read_text(encoding="utf-8"))
        print(f"[Resumed] Loaded checkpoint '{ckpt_path}' -> starting at epoch {start_epoch}")

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "arch": cfg["model"]["arch"],
            "batch_size": cfg["data"]["batch_size"],
            "num_workers": cfg["data"]["num_workers"],
            "optimizer": cfg["training"]["optimizer"],
            "lr": cfg["training"]["lr"],
            "scheduler": cfg["training"]["scheduler"],
            "grad_accum_steps": grad_accum,
            "seed": cfg["experiment"]["seed"],
            "device": str(device),
        })

        for epoch in range(start_epoch, epochs + 1):
            model.train()
            optimizer.zero_grad()
            running_loss = 0.0
            seen_samples = 0
            epoch_start = time.perf_counter()

            for step, (x, y) in enumerate(train_loader, start=1):
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y) / grad_accum
                loss.backward()

                if step % grad_accum == 0 or step == len(train_loader):
                    optimizer.step()
                    optimizer.zero_grad()

                running_loss += (loss.item() * grad_accum) * x.size(0)
                seen_samples += x.size(0)

            elapsed = max(1e-6, time.perf_counter() - epoch_start)
            throughput = seen_samples / elapsed
            train_loss = running_loss / max(1, seen_samples)
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            current_lr = optimizer.param_groups[0]["lr"]

            if scheduler is not None:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_acc)
                else:
                    scheduler.step()

            history["train_loss"].append(round(train_loss, 6))
            history["val_loss"].append(round(val_loss, 6))
            history["val_acc"].append(round(val_acc, 4))
            history["lr"].append(current_lr)
            history["throughput_samples_per_sec"].append(round(throughput, 2))

            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "lr": current_lr,
                "throughput_samples_per_sec": throughput,
            }, step=epoch)

            print(
                f"Epoch {epoch:02d}/{epochs:02d} | LR: {current_lr:.6f} | "
                f"Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.2f}% | "
                f"Throughput: {throughput:.1f} img/s"
            )

            history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

            if epoch % save_every == 0 or (stop_after is not None and epoch == int(stop_after)):
                ckpt_file = ckpt_dir / f"ckpt_epoch_{epoch}.pt"
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                    "history": history,
                }, ckpt_file)

            if stop_after is not None and epoch >= int(stop_after):
                print(f"[Stopped at epoch {epoch}] Checkpoint saved at {ckpt_dir / f'ckpt_epoch_{epoch}.pt'}")
                mlflow.log_artifact(str(history_path))
                return {"history": history, "run_dir": str(run_dir)}

        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        mlflow.log_metrics({"test_loss": test_loss, "test_acc": test_acc})
        mlflow.log_artifact(str(history_path))
        print(f"Final Test Accuracy: {test_acc:.2f}% (Test Loss: {test_loss:.4f})")

    return {"history": history, "test_acc": test_acc, "run_dir": str(run_dir)}


if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    cfg_path = script_dir / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict) or "experiment" not in cfg:
        raise ValueError(f"Config at {cfg_path} is empty or invalid.")

    train_pipeline(cfg, script_dir=script_dir)