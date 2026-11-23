import json
from pathlib import Path
import matplotlib.pyplot as plt


def plot_run_history(history_path: str, out_path: str) -> None:
    history = json.loads(Path(history_path).read_text(encoding="utf-8"))
    epochs = list(range(1, len(history["train_loss"]) + 1))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].plot(epochs, history["train_loss"], marker="o", label="Train Loss")
    axes[0].set_title("Train Loss vs. Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True)

    axes[1].plot(epochs, history["val_acc"], marker="o", color="green", label="Val Acc (%)")
    axes[1].set_title("Validation Accuracy vs. Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].grid(True)

    if "throughput_samples_per_sec" in history:
        axes[2].plot(epochs, history["throughput_samples_per_sec"], marker="o", color="purple")
        axes[2].set_title("Throughput vs. Epoch")
        axes[2].set_ylabel("Samples / sec")
    else:
        axes[2].plot(epochs, history["lr"], marker="o", color="purple")
        axes[2].set_title("Learning Rate vs. Epoch")
        axes[2].set_ylabel("Learning Rate")
    axes[2].set_xlabel("Epoch")
    axes[2].grid(True)

    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved chart to {out_path}")


if __name__ == "__main__":
    base = Path(__file__).resolve().parent.parent / "results" / "lab6"
    hist = base / "task7_interrupt_resume" / "history.json"
    if hist.exists():
        plot_run_history(str(hist), str(base / "task7_interrupt_resume" / "training_curves.png"))