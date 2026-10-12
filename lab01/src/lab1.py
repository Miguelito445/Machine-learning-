
import csv
import random
import statistics
import time
from pathlib import Path

import joblib
import numpy as np
from memory_profiler import memory_usage
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

SEED = 42

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "lab01" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)


def peak_memory_mib(function, *args):
    """Measure peak process memory in MiB during a function call."""
    samples = memory_usage(
        (function, args, {}),
        interval=0.01,
        max_usage=False,
        retval=False,
        include_children=True,
    )
    return max(samples)


def main():

    X, y = load_breast_cancer(return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        stratify=y,
        random_state=42,
    )

    models = {
        "LogisticRegression": lambda: LogisticRegression(
            max_iter=1000,
            random_state=42,
        ),
        "RandomForestClassifier": lambda: RandomForestClassifier(
            n_estimators=100,
            random_state=42,
        ),
    }

    accuracy_rows = []
    measurement_rows = []

    print("Dataset: Breast Cancer Wisconsin (Diagnostic)")
    print(f"Dataset shape: {X.shape}")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")

    for name, make_model in models.items():
        print(f"\n--- {name} ---")

        warmup_model = make_model()
        warmup_model.fit(X_train, y_train)

        training_times = []

        for _ in range(5):
            model = make_model()

            start = time.perf_counter()
            model.fit(X_train, y_train)
            elapsed = time.perf_counter() - start

            training_times.append(elapsed)

        median_training_seconds = statistics.median(
            training_times
        )

        final_model = make_model()
        final_model.fit(X_train, y_train)

        accuracy = final_model.score(X_test, y_test)

        memory_model = make_model()
        training_peak_mib = peak_memory_mib(
            memory_model.fit,
            X_train,
            y_train,
        )

        model_path = RESULTS / f"{name.lower()}.joblib"
        joblib.dump(final_model, model_path)

        size_bytes = model_path.stat().st_size
        size_kb = size_bytes / 1024

        sample = X_test[0:1]

        final_model.predict(sample) 
        inference_times_ms = []

        for _ in range(100):
            start = time.perf_counter()
            final_model.predict(sample)
            elapsed_ms = (
                time.perf_counter() - start
            ) * 1000

            inference_times_ms.append(elapsed_ms)

        median_inference_ms = statistics.median(
            inference_times_ms
        )

        inference_peak_mib = peak_memory_mib(
            final_model.predict,
            sample,
        )

        peak_memory = max(
            training_peak_mib,
            inference_peak_mib,
        )

        print(f"Test accuracy: {accuracy:.4f}")
        print(
            "Median training time: "
            f"{median_training_seconds:.6f} seconds"
        )
        print(
            "Median inference latency: "
            f"{median_inference_ms:.6f} ms"
        )
        print(f"Model size: {size_bytes} bytes")
        print(f"Model size: {size_kb:.3f} KB")
        print(
            f"Peak training process memory: "
            f"{training_peak_mib:.3f} MiB"
        )
        print(
            f"Peak inference process memory: "
            f"{inference_peak_mib:.3f} MiB"
        )

        accuracy_rows.append({
            "model": name,
            "accuracy": f"{accuracy:.4f}",
        })

        measurement_rows.append({
            "model": name,
            "accuracy": f"{accuracy:.4f}",
            "median_training_time_seconds":
                f"{median_training_seconds:.6f}",
            "median_single_sample_inference_ms":
                f"{median_inference_ms:.6f}",
            "model_size_bytes": size_bytes,
            "model_size_KB": f"{size_kb:.3f}",
            "peak_training_process_memory_MiB":
                f"{training_peak_mib:.3f}",
            "peak_inference_process_memory_MiB":
                f"{inference_peak_mib:.3f}",
            "peak_process_memory_MiB":
                f"{peak_memory:.3f}",
        })

    with open(
        RESULTS / "baseline_accuracy.csv",
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["model", "accuracy"],
        )
        writer.writeheader()
        writer.writerows(accuracy_rows)

    with open(
        RESULTS / "system_measurements.csv",
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=measurement_rows[0].keys(),
        )
        writer.writeheader()
        writer.writerows(measurement_rows)

    print("\nAll measurements saved to lab01/results/")


if __name__ == "__main__":
    main()
