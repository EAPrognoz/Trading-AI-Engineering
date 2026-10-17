#!/usr/bin/env python3
"""Train a tiny tanh network on a fixed synthetic curve for animation data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
from pathlib import Path
from typing import Any

import numpy as np


SEED = 20261015
CONFIG: dict[str, Any] = {
    "data_label": "SYNTHETIC — not market data",
    "generator": "y = sin(pi*x) + 0.25*sin(3*pi*x) + Normal(0, 0.05)",
    "x_distribution": "Uniform(-1, 1)",
    "seed": SEED,
    "split_sizes": {"train": 128, "validation": 64, "test": 64},
    "architecture": [1, 8, 1],
    "hidden_activation": "tanh",
    "output_activation": "linear",
    "optimizer": "Adam",
    "learning_rate": 0.03,
    "beta1": 0.9,
    "beta2": 0.999,
    "epsilon": 1e-8,
    "batch_size": 128,
    "updates_per_epoch": 1,
    "epochs": 400,
    "checkpoints": [0, 1, 5, 10, 25, 50, 100, 200, 400],
    "fixed_unseen_query_x": 0.25,
}


def target_function(x: np.ndarray, noise: np.ndarray) -> np.ndarray:
    return np.sin(np.pi * x) + 0.25 * np.sin(3.0 * np.pi * x) + noise


def make_splits() -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, Any]]:
    streams = np.random.SeedSequence(SEED).spawn(4)
    data: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    stream_records = []
    for name, size, stream in zip(
        ("train", "validation", "test"), (128, 64, 64), streams[:3]
    ):
        rng = np.random.default_rng(stream)
        x = rng.uniform(-1.0, 1.0, size=(size, 1))
        noise = rng.normal(0.0, 0.05, size=(size, 1))
        y = target_function(x, noise)
        order = np.argsort(x[:, 0])
        data[name] = (x[order], y[order])
        stream_records.append({"split": name, "spawn_key": list(stream.spawn_key)})
    init_stream = streams[3]
    stream_records.append({"split": "initialization", "spawn_key": list(init_stream.spawn_key)})
    provenance = {"master_seed": SEED, "child_streams": stream_records}
    return data, {"provenance": provenance, "init_rng": np.random.default_rng(init_stream)}


def initialize(rng: np.random.Generator) -> dict[str, np.ndarray]:
    # Xavier uniform for tanh-hidden and linear-output layers.
    w1_limit = math.sqrt(6.0 / (1 + 8))
    w2_limit = math.sqrt(6.0 / (8 + 1))
    return {
        "W1": rng.uniform(-w1_limit, w1_limit, size=(1, 8)),
        "b1": np.zeros((1, 8)),
        "W2": rng.uniform(-w2_limit, w2_limit, size=(8, 1)),
        "b2": np.zeros((1, 1)),
    }


def predict(params: dict[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    return np.tanh(x @ params["W1"] + params["b1"]) @ params["W2"] + params["b2"]


def loss_and_gradients(
    params: dict[str, np.ndarray], x: np.ndarray, y: np.ndarray
) -> tuple[float, dict[str, np.ndarray]]:
    n = x.shape[0]
    z1 = x @ params["W1"] + params["b1"]
    h1 = np.tanh(z1)
    prediction = h1 @ params["W2"] + params["b2"]
    residual = prediction - y
    loss = float(np.mean(residual**2))
    d_prediction = (2.0 / n) * residual
    d_w2 = h1.T @ d_prediction
    d_b2 = np.sum(d_prediction, axis=0, keepdims=True)
    d_hidden = (d_prediction @ params["W2"].T) * (1.0 - h1**2)
    gradients = {
        "W1": x.T @ d_hidden,
        "b1": np.sum(d_hidden, axis=0, keepdims=True),
        "W2": d_w2,
        "b2": d_b2,
    }
    return loss, gradients


def gradient_check(params: dict[str, np.ndarray], x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    _, analytic = loss_and_gradients(params, x, y)
    eps = 1e-6
    checks: list[dict[str, Any]] = []
    for name, values in params.items():
        for index in np.ndindex(values.shape):
            original = float(values[index])
            values[index] = original + eps
            plus = loss_and_gradients(params, x, y)[0]
            values[index] = original - eps
            minus = loss_and_gradients(params, x, y)[0]
            values[index] = original
            numeric = (plus - minus) / (2.0 * eps)
            exact = float(analytic[name][index])
            absolute = abs(numeric - exact)
            relative = absolute / max(1e-8, abs(numeric) + abs(exact))
            checks.append({"parameter": name, "index": list(index), "analytic": exact,
                           "numeric": numeric, "absolute_error": absolute,
                           "relative_error": relative})
    max_abs = max(c["absolute_error"] for c in checks)
    max_rel = max(c["relative_error"] for c in checks)
    return {"method": "central finite difference", "epsilon": eps,
            "parameter_count": len(checks), "max_absolute_error": max_abs,
            "max_relative_error": max_rel, "tolerance": 1e-5,
            "passed": max_abs < 1e-5, "checks": checks}


def metric_record(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    error = prediction - y
    mse = float(np.mean(error**2))
    return {"mse": mse, "rmse": math.sqrt(mse), "mae": float(np.mean(np.abs(error)))}


def jsonable_params(params: dict[str, np.ndarray]) -> dict[str, Any]:
    return {name: values.tolist() for name, values in params.items()}


def write_csv(path: Path, header: list[str], rows: list[list[Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    data, rng_bundle = make_splits()
    params = initialize(rng_bundle["init_rng"])
    initial_params = {name: values.copy() for name, values in params.items()}
    x_train, y_train = data["train"]
    check = gradient_check(params, x_train[:8], y_train[:8])
    if not check["passed"]:
        raise RuntimeError(f"Gradient check failed: {check['max_absolute_error']}")

    for split, (x, y) in data.items():
        rows = [[float(a), float(b)] for a, b in zip(x[:, 0], y[:, 0])]
        write_csv(output_dir / f"{split}.csv", ["x", "y_synthetic"], rows)

    grid = np.linspace(-1.0, 1.0, 201).reshape(-1, 1)
    query_x = np.array([[CONFIG["fixed_unseen_query_x"]]], dtype=float)
    query_y_clean = float(target_function(query_x, np.zeros_like(query_x))[0, 0])
    checkpoints = set(CONFIG["checkpoints"])
    timeline: list[dict[str, Any]] = []
    loss_rows: list[list[Any]] = []
    prediction_rows: list[list[Any]] = []

    # Adam state is initialized to zero and updated using training data only.
    first = {k: np.zeros_like(v) for k, v in params.items()}
    second = {k: np.zeros_like(v) for k, v in params.items()}
    beta1, beta2 = CONFIG["beta1"], CONFIG["beta2"]

    def save_checkpoint(epoch: int) -> None:
        split_metrics: dict[str, Any] = {}
        for split in ("train", "validation"):
            x, y = data[split]
            pred = predict(params, x)
            split_metrics[split] = metric_record(y, pred)
        curve = predict(params, grid)[:, 0]
        item = {
            "epoch": epoch,
            "weights": jsonable_params(params),
            "metrics": split_metrics,
            "curve": [{"x": float(a), "prediction": float(b)} for a, b in zip(grid[:, 0], curve)],
            "query": {"x": float(query_x[0, 0]), "prediction": float(predict(params, query_x)[0, 0]),
                      "clean_synthetic_reference": query_y_clean},
        }
        timeline.append(item)
        for split, metrics in split_metrics.items():
            for metric, value in metrics.items():
                loss_rows.append([epoch, split, metric, value])
        for split in ("train", "validation"):
            x, _ = data[split]
            pred = predict(params, x)
            prediction_rows.extend([[epoch, split, float(a), float(b)] for a, b in zip(x[:, 0], pred[:, 0])])

    save_checkpoint(0)
    step = 0
    for epoch in range(1, CONFIG["epochs"] + 1):
        step += 1
        _, gradients = loss_and_gradients(params, x_train, y_train)
        for name in params:
            first[name] = beta1 * first[name] + (1.0 - beta1) * gradients[name]
            second[name] = beta2 * second[name] + (1.0 - beta2) * gradients[name] ** 2
            m_hat = first[name] / (1.0 - beta1**step)
            v_hat = second[name] / (1.0 - beta2**step)
            params[name] -= CONFIG["learning_rate"] * m_hat / (np.sqrt(v_hat) + CONFIG["epsilon"])
        if epoch in checkpoints:
            save_checkpoint(epoch)

    # The locked test split is first read only after the fixed fit and all
    # predeclared checkpoint saves have completed. It cannot select a model.
    x_test, y_test = data["test"]
    final_test = metric_record(y_test, predict(params, x_test))
    timeline[-1]["metrics"]["test_final_only"] = final_test
    for metric, value in final_test.items():
        loss_rows.append([CONFIG["epochs"], "test_final_only", metric, value])
    final_test_prediction = predict(params, x_test)
    prediction_rows.extend([[CONFIG["epochs"], "test_final_only", float(a), float(b)]
                            for a, b in zip(x_test[:, 0], final_test_prediction[:, 0])])

    # Additional descriptive comparisons, computed post-training and never
    # used to update weights or select a checkpoint.
    initial_test = metric_record(y_test, predict(initial_params, x_test))
    train_mean = float(np.mean(y_train))
    constant_test = metric_record(y_test, np.full_like(y_test, train_mean))
    post_training_diagnostics = {
        "label": "Additional post-training diagnostics; not part of training or checkpoint selection",
        "test_access_timing": "computed after fixed 400-epoch training completed",
        "test_used_for_tuning": False,
        "initial_untrained_network_test_metrics": initial_test,
        "constant_train_mean_test_metrics": constant_test,
        "constant_prediction": {"value": train_mean, "computed_from": "train targets only"},
        "final_epoch_400_test_metrics": final_test,
    }

    with (output_dir / "initial_weights.json").open("w", encoding="utf-8") as f:
        json.dump(timeline[0]["weights"], f, indent=2)
    np.savez_compressed(output_dir / "checkpoint_weights.npz", **{
        f"epoch_{item['epoch']}_{name}": np.asarray(values)
        for item in timeline for name, values in item["weights"].items()
    })
    with (output_dir / "gradient_check.json").open("w", encoding="utf-8") as f:
        json.dump(check, f, indent=2)
    with (output_dir / "post_training_test_diagnostics.json").open("w", encoding="utf-8") as f:
        json.dump(post_training_diagnostics, f, indent=2)
    with (output_dir / "animation_timeline.json").open("w", encoding="utf-8") as f:
        json.dump({"label": CONFIG["data_label"], "protocol": CONFIG,
                   "split_provenance": rng_bundle["provenance"], "checkpoints": timeline}, f)
    write_csv(output_dir / "metrics.csv", ["epoch", "split", "metric", "value"], loss_rows)
    write_csv(output_dir / "predictions.csv", ["epoch", "split", "x", "prediction"], prediction_rows)

    files = sorted(p for p in output_dir.iterdir() if p.is_file() and p.name != "manifest.json")
    manifest = {
        "label": CONFIG["data_label"], "protocol": CONFIG,
        "split_provenance": rng_bundle["provenance"],
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "platform": platform.platform()},
        "gradient_check": {k: v for k, v in check.items() if k != "checks"},
        "artifacts": {p.name: {"sha256": sha256(p), "bytes": p.stat().st_size} for p in files},
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return {"output_dir": str(output_dir), "gradient_check": manifest["gradient_check"],
            "checkpoints": timeline, "manifest": manifest}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "outputs" / f"seed{SEED}")
    args = parser.parse_args()
    result = run(args.output_dir)
    print(json.dumps({"output_dir": result["output_dir"],
                      "gradient_check": result["gradient_check"],
                      "checkpoint_metrics": [{"epoch": c["epoch"], "metrics": c["metrics"], "query": c["query"]}
                                              for c in result["checkpoints"]]}, indent=2))


if __name__ == "__main__":
    main()
