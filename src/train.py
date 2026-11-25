"""Baseline experiments and Stacked LSTM training/evaluation script (CPU-only).

Evaluates Persistence and Seasonal Naive (24h) baselines alongside the specified
Stacked LSTM architecture on hourly Household Electric Power Consumption sequences,
computing metrics in original kW units without hyperparameter tuning on the test set.
"""

from __future__ import annotations

import os

# Enforce CPU-only execution and deterministic hashing BEFORE importing TensorFlow
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["PYTHONHASHSEED"] = "42"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import ctypes
import hashlib
import io
import json
import platform
import random
import time
from pathlib import Path
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
)
import tensorflow as tf


SEED = 42
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.001
CLIPNORM = 1.0
SHUFFLE_BUFFER_SIZE = 4096

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "household_power_consumption.txt"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
HOURLY_CSV_PATH = PROCESSED_DIR / "hourly_dataset.csv"
METADATA_PATH = PROCESSED_DIR / "preprocessing_metadata.json"
TARGET_SCALER_PATH = PROCESSED_DIR / "scalers" / "target_scaler.joblib"

MODELS_DIR = PROJECT_ROOT / "models"
BEST_MODEL_PATH = MODELS_DIR / "stacked_lstm_best.keras"
MODEL_SUMMARY_PATH = MODELS_DIR / "model_summary.txt"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
HISTORY_CSV_PATH = REPORTS_DIR / "training_history.csv"
HISTORY_JSON_PATH = REPORTS_DIR / "training_history.json"
PRED_VAL_CSV_PATH = REPORTS_DIR / "predictions_val.csv"
PRED_TEST_CSV_PATH = REPORTS_DIR / "predictions_test.csv"
EVAL_CSV_PATH = REPORTS_DIR / "evaluation_metrics.csv"
EVAL_JSON_PATH = REPORTS_DIR / "evaluation_metrics.json"


class _ProcessMemoryCounters(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_uint32),
        ("PageFaultCount", ctypes.c_uint32),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def get_process_memory_mb() -> dict[str, float | None]:
    """Return current and peak RSS working set memory (MB) on Windows."""
    try:
        counters = _ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        ok = ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
        if ok:
            return {
                "rss_mb": round(counters.WorkingSetSize / (1024 ** 2), 2),
                "peak_rss_mb": round(counters.PeakWorkingSetSize / (1024 ** 2), 2),
            }
    except Exception:
        pass
    return {"rss_mb": None, "peak_rss_mb": None}


def compute_sha256(file_path: Path) -> str:
    """Compute SHA-256 hex digest of a file in streaming chunks."""
    hasher = hashlib.sha256()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def set_reproducibility_and_cpu() -> dict[str, Any]:
    """Fix random seeds and verify strict CPU-only TensorFlow execution."""
    random.seed(SEED)
    np.random.seed(SEED)
    tf.keras.utils.set_random_seed(SEED)

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        tf.config.set_visible_devices([], "GPU")

    logical_gpus = tf.config.list_logical_devices("GPU")
    logical_cpus = tf.config.list_logical_devices("CPU")
    assert len(logical_gpus) == 0, f"Expected 0 logical GPUs, found {logical_gpus}"
    assert len(logical_cpus) >= 1, "Expected at least 1 logical CPU device"

    return {
        "python_version": platform.python_version(),
        "tensorflow_version": tf.__version__,
        "keras_version": tf.keras.__version__,
        "processor": platform.processor(),
        "logical_cpu_cores": os.cpu_count(),
        "logical_cpus": [d.name for d in logical_cpus],
        "logical_gpus": [d.name for d in logical_gpus],
        "seed": SEED,
    }


def load_processed_split(split_name: str) -> dict[str, np.ndarray]:
    """Load preprocessed NPZ arrays for a given split."""
    npz_path = PROCESSED_DIR / f"sequences_{split_name}.npz"
    with np.load(npz_path) as data:
        return {key: data[key] for key in data.files}


def compute_regression_metrics(y_true_kw: np.ndarray, y_pred_kw: np.ndarray) -> dict[str, float]:
    """Compute MAE, MSE, RMSE, and R2 in original kW units."""
    y_t = np.asarray(y_true_kw, dtype=np.float64).ravel()
    y_p = np.asarray(y_pred_kw, dtype=np.float64).ravel()
    return {
        "mae_kw": round(float(mean_absolute_error(y_t, y_p)), 6),
        "mse_kw2": round(float(mean_squared_error(y_t, y_p)), 6),
        "rmse_kw": round(float(root_mean_squared_error(y_t, y_p)), 6),
        "r2": round(float(r2_score(y_t, y_p)), 6),
    }


def evaluate_baselines(
    splits_data: dict[str, dict[str, np.ndarray]],
    target_scaler: Any,
    hourly_df: pd.DataFrame,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, dict[str, dict[str, float]]]]:
    """Compute Persistence (t-1h) and Seasonal Naive (t-24h) predictions and verify against CSV."""
    baseline_preds: dict[str, dict[str, np.ndarray]] = {}
    baseline_metrics: dict[str, dict[str, dict[str, float]]] = {}

    for split_name in ("val", "test"):
        arr = splits_data[split_name]
        x = arr["X"]
        y_true_kw = arr["y_raw_kw"].ravel()
        target_dt = pd.to_datetime(arr["target_ts"], unit="s")

        # X[:, -1, 0] corresponds to t-1h (index 59), X[:, -24, 0] corresponds to t-24h (index 36)
        pred_pers_kw = target_scaler.inverse_transform(x[:, -1, 0].reshape(-1, 1)).ravel().astype(np.float32)
        pred_seas_kw = target_scaler.inverse_transform(x[:, -24, 0].reshape(-1, 1)).ravel().astype(np.float32)

        # Cross-verify exact lag alignment against hourly_dataset.csv
        expected_pers = hourly_df.loc[target_dt - pd.Timedelta(hours=1), "Global_active_power"].to_numpy(dtype=np.float32)
        expected_seas = hourly_df.loc[target_dt - pd.Timedelta(hours=24), "Global_active_power"].to_numpy(dtype=np.float32)
        pers_diff = float(np.max(np.abs(pred_pers_kw - expected_pers)))
        seas_diff = float(np.max(np.abs(pred_seas_kw - expected_seas)))
        assert pers_diff < 1e-4, f"Persistence lag mismatch on {split_name}: {pers_diff}"
        assert seas_diff < 1e-4, f"Seasonal Naive 24h lag mismatch on {split_name}: {seas_diff}"

        baseline_preds[split_name] = {
            "persistence": pred_pers_kw,
            "seasonal_naive_24h": pred_seas_kw,
        }
        baseline_metrics[split_name] = {
            "persistence": compute_regression_metrics(y_true_kw, pred_pers_kw),
            "seasonal_naive_24h": compute_regression_metrics(y_true_kw, pred_seas_kw),
        }

    return baseline_preds, baseline_metrics


def build_stacked_lstm(input_shape: tuple[int, int]) -> tf.keras.Model:
    """Build the specified 2-layer Stacked LSTM architecture."""
    inputs = tf.keras.Input(shape=input_shape, name="input_sequence")
    x = tf.keras.layers.LSTM(64, return_sequences=True, name="lstm_1")(inputs)
    x = tf.keras.layers.Dropout(0.2, name="dropout_1")(x)
    x = tf.keras.layers.LSTM(32, return_sequences=False, name="lstm_2")(x)
    x = tf.keras.layers.Dropout(0.2, name="dropout_2")(x)
    x = tf.keras.layers.Dense(16, activation="relu", name="dense_hidden")(x)
    outputs = tf.keras.layers.Dense(1, activation="linear", name="output_scaled")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="stacked_lstm_power_forecaster")
    return model


class EpochMetricsCallback(tf.keras.callbacks.Callback):
    """Record per-epoch wall-clock duration and learning rate."""

    def __init__(self) -> None:
        super().__init__()
        self.epoch_seconds: list[float] = []
        self.learning_rates_start: list[float] = []
        self._start_time: float = 0.0

    def on_epoch_begin(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        self._start_time = time.perf_counter()
        lr_val = float(np.asarray(self.model.optimizer.learning_rate))
        self.learning_rates_start.append(lr_val)

    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        elapsed = time.perf_counter() - self._start_time
        self.epoch_seconds.append(round(elapsed, 3))


def analyze_residuals_and_errors(
    target_ts: np.ndarray,
    y_true_kw: np.ndarray,
    y_pred_lstm_kw: np.ndarray,
    y_pred_pers_kw: np.ndarray,
) -> dict[str, Any]:
    """Compute detailed residual distribution, hourly errors, and load regime breakdowns."""
    residuals = (y_true_kw - y_pred_lstm_kw).astype(np.float64)
    abs_err = np.abs(residuals)
    target_dt = pd.to_datetime(target_ts, unit="s")

    residual_summary = {
        "mean_bias_kw": round(float(residuals.mean()), 6),
        "std_kw": round(float(residuals.std(ddof=1)), 6),
        "min_kw": round(float(residuals.min()), 6),
        "q25_kw": round(float(np.percentile(residuals, 25)), 6),
        "median_kw": round(float(np.median(residuals)), 6),
        "q75_kw": round(float(np.percentile(residuals, 75)), 6),
        "max_kw": round(float(residuals.max()), 6),
        "p95_abs_err_kw": round(float(np.percentile(abs_err, 95)), 6),
        "p99_abs_err_kw": round(float(np.percentile(abs_err, 99)), 6),
    }

    df_err = pd.DataFrame({
        "hour": target_dt.hour,
        "y_true": y_true_kw,
        "y_pred_lstm": y_pred_lstm_kw,
        "y_pred_pers": y_pred_pers_kw,
        "residual_lstm": residuals,
        "abs_err_lstm": abs_err,
        "sq_err_lstm": residuals ** 2,
        "abs_err_pers": np.abs(y_true_kw - y_pred_pers_kw),
        "sq_err_pers": (y_true_kw - y_pred_pers_kw) ** 2,
    })

    hourly_grp = df_err.groupby("hour").agg(
        count=("y_true", "count"),
        mae_lstm=("abs_err_lstm", "mean"),
        rmse_lstm=("sq_err_lstm", lambda s: float(np.sqrt(s.mean()))),
        bias_lstm=("residual_lstm", "mean"),
        mae_pers=("abs_err_pers", "mean"),
        rmse_pers=("sq_err_pers", lambda s: float(np.sqrt(s.mean()))),
    )
    hourly_dict = {
        str(int(h)): {k: (int(v) if k == "count" else round(float(v), 5)) for k, v in row.items()}
        for h, row in hourly_grp.to_dict(orient="index").items()
    }

    regimes = [
        ("low_lt_0.5kw", y_true_kw < 0.5),
        ("medium_0.5_to_2.0kw", (y_true_kw >= 0.5) & (y_true_kw <= 2.0)),
        ("high_gt_2.0kw", y_true_kw > 2.0),
    ]
    regime_dict: dict[str, dict[str, Any]] = {}
    for label, mask in regimes:
        n_reg = int(mask.sum())
        if n_reg == 0:
            continue
        r_res = residuals[mask]
        regime_dict[label] = {
            "count": n_reg,
            "pct": round(n_reg / len(y_true_kw) * 100.0, 2),
            "mae_kw": round(float(np.abs(r_res).mean()), 5),
            "rmse_kw": round(float(np.sqrt((r_res ** 2).mean())), 5),
            "mean_bias_kw": round(float(r_res.mean()), 5),
        }

    return {
        "residual_summary": residual_summary,
        "hourly_errors": hourly_dict,
        "load_regime_errors": regime_dict,
    }


def generate_evaluation_figures(
    history_df: pd.DataFrame,
    best_epoch: int,
    test_pred_df: pd.DataFrame,
    test_err_analysis: dict[str, Any],
    all_metrics: dict[str, dict[str, dict[str, float]]],
) -> list[str]:
    """Generate 4 evaluation and audit figures in reports/figures/."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_style("whitegrid")
    saved: list[str] = []

    # 10. Training & Validation Loss / MAE curves
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5))
    epochs_x = history_df["epoch"].to_numpy()
    axes[0].plot(epochs_x, history_df["loss"], marker="o", markersize=3.5, linewidth=1.8,
                 color="#1b9e77", label="Train Loss (MSE)")
    axes[0].plot(epochs_x, history_df["val_loss"], marker="s", markersize=3.5, linewidth=1.8,
                 color="#d95f02", label="Validation Loss (MSE)")
    axes[0].axvline(best_epoch, color="#386cb0", linestyle="--", linewidth=1.4,
                    label=f"Best Epoch = {best_epoch} (val_loss={history_df.loc[best_epoch - 1, 'val_loss']:.5f})")
    axes[0].set_title("Training vs. Validation Loss (Scaled MSE)", fontsize=11.5, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE Loss (Scaled)")
    axes[0].legend(fontsize=8.5)

    axes[1].plot(epochs_x, history_df["mae"], marker="o", markersize=3.5, linewidth=1.8,
                 color="#1b9e77", label="Train MAE (Scaled)")
    axes[1].plot(epochs_x, history_df["val_mae"], marker="s", markersize=3.5, linewidth=1.8,
                 color="#d95f02", label="Validation MAE (Scaled)")
    axes[1].axvline(best_epoch, color="#386cb0", linestyle="--", linewidth=1.4, label=f"Best Epoch = {best_epoch}")
    ax_lr = axes[1].twinx()
    ax_lr.step(epochs_x, history_df["lr_start"], where="post", color="#7570b3",
               linestyle=":", linewidth=1.5, alpha=0.85, label="Learning Rate")
    ax_lr.set_ylabel("Learning Rate", color="#7570b3")
    ax_lr.tick_params(axis="y", labelcolor="#7570b3")
    axes[1].set_title("Training vs. Validation MAE & Learning Rate Schedule", fontsize=11.5, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MAE (Scaled)")
    lines_1, labels_1 = axes[1].get_legend_handles_labels()
    lines_2, labels_2 = ax_lr.get_legend_handles_labels()
    axes[1].legend(lines_1 + lines_2, labels_1 + labels_2, fontsize=8, loc="upper right")

    fig.tight_layout()
    p10 = FIGURES_DIR / "10_training_validation_loss.png"
    fig.savefig(p10, dpi=300, bbox_inches="tight")
    plt.close(fig)
    saved.append(str(p10.relative_to(PROJECT_ROOT)).replace("\\", "/"))

    # 11. Actual vs Predicted (7-day Test slice + full Test scatter)
    dt_series = pd.to_datetime(test_pred_df["datetime"])
    slice_mask = (dt_series >= pd.Timestamp("2010-10-11 00:00:00")) & (
        dt_series <= pd.Timestamp("2010-10-17 23:00:00")
    )
    if int(slice_mask.sum()) == 168:
        zoom_df = test_pred_df.loc[slice_mask].copy()
        zoom_title_range = "2010-10-11 to 2010-10-17 (168h)"
    else:
        seg0 = test_pred_df[test_pred_df["segment_id"] == 0]
        zoom_df = seg0.iloc[:168].copy()
        zoom_title_range = f"{zoom_df['datetime'].iloc[0]} to {zoom_df['datetime'].iloc[-1]}"

    zoom_dt = pd.to_datetime(zoom_df["datetime"])
    fig, axes = plt.subplots(2, 1, figsize=(15, 8.5), gridspec_kw={"height_ratios": [1.2, 1]})
    axes[0].plot(zoom_dt, zoom_df["y_true_kw"], color="black", linewidth=1.8, label="Actual (kW)")
    axes[0].plot(zoom_dt, zoom_df["pred_stacked_lstm_kw"], color="#1b9e77", linewidth=1.6,
                 label="Stacked LSTM (kW)")
    axes[0].plot(zoom_dt, zoom_df["pred_persistence_kw"], color="#d95f02", linewidth=1.0,
                 linestyle="--", alpha=0.8, label="Persistence (t-1h)")
    axes[0].plot(zoom_dt, zoom_df["pred_seasonal_24h_kw"], color="#7570b3", linewidth=1.0,
                 linestyle=":", alpha=0.85, label="Seasonal Naive (t-24h)")
    axes[0].set_title(f"Test Set Actual vs. Predicted Hourly Power ({zoom_title_range})",
                      fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Global Active Power (kW)")
    axes[0].set_xlabel("Timestamp")
    axes[0].xaxis.set_major_locator(mdates.DayLocator())
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%a %Y-%m-%d"))
    axes[0].legend(ncol=4, fontsize=9)

    y_t = test_pred_df["y_true_kw"].to_numpy()
    y_p = test_pred_df["pred_stacked_lstm_kw"].to_numpy()
    test_lstm_m = all_metrics["test"]["stacked_lstm"]
    axes[1].scatter(y_t, y_p, s=8, alpha=0.35, color="#386cb0", edgecolors="none", rasterized=True)
    max_lim = max(float(y_t.max()), float(y_p.max())) * 1.05
    axes[1].plot([0, max_lim], [0, max_lim], color="#e41a1c", linestyle="--", linewidth=1.5,
                 label="Ideal Prediction (y = x)")
    axes[1].set_xlim(0, max_lim)
    axes[1].set_ylim(0, max_lim)
    axes[1].set_title(
        f"Test Set Scatter: Actual vs. Stacked LSTM (RMSE = {test_lstm_m['rmse_kw']:.4f} kW, "
        f"MAE = {test_lstm_m['mae_kw']:.4f} kW, R2 = {test_lstm_m['r2']:.4f})",
        fontsize=11.5,
        fontweight="bold",
    )
    axes[1].set_xlabel("Actual Global Active Power (kW)")
    axes[1].set_ylabel("Predicted Global Active Power (kW)")
    axes[1].legend(fontsize=9)

    fig.tight_layout()
    p11 = FIGURES_DIR / "11_actual_vs_predicted_test.png"
    fig.savefig(p11, dpi=300, bbox_inches="tight")
    plt.close(fig)
    saved.append(str(p11.relative_to(PROJECT_ROOT)).replace("\\", "/"))

    # 12. Residual & Error Analysis (Test Set)
    residuals = test_pred_df["residual_stacked_lstm_kw"].to_numpy()
    res_bias = test_err_analysis["residual_summary"]["mean_bias_kw"]
    res_std = test_err_analysis["residual_summary"]["std_kw"]

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    axes[0].hist(residuals, bins=60, density=True, color="#386cb0", alpha=0.65, edgecolor="white", linewidth=0.3)
    sns.kdeplot(x=residuals, ax=axes[0], color="#e41a1c", linewidth=1.8, label="KDE")
    axes[0].axvline(0.0, color="black", linestyle="-", linewidth=1.1, label="Zero Error")
    axes[0].axvline(res_bias, color="#d95f02", linestyle="--", linewidth=1.4,
                    label=f"Mean Bias = {res_bias:+.4f} kW")
    axes[0].set_title(f"Test Residual Distribution (Std = {res_std:.4f} kW)", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Residual (Actual - Predicted, kW)")
    axes[0].set_ylabel("Density")
    axes[0].legend(fontsize=8)

    axes[1].scatter(y_p, residuals, s=8, alpha=0.32, color="#1b9e77", edgecolors="none", rasterized=True)
    axes[1].axhline(0.0, color="#e41a1c", linestyle="--", linewidth=1.3)
    axes[1].set_title("Residuals vs. Predicted Power (Test Set)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Predicted Global Active Power (kW)")
    axes[1].set_ylabel("Residual (Actual - Predicted, kW)")

    hours = list(range(24))
    mae_lstm_h = [test_err_analysis["hourly_errors"][str(h)]["mae_lstm"] for h in hours]
    mae_pers_h = [test_err_analysis["hourly_errors"][str(h)]["mae_pers"] for h in hours]
    axes[2].plot(hours, mae_lstm_h, marker="o", markersize=4, linewidth=1.8,
                 color="#1b9e77", label="Stacked LSTM MAE")
    axes[2].plot(hours, mae_pers_h, marker="s", markersize=3.5, linewidth=1.4,
                 linestyle="--", color="#d95f02", label="Persistence MAE")
    axes[2].set_title("Hourly MAE Comparison on Test Set", fontsize=11, fontweight="bold")
    axes[2].set_xlabel("Target Hour of Day (0-23)")
    axes[2].set_ylabel("MAE (kW)")
    axes[2].set_xticks(range(0, 24, 3))
    axes[2].legend(fontsize=8.5)

    fig.tight_layout()
    p12 = FIGURES_DIR / "12_residual_and_error_analysis.png"
    fig.savefig(p12, dpi=300, bbox_inches="tight")
    plt.close(fig)
    saved.append(str(p12.relative_to(PROJECT_ROOT)).replace("\\", "/"))

    # 13. Model Comparison Metrics (Validation & Test)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    models_order = ["persistence", "seasonal_naive_24h", "stacked_lstm"]
    model_labels = ["Persistence\n(t-1h)", "Seasonal Naive\n(t-24h)", "Stacked LSTM\n(60h->1h)"]
    x_idx = np.arange(len(models_order))
    bar_w = 0.35

    metric_specs = [
        ("mae_kw", "MAE (kW) - Lower is Better"),
        ("rmse_kw", "RMSE (kW) - Lower is Better"),
        ("r2", "R2 Score - Higher is Better"),
    ]
    for ax, (m_key, m_title) in zip(axes, metric_specs):
        val_vals = [all_metrics["val"][m][m_key] for m in models_order]
        test_vals = [all_metrics["test"][m][m_key] for m in models_order]
        b1 = ax.bar(x_idx - bar_w / 2, val_vals, width=bar_w, color="#386cb0", alpha=0.88, label="Validation")
        b2 = ax.bar(x_idx + bar_w / 2, test_vals, width=bar_w, color="#d95f02", alpha=0.88, label="Test")
        ax.set_title(m_title, fontsize=11, fontweight="bold")
        ax.set_xticks(x_idx)
        ax.set_xticklabels(model_labels, fontsize=9)
        y_top = max(max(val_vals), max(test_vals))
        ax.set_ylim(min(0.0, min(min(val_vals), min(test_vals)) - 0.05), y_top * 1.18)
        for bar in list(b1) + list(b2):
            h_val = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h_val + y_top * 0.02, f"{h_val:.4f}",
                    ha="center", va="bottom", fontsize=7.8, fontweight="bold")
        ax.legend(fontsize=8.5)

    fig.tight_layout()
    p13 = FIGURES_DIR / "13_model_comparison_metrics.png"
    fig.savefig(p13, dpi=300, bbox_inches="tight")
    plt.close(fig)
    saved.append(str(p13.relative_to(PROJECT_ROOT)).replace("\\", "/"))

    return saved


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    raw_sha_before = compute_sha256(RAW_DATA_PATH)
    meta_sha_before = compute_sha256(METADATA_PATH)

    env_info = set_reproducibility_and_cpu()
    mem_start = get_process_memory_mb()

    # Load preprocessed data
    splits_data = {name: load_processed_split(name) for name in ("train", "val", "test")}
    target_scaler = joblib.load(TARGET_SCALER_PATH)
    hourly_df = pd.read_csv(HOURLY_CSV_PATH, parse_dates=["Datetime"]).set_index("Datetime")

    # Baseline experiments
    baseline_preds, baseline_metrics = evaluate_baselines(splits_data, target_scaler, hourly_df)
    del hourly_df  # Free unused reference before training

    # Build tf.data pipelines (avoiding extra dataset copies; shuffle train windows only)
    x_train = splits_data["train"]["X"]
    y_train_scaled = splits_data["train"]["y_scaled"]
    x_val = splits_data["val"]["X"]
    y_val_scaled = splits_data["val"]["y_scaled"]
    x_test = splits_data["test"]["X"]

    train_ds = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train_scaled))
        .shuffle(buffer_size=SHUFFLE_BUFFER_SIZE, seed=SEED, reshuffle_each_iteration=True)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = (
        tf.data.Dataset.from_tensor_slices((x_val, y_val_scaled))
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    # Build and compile Stacked LSTM
    input_shape = (int(x_train.shape[1]), int(x_train.shape[2]))
    model = build_stacked_lstm(input_shape)
    summary_buf = io.StringIO()
    model.summary(print_fn=lambda line: summary_buf.write(line + "\n"))
    model_summary_text = summary_buf.getvalue()
    MODEL_SUMMARY_PATH.write_text(model_summary_text, encoding="utf-8")

    trainable_params = int(sum(np.prod(w.shape) for w in model.trainable_weights))
    non_trainable_params = int(sum(np.prod(w.shape) for w in model.non_trainable_weights))
    total_params = trainable_params + non_trainable_params

    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=CLIPNORM)
    model.compile(
        optimizer=optimizer,
        loss="mse",
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")],
    )

    epoch_cb = EpochMetricsCallback()
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(BEST_MODEL_PATH),
            monitor="val_loss",
            save_best_only=True,
            mode="min",
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=1,
        ),
        epoch_cb,
    ]

    print(f"Starting CPU training: epochs={EPOCHS}, batch_size={BATCH_SIZE}, train_seq={len(x_train)}, val_seq={len(x_val)}")
    t_train_start = time.perf_counter()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=2,
    )
    training_wall_seconds = round(time.perf_counter() - t_train_start, 2)
    mem_after_train = get_process_memory_mb()

    # Persist training history
    hist_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    n_epochs_ran = len(hist_dict["loss"])
    best_epoch = int(np.argmin(hist_dict["val_loss"])) + 1

    history_df = pd.DataFrame({
        "epoch": np.arange(1, n_epochs_ran + 1),
        "loss": hist_dict["loss"],
        "mae": hist_dict["mae"],
        "val_loss": hist_dict["val_loss"],
        "val_mae": hist_dict["val_mae"],
        "lr_start": epoch_cb.learning_rates_start,
        "epoch_seconds": epoch_cb.epoch_seconds,
    })
    history_df.to_csv(HISTORY_CSV_PATH, index=False)
    with HISTORY_JSON_PATH.open("w", encoding="utf-8") as f:
        json.dump(history_df.to_dict(orient="records"), f, indent=2)

    # Reload best saved checkpoint to guarantee reproducible evaluation from disk artifact
    best_model = tf.keras.models.load_model(BEST_MODEL_PATH)

    t_eval_start = time.perf_counter()
    val_pred_scaled = best_model.predict(x_val, batch_size=256, verbose=0)
    test_pred_scaled = best_model.predict(x_test, batch_size=256, verbose=0)
    eval_seconds = round(time.perf_counter() - t_eval_start, 3)

    val_pred_kw = target_scaler.inverse_transform(val_pred_scaled).ravel().astype(np.float32)
    test_pred_kw = target_scaler.inverse_transform(test_pred_scaled).ravel().astype(np.float32)

    val_true_kw = splits_data["val"]["y_raw_kw"].ravel()
    test_true_kw = splits_data["test"]["y_raw_kw"].ravel()

    all_metrics = {
        "val": {
            "persistence": baseline_metrics["val"]["persistence"],
            "seasonal_naive_24h": baseline_metrics["val"]["seasonal_naive_24h"],
            "stacked_lstm": compute_regression_metrics(val_true_kw, val_pred_kw),
        },
        "test": {
            "persistence": baseline_metrics["test"]["persistence"],
            "seasonal_naive_24h": baseline_metrics["test"]["seasonal_naive_24h"],
            "stacked_lstm": compute_regression_metrics(test_true_kw, test_pred_kw),
        },
    }

    # Compute relative improvements (%) of Stacked LSTM over baselines
    improvements: dict[str, dict[str, dict[str, float]]] = {}
    for split_name in ("val", "test"):
        improvements[split_name] = {}
        lstm_m = all_metrics[split_name]["stacked_lstm"]
        for base_name in ("persistence", "seasonal_naive_24h"):
            base_m = all_metrics[split_name][base_name]
            improvements[split_name][f"vs_{base_name}"] = {
                "mae_reduction_pct": round((base_m["mae_kw"] - lstm_m["mae_kw"]) / base_m["mae_kw"] * 100.0, 2),
                "mse_reduction_pct": round((base_m["mse_kw2"] - lstm_m["mse_kw2"]) / base_m["mse_kw2"] * 100.0, 2),
                "rmse_reduction_pct": round((base_m["rmse_kw"] - lstm_m["rmse_kw"]) / base_m["rmse_kw"] * 100.0, 2),
                "r2_gain": round(lstm_m["r2"] - base_m["r2"], 6),
            }

    # Residual & error analysis
    val_err_analysis = analyze_residuals_and_errors(
        splits_data["val"]["target_ts"],
        val_true_kw,
        val_pred_kw,
        baseline_preds["val"]["persistence"],
    )
    test_err_analysis = analyze_residuals_and_errors(
        splits_data["test"]["target_ts"],
        test_true_kw,
        test_pred_kw,
        baseline_preds["test"]["persistence"],
    )

    # Save prediction CSVs
    val_pred_df = pd.DataFrame({
        "target_ts": splits_data["val"]["target_ts"],
        "datetime": pd.to_datetime(splits_data["val"]["target_ts"], unit="s").strftime("%Y-%m-%d %H:%M:%S"),
        "segment_id": splits_data["val"]["segment_id"],
        "y_true_kw": val_true_kw,
        "pred_persistence_kw": baseline_preds["val"]["persistence"],
        "pred_seasonal_24h_kw": baseline_preds["val"]["seasonal_naive_24h"],
        "pred_stacked_lstm_kw": val_pred_kw,
        "residual_stacked_lstm_kw": (val_true_kw - val_pred_kw).astype(np.float32),
    })
    val_pred_df.to_csv(PRED_VAL_CSV_PATH, index=False)

    test_pred_df = pd.DataFrame({
        "target_ts": splits_data["test"]["target_ts"],
        "datetime": pd.to_datetime(splits_data["test"]["target_ts"], unit="s").strftime("%Y-%m-%d %H:%M:%S"),
        "segment_id": splits_data["test"]["segment_id"],
        "y_true_kw": test_true_kw,
        "pred_persistence_kw": baseline_preds["test"]["persistence"],
        "pred_seasonal_24h_kw": baseline_preds["test"]["seasonal_naive_24h"],
        "pred_stacked_lstm_kw": test_pred_kw,
        "residual_stacked_lstm_kw": (test_true_kw - test_pred_kw).astype(np.float32),
    })
    test_pred_df.to_csv(PRED_TEST_CSV_PATH, index=False)

    # Save comparison metrics CSV
    eval_rows = []
    for split_name in ("val", "test"):
        for model_name in ("persistence", "seasonal_naive_24h", "stacked_lstm"):
            m = all_metrics[split_name][model_name]
            eval_rows.append({
                "split": split_name,
                "model": model_name,
                "n_sequences": int(len(splits_data[split_name]["X"])),
                **m,
            })
    pd.DataFrame(eval_rows).to_csv(EVAL_CSV_PATH, index=False)

    # Generate 4 evaluation visualizations
    saved_figures = generate_evaluation_figures(
        history_df=history_df,
        best_epoch=best_epoch,
        test_pred_df=test_pred_df,
        test_err_analysis=test_err_analysis,
        all_metrics=all_metrics,
    )

    raw_sha_after = compute_sha256(RAW_DATA_PATH)
    meta_sha_after = compute_sha256(METADATA_PATH)
    assert raw_sha_before == raw_sha_after, "Raw data file was altered!"
    assert meta_sha_before == meta_sha_after, "Preprocessed metadata file was altered!"

    mem_end = get_process_memory_mb()

    eval_payload = {
        "environment": env_info,
        "resource_usage": {
            "training_wall_seconds": training_wall_seconds,
            "training_wall_minutes": round(training_wall_seconds / 60.0, 2),
            "avg_epoch_seconds": round(float(np.mean(epoch_cb.epoch_seconds)), 2),
            "evaluation_predict_seconds": eval_seconds,
            "memory_before_train": mem_start,
            "memory_after_train": mem_after_train,
            "memory_end": mem_end,
        },
        "model_architecture": {
            "input_shape": list(input_shape),
            "layers": [
                "LSTM(64, return_sequences=True)",
                "Dropout(0.2)",
                "LSTM(32)",
                "Dropout(0.2)",
                "Dense(16, activation='relu')",
                "Dense(1, activation='linear')",
            ],
            "total_params": total_params,
            "trainable_params": trainable_params,
            "non_trainable_params": non_trainable_params,
            "optimizer": f"Adam(lr={LEARNING_RATE}, clipnorm={CLIPNORM})",
            "loss": "mse",
            "batch_size": BATCH_SIZE,
            "max_epochs": EPOCHS,
        },
        "training_convergence": {
            "epochs_ran": n_epochs_ran,
            "stopped_early": bool(n_epochs_ran < EPOCHS),
            "best_epoch": best_epoch,
            "best_val_loss_scaled": round(float(hist_dict["val_loss"][best_epoch - 1]), 6),
            "train_loss_at_best_epoch_scaled": round(float(hist_dict["loss"][best_epoch - 1]), 6),
            "best_val_mae_scaled": round(float(hist_dict["val_mae"][best_epoch - 1]), 6),
            "train_mae_at_best_epoch_scaled": round(float(hist_dict["mae"][best_epoch - 1]), 6),
            "final_epoch_lr": float(epoch_cb.learning_rates_start[-1]),
            "lr_reductions": int(len(set(epoch_cb.learning_rates_start)) - 1),
        },
        "dataset_sizes": {
            "train_sequences": int(len(x_train)),
            "val_sequences": int(len(x_val)),
            "test_sequences": int(len(x_test)),
        },
        "metrics_kw": all_metrics,
        "relative_improvements": improvements,
        "residual_analysis": {
            "val": val_err_analysis,
            "test": test_err_analysis,
        },
        "integrity_checks": {
            "raw_data_untouched": True,
            "preprocessed_metadata_untouched": True,
            "best_model_reloaded_and_evaluated": True,
            "tuned_on_test_set": False,
        },
        "saved_artifacts": {
            "best_model": str(BEST_MODEL_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "model_summary": str(MODEL_SUMMARY_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "training_history_csv": str(HISTORY_CSV_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "training_history_json": str(HISTORY_JSON_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "predictions_val_csv": str(PRED_VAL_CSV_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "predictions_test_csv": str(PRED_TEST_CSV_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "evaluation_metrics_csv": str(EVAL_CSV_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "evaluation_metrics_json": str(EVAL_JSON_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "figures": saved_figures,
        },
    }

    with EVAL_JSON_PATH.open("w", encoding="utf-8") as f:
        json.dump(eval_payload, f, indent=2)

    print("\n=== Experiment Complete ===")
    print(f"Training time: {training_wall_seconds}s across {n_epochs_ran} epochs (best epoch: {best_epoch})")
    for split_name in ("val", "test"):
        print(f"\n[{split_name.upper()} Metrics in Original kW]")
        for model_name in ("persistence", "seasonal_naive_24h", "stacked_lstm"):
            m = all_metrics[split_name][model_name]
            print(
                f"  {model_name:20s} -> MAE={m['mae_kw']:.4f}, MSE={m['mse_kw2']:.4f}, "
                f"RMSE={m['rmse_kw']:.4f}, R2={m['r2']:.4f}"
            )


if __name__ == "__main__":
    main()
