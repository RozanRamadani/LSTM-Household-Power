"""Re-runnable hourly preprocessing and validation pipeline for Stacked LSTM forecasting.

Transforms raw 1-minute Household Electric Power Consumption observations into
leakage-free chronological train/validation/test sequences (lookback=60 hours,
horizon=1 hour, target=next-hour mean Global_active_power in kW).
"""

from __future__ import annotations

import hashlib
import json
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
from sklearn.preprocessing import MinMaxScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "household_power_consumption.txt"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SCALERS_DIR = PROCESSED_DIR / "scalers"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
METADATA_PATH = PROCESSED_DIR / "preprocessing_metadata.json"
HOURLY_CSV_PATH = PROCESSED_DIR / "hourly_dataset.csv"

LOOKBACK_HOURS = 60
FORECAST_HORIZON_HOURS = 1
MIN_VALID_MINUTES_PER_HOUR = 45
MAX_INTERPOLATION_GAP_MINUTES = 60

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

TARGET_COL = "Global_active_power"
EXCLUDED_RAW_COLS = ["Global_intensity"]

RAW_MEASUREMENT_COLS = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
]
CONTINUOUS_FEATURES = list(RAW_MEASUREMENT_COLS)
TEMPORAL_FEATURES = [
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    "is_weekend",
]
ALL_FEATURES = CONTINUOUS_FEATURES + TEMPORAL_FEATURES

MEAN_AGG_COLS = ["Global_active_power", "Global_reactive_power", "Voltage"]
ENERGY_SUM_COLS = ["Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]


def compute_sha256(file_path: Path) -> str:
    """Compute SHA-256 hex digest of a file in streaming chunks."""
    hasher = hashlib.sha256()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def to_unix_seconds(index: pd.DatetimeIndex) -> np.ndarray:
    """Convert DatetimeIndex to int64 Unix seconds safely across pandas datetime units."""
    return ((index - pd.Timestamp("1970-01-01")) // pd.Timedelta(seconds=1)).to_numpy(dtype=np.int64)


def detect_long_raw_gap_hours(
    raw_missing_mask: np.ndarray,
    raw_datetimes: pd.Series,
    max_gap_minutes: int,
) -> tuple[set[pd.Timestamp], list[dict[str, Any]]]:
    """Return hourly timestamps touched by any raw contiguous outage > max_gap_minutes."""
    padded = np.concatenate(([False], raw_missing_mask, [False]))
    starts = np.where((~padded[:-1]) & padded[1:])[0]
    ends = np.where(padded[:-1] & (~padded[1:]))[0] - 1

    long_hours: set[pd.Timestamp] = set()
    long_raw_gaps: list[dict[str, Any]] = []
    for s, e in zip(starts, ends):
        start_dt = raw_datetimes.iloc[s]
        end_dt = raw_datetimes.iloc[e]
        duration_min = int((end_dt - start_dt).total_seconds() // 60) + 1
        if duration_min > max_gap_minutes:
            for ts in pd.date_range(start_dt.floor("1h"), end_dt.floor("1h"), freq="1h"):
                long_hours.add(ts)
            long_raw_gaps.append({
                "start": start_dt.strftime("%Y-%m-%d %H:%M:%S"),
                "end": end_dt.strftime("%Y-%m-%d %H:%M:%S"),
                "duration_minutes": duration_min,
            })
    return long_hours, long_raw_gaps


def interpolate_short_gaps_within_split(
    split_df: pd.DataFrame,
    long_raw_gap_hours: set[pd.Timestamp],
    max_gap_minutes: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Interpolate ONLY interior 1-hour missing slots whose raw outage is <= max_gap_minutes."""
    df_out = split_df.copy()
    max_hourly_run = max(1, max_gap_minutes // 60)

    is_na = df_out[TARGET_COL].isna().to_numpy()
    padded = np.concatenate(([False], is_na, [False]))
    starts = np.where((~padded[:-1]) & padded[1:])[0]
    ends = np.where(padded[:-1] & (~padded[1:]))[0] - 1

    missing_before = int(is_na.sum())
    qualifying_indices: list[int] = []
    skipped_blocks: list[dict[str, Any]] = []

    for s, e in zip(starts, ends):
        run_len = int(e - s + 1)
        is_interior = (s > 0) and (e < len(df_out) - 1)
        timestamps_in_run = df_out.index[s : e + 1]
        overlaps_long_raw = any(ts in long_raw_gap_hours for ts in timestamps_in_run)

        if is_interior and (run_len <= max_hourly_run) and (not overlaps_long_raw):
            qualifying_indices.extend(range(s, e + 1))
        else:
            reason = (
                "boundary_gap"
                if not is_interior
                else (
                    "exceeds_hourly_max_gap"
                    if run_len > max_hourly_run
                    else "underlying_raw_outage_gt_60min"
                )
            )
            skipped_blocks.append({
                "start": timestamps_in_run[0].strftime("%Y-%m-%d %H:%M:%S"),
                "end": timestamps_in_run[-1].strftime("%Y-%m-%d %H:%M:%S"),
                "hourly_length": run_len,
                "reason": reason,
            })

    df_out["is_interpolated"] = False
    if qualifying_indices:
        interp_full = df_out[CONTINUOUS_FEATURES].interpolate(method="time")
        qual_ts = df_out.index[qualifying_indices]
        df_out.loc[qual_ts, CONTINUOUS_FEATURES] = interp_full.loc[qual_ts, CONTINUOUS_FEATURES]
        df_out.loc[qual_ts, "is_interpolated"] = True

    df_out["is_valid"] = ~df_out[TARGET_COL].isna()

    # Assign contiguous valid segment IDs within this split
    segment_ids = np.full(len(df_out), -1, dtype=np.int32)
    valid_arr = df_out["is_valid"].to_numpy()
    current_seg = -1
    in_seg = False
    for idx_pos, valid_flag in enumerate(valid_arr):
        if valid_flag:
            if not in_seg:
                current_seg += 1
                in_seg = True
            segment_ids[idx_pos] = current_seg
        else:
            in_seg = False
    df_out["segment_id"] = segment_ids

    interpolated_ts = [
        ts.strftime("%Y-%m-%d %H:%M:%S")
        for ts in df_out.index[df_out["is_interpolated"]]
    ]
    missing_after = int((~df_out["is_valid"]).sum())
    info = {
        "total_hours": int(len(df_out)),
        "missing_hours_before_interp": missing_before,
        "interpolated_hours_count": len(interpolated_ts),
        "interpolated_timestamps": interpolated_ts,
        "missing_hours_after_interp": missing_after,
        "valid_hours_after_interp": int(df_out["is_valid"].sum()),
        "unimputed_gap_blocks": skipped_blocks,
    }
    return df_out, info


def add_temporal_features(df_in: pd.DataFrame) -> pd.DataFrame:
    """Compute deterministic cyclical and calendar features from DatetimeIndex."""
    df_out = df_in.copy()
    idx = df_out.index
    df_out["hour_sin"] = np.sin(2.0 * np.pi * idx.hour / 24.0)
    df_out["hour_cos"] = np.cos(2.0 * np.pi * idx.hour / 24.0)
    df_out["dow_sin"] = np.sin(2.0 * np.pi * idx.dayofweek / 7.0)
    df_out["dow_cos"] = np.cos(2.0 * np.pi * idx.dayofweek / 7.0)
    df_out["month_sin"] = np.sin(2.0 * np.pi * (idx.month - 1) / 12.0)
    df_out["month_cos"] = np.cos(2.0 * np.pi * (idx.month - 1) / 12.0)
    df_out["is_weekend"] = (idx.dayofweek >= 5).astype(np.float64)
    return df_out


def build_sequences_by_segment(
    split_df: pd.DataFrame,
    scaled_continuous: np.ndarray,
    scaled_target: np.ndarray,
    lookback: int,
    horizon: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Build sliding window sequences strictly inside contiguous valid segments."""
    temporal_arr = split_df[TEMPORAL_FEATURES].to_numpy(dtype=np.float32)
    full_features_scaled = np.hstack([scaled_continuous.astype(np.float32), temporal_arr])
    raw_target_arr = split_df[TARGET_COL].to_numpy(dtype=np.float32)
    unix_ts = to_unix_seconds(split_df.index)
    seg_ids = split_df["segment_id"].to_numpy()

    x_list: list[np.ndarray] = []
    y_scaled_list: list[np.ndarray] = []
    y_raw_list: list[np.ndarray] = []
    start_ts_list: list[np.ndarray] = []
    target_ts_list: list[np.ndarray] = []
    seq_seg_list: list[np.ndarray] = []

    segment_details: list[dict[str, Any]] = []
    skipped_short_segments: list[dict[str, Any]] = []
    unique_segs = sorted(int(s) for s in np.unique(seg_ids) if s >= 0)
    min_len_required = lookback + horizon

    for seg in unique_segs:
        pos = np.where(seg_ids == seg)[0]
        assert np.array_equal(pos, np.arange(pos[0], pos[-1] + 1)), f"Non-contiguous segment {seg}"
        seg_len = len(pos)
        seg_ts = unix_ts[pos]
        if seg_len > 1:
            diffs = np.diff(seg_ts)
            assert np.all(diffs == 3600), f"Segment {seg} has non-hourly timestamp gap"

        seg_feat = full_features_scaled[pos]
        seg_y_scaled = scaled_target[pos]
        seg_y_raw = raw_target_arr[pos]
        assert not np.isnan(seg_feat).any(), f"NaN found inside valid segment {seg}"
        assert not np.isnan(seg_y_scaled).any(), f"NaN target found inside valid segment {seg}"

        if seg_len < min_len_required:
            skipped_short_segments.append({
                "segment_id": seg,
                "length_hours": seg_len,
                "start": split_df.index[pos[0]].strftime("%Y-%m-%d %H:%M:%S"),
                "end": split_df.index[pos[-1]].strftime("%Y-%m-%d %H:%M:%S"),
            })
            continue

        n_seq = seg_len - lookback - horizon + 1
        # Build sliding windows cleanly via index slicing
        idx_windows = np.arange(n_seq)[:, None] + np.arange(lookback)[None, :]
        x_seg = seg_feat[idx_windows]  # shape: (n_seq, lookback, n_features)
        target_offsets = np.arange(n_seq) + lookback + horizon - 1
        y_s_seg = seg_y_scaled[target_offsets].reshape(-1, 1).astype(np.float32)
        y_r_seg = seg_y_raw[target_offsets].reshape(-1, 1).astype(np.float32)
        start_s_seg = seg_ts[np.arange(n_seq)]
        target_s_seg = seg_ts[target_offsets]
        seg_id_arr = np.full(n_seq, seg, dtype=np.int32)

        x_list.append(x_seg)
        y_scaled_list.append(y_s_seg)
        y_raw_list.append(y_r_seg)
        start_ts_list.append(start_s_seg)
        target_ts_list.append(target_s_seg)
        seq_seg_list.append(seg_id_arr)

        segment_details.append({
            "segment_id": seg,
            "length_hours": seg_len,
            "sequences": n_seq,
            "start": split_df.index[pos[0]].strftime("%Y-%m-%d %H:%M:%S"),
            "end": split_df.index[pos[-1]].strftime("%Y-%m-%d %H:%M:%S"),
        })

    arrays = {
        "X": np.concatenate(x_list, axis=0).astype(np.float32),
        "y_scaled": np.concatenate(y_scaled_list, axis=0).astype(np.float32),
        "y_raw_kw": np.concatenate(y_raw_list, axis=0).astype(np.float32),
        "window_start_ts": np.concatenate(start_ts_list, axis=0).astype(np.int64),
        "target_ts": np.concatenate(target_ts_list, axis=0).astype(np.int64),
        "segment_id": np.concatenate(seq_seg_list, axis=0).astype(np.int32),
    }
    stats = {
        "total_valid_segments": len(unique_segs),
        "used_segments": len(segment_details),
        "skipped_short_segments_count": len(skipped_short_segments),
        "skipped_short_segments": skipped_short_segments,
        "total_sequences": int(arrays["X"].shape[0]),
        "X_shape": list(arrays["X"].shape),
        "y_shape": list(arrays["y_scaled"].shape),
        "segment_details": segment_details,
    }
    return arrays, stats


def run_validation_checks(
    splits: dict[str, pd.DataFrame],
    seq_arrays: dict[str, dict[str, np.ndarray]],
    split_interp_info: dict[str, dict[str, Any]],
    feature_scaler: MinMaxScaler,
    target_scaler: MinMaxScaler,
    sha256_before: str,
    sha256_after: str,
) -> dict[str, Any]:
    """Run strict automated checks for gap safety, chronology, leakage, shapes, and inverse scaling."""
    checks: dict[str, Any] = {}

    # 1. No sequence crosses a gap or invalid hour
    expected_span_sec = (LOOKBACK_HOURS + FORECAST_HORIZON_HOURS - 1) * 3600
    gap_check_details: dict[str, Any] = {}
    no_gap_crossing = True
    for name, arrays in seq_arrays.items():
        spans = arrays["target_ts"] - arrays["window_start_ts"]
        span_ok = bool(np.all(spans == expected_span_sec))
        finite_ok = bool(
            np.isfinite(arrays["X"]).all()
            and np.isfinite(arrays["y_scaled"]).all()
            and np.isfinite(arrays["y_raw_kw"]).all()
        )
        # Cross-verify against valid timestamp set in split DataFrame
        valid_ts_set = set(to_unix_seconds(splits[name].index[splits[name]["is_valid"]]))
        sample_check_step = max(1, len(arrays["window_start_ts"]) // 500)
        all_61_valid = True
        for st_ts in arrays["window_start_ts"][::sample_check_step]:
            for h in range(LOOKBACK_HOURS + FORECAST_HORIZON_HOURS):
                if int(st_ts) + h * 3600 not in valid_ts_set:
                    all_61_valid = False
                    break
        # Exhaustive check: target_ts and window_start_ts in valid_ts_set and same segment_id
        seg_map = dict(zip(to_unix_seconds(splits[name].index), splits[name]["segment_id"].to_numpy()))
        same_seg = all(
            seg_map[int(s)] == seg_map[int(t)] >= 0
            for s, t in zip(arrays["window_start_ts"], arrays["target_ts"])
        )
        split_ok = span_ok and finite_ok and all_61_valid and same_seg
        no_gap_crossing = no_gap_crossing and split_ok
        gap_check_details[name] = {
            "exact_60h_span_all": span_ok,
            "all_values_finite": finite_ok,
            "same_valid_segment_all": same_seg,
            "sampled_all_61_hours_valid": all_61_valid,
        }
    checks["no_gap_crossing"] = {"passed": no_gap_crossing, "details": gap_check_details}

    # 2. Chronological order within and across splits
    chrono_details: dict[str, Any] = {}
    within_ok = True
    for name, arrays in seq_arrays.items():
        inc_target = bool(np.all(np.diff(arrays["target_ts"]) > 0))
        inc_start = bool(np.all(np.diff(arrays["window_start_ts"]) > 0))
        within_ok = within_ok and inc_target and inc_start
        chrono_details[name] = {
            "strictly_increasing_target_ts": inc_target,
            "strictly_increasing_start_ts": inc_start,
            "first_window_start": pd.to_datetime(int(arrays["window_start_ts"][0]), unit="s").strftime("%Y-%m-%d %H:%M:%S"),
            "last_target_ts": pd.to_datetime(int(arrays["target_ts"][-1]), unit="s").strftime("%Y-%m-%d %H:%M:%S"),
        }
    train_last = int(seq_arrays["train"]["target_ts"][-1])
    val_first = int(seq_arrays["val"]["window_start_ts"][0])
    val_last = int(seq_arrays["val"]["target_ts"][-1])
    test_first = int(seq_arrays["test"]["window_start_ts"][0])
    cross_split_ok = (train_last < val_first) and (val_last < test_first)
    checks["chronological_order"] = {
        "passed": within_ok and cross_split_ok,
        "cross_split_strictly_ordered": cross_split_ok,
        "details": chrono_details,
    }

    # 3. Zero data leakage (scaler fit strictly on train valid rows; interpolation local to split)
    train_valid_df = splits["train"].loc[splits["train"]["is_valid"], CONTINUOUS_FEATURES]
    expected_n = len(train_valid_df)
    n_seen_ok = (
        int(feature_scaler.n_samples_seen_) == expected_n
        and int(target_scaler.n_samples_seen_) == expected_n
    )
    min_match = bool(np.allclose(feature_scaler.data_min_, train_valid_df.min().to_numpy(), atol=1e-9))
    max_match = bool(np.allclose(feature_scaler.data_max_, train_valid_df.max().to_numpy(), atol=1e-9))
    target_match = bool(
        np.isclose(feature_scaler.min_[0], target_scaler.min_[0], atol=1e-12)
        and np.isclose(feature_scaler.scale_[0], target_scaler.scale_[0], atol=1e-12)
    )
    # Check whether val/test interpolated hours touched split boundaries (already prevented by interior check)
    split_interp_isolated = True
    for name in ("train", "val", "test"):
        s_df = splits[name]
        if s_df["is_interpolated"].iloc[0] or s_df["is_interpolated"].iloc[-1]:
            split_interp_isolated = False
    checks["no_data_leakage"] = {
        "passed": n_seen_ok and min_match and max_match and target_match and split_interp_isolated,
        "scaler_fit_sample_count": int(feature_scaler.n_samples_seen_),
        "expected_train_valid_hours": expected_n,
        "scaler_min_max_exact_match_train": min_match and max_match,
        "target_and_feature_scaler_idx0_identical": target_match,
        "interpolation_strictly_interior_to_split": split_interp_isolated,
        "val_interpolated_count": split_interp_info["val"]["interpolated_hours_count"],
        "test_interpolated_count": split_interp_info["test"]["interpolated_hours_count"],
    }

    # 4. Input feature dimensions & value bounds
    input_ok = True
    input_details: dict[str, Any] = {}
    n_features = len(ALL_FEATURES)
    for name, arrays in seq_arrays.items():
        x = arrays["X"]
        shape_ok = (x.ndim == 3 and x.shape[1] == LOOKBACK_HOURS and x.shape[2] == n_features)
        dtype_ok = (x.dtype == np.float32)
        cyc_ok = bool(np.all((x[:, :, 6:12] >= -1.0001) & (x[:, :, 6:12] <= 1.0001)))
        wknd_ok = bool(np.all(np.isin(x[:, :, 12], [0.0, 1.0])))
        cont_min = float(x[:, :, :6].min())
        cont_max = float(x[:, :, :6].max())
        if name == "train":
            train_bounds_ok = (cont_min >= -1e-6) and (cont_max <= 1.0 + 1e-6)
        else:
            train_bounds_ok = True  # val/test may slightly exceed [0, 1] without clipping/leakage
        split_ok = shape_ok and dtype_ok and cyc_ok and wknd_ok and train_bounds_ok
        input_ok = input_ok and split_ok
        input_details[name] = {
            "shape": list(x.shape),
            "dtype": str(x.dtype),
            "continuous_scaled_min": round(cont_min, 6),
            "continuous_scaled_max": round(cont_max, 6),
            "cyclical_bounds_ok": cyc_ok,
            "weekend_binary_ok": wknd_ok,
        }
    checks["consistent_input_dimensions"] = {"passed": input_ok, "details": input_details}

    # 5. Target dimensions & exact inverse_transform back to original kW
    target_ok = True
    target_details: dict[str, Any] = {}
    for name, arrays in seq_arrays.items():
        ys = arrays["y_scaled"]
        yr = arrays["y_raw_kw"]
        shape_ok = (ys.ndim == 2 and ys.shape[1] == 1 and ys.shape == yr.shape)
        dtype_ok = (ys.dtype == np.float32 and yr.dtype == np.float32)
        inv = target_scaler.inverse_transform(ys.astype(np.float64)).astype(np.float32)
        max_abs_err = float(np.max(np.abs(inv - yr)))
        inv_ok = max_abs_err < 1e-5
        split_ok = shape_ok and dtype_ok and inv_ok
        target_ok = target_ok and split_ok
        target_details[name] = {
            "y_scaled_shape": list(ys.shape),
            "y_raw_kw_shape": list(yr.shape),
            "y_scaled_min": round(float(ys.min()), 6),
            "y_scaled_max": round(float(ys.max()), 6),
            "y_raw_kw_min": round(float(yr.min()), 4),
            "y_raw_kw_max": round(float(yr.max()), 4),
            "max_inverse_transform_abs_err_kw": max_abs_err,
        }
    checks["correct_target_dimensions_and_inverse_kw"] = {"passed": target_ok, "details": target_details}

    # 6. Raw file untouched
    raw_ok = (sha256_before == sha256_after)
    checks["raw_file_untouched"] = {"passed": raw_ok, "sha256": sha256_after}

    all_passed = all(v["passed"] for v in checks.values())
    checks["all_passed"] = all_passed
    return checks


def generate_preprocessing_figures(
    raw_df: pd.DataFrame,
    splits: dict[str, pd.DataFrame],
    seq_arrays: dict[str, dict[str, np.ndarray]],
    seq_stats: dict[str, dict[str, Any]],
) -> list[str]:
    """Generate preprocessing figures in reports/figures/."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_style("whitegrid")
    saved: list[str] = []

    # 7. Hourly dataset chronological splits & un-imputed gap breaks
    fig, ax = plt.subplots(figsize=(15, 5.2))
    colors = {"train": "#1b9e77", "val": "#386cb0", "test": "#d95f02"}
    labels = {"train": "Train (70%)", "val": "Validation (15%)", "test": "Test (15%)"}
    for name in ("train", "val", "test"):
        s_df = splits[name]
        ax.plot(s_df.index, s_df[TARGET_COL], color=colors[name], linewidth=0.45, alpha=0.8,
                label=labels[name], rasterized=True)

    # Highlight split boundaries
    val_start = splits["val"].index[0]
    test_start = splits["test"].index[0]
    ax.axvline(val_start, color="black", linestyle="--", linewidth=1.3, label=f"Val Start ({val_start.strftime('%Y-%m-%d')})")
    ax.axvline(test_start, color="#6a3d9a", linestyle="--", linewidth=1.3, label=f"Test Start ({test_start.strftime('%Y-%m-%d')})")

    # Shade long un-imputed gap breaks (>= 2 hours)
    combined = pd.concat([splits["train"], splits["val"], splits["test"]])
    invalid_mask = (~combined["is_valid"]).to_numpy()
    padded = np.concatenate(([False], invalid_mask, [False]))
    g_starts = np.where((~padded[:-1]) & padded[1:])[0]
    g_ends = np.where(padded[:-1] & (~padded[1:]))[0] - 1
    first_shade = True
    for s, e in zip(g_starts, g_ends):
        if (e - s + 1) >= 2:
            ax.axvspan(
                combined.index[s],
                combined.index[e] + pd.Timedelta(hours=1),
                color="#e41a1c",
                alpha=0.22,
                label="Un-imputed Gap Break" if first_shade else None,
            )
            first_shade = False

    ax.set_title("Hourly Global Active Power: Chronological 70/15/15 Splits & Segment-Break Gaps",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Timestamp (Hourly)")
    ax.set_ylabel("Hourly Mean Global Active Power (kW)")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.legend(loc="upper right", ncol=3, fontsize=8.5)
    fig.tight_layout()
    p7 = FIGURES_DIR / "07_preprocessing_hourly_splits_and_gaps.png"
    fig.savefig(p7, dpi=300, bbox_inches="tight")
    plt.close(fig)
    saved.append(str(p7.relative_to(PROJECT_ROOT)).replace("\\", "/"))

    # 8. Raw 1-min vs Hourly Mean comparison (7-day window: 2007-01-01 to 2007-01-07)
    w_start = pd.Timestamp("2007-01-01 00:00:00")
    w_end = pd.Timestamp("2007-01-07 23:59:59")
    raw_zoom = raw_df.loc[w_start:w_end, TARGET_COL]
    hourly_zoom = splits["train"].loc[w_start:w_end, TARGET_COL]

    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(raw_zoom.index, raw_zoom.values, color="#a6bddb", linewidth=0.6, alpha=0.85,
            label="Raw 1-Minute Observations (kW)")
    ax.step(hourly_zoom.index, hourly_zoom.values, where="post", color="#045a8d", linewidth=1.8,
            label="Hourly Mean Global Active Power (kW)")
    ax.set_title("1-Minute Raw vs. 1-Hour Mean Aggregation (2007-01-01 to 2007-01-07)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Global Active Power (kW)")
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %Y-%m-%d"))
    ax.legend(fontsize=9.5)
    fig.tight_layout()
    p8 = FIGURES_DIR / "08_preprocessing_1min_vs_hourly_comparison.png"
    fig.savefig(p8, dpi=300, bbox_inches="tight")
    plt.close(fig)
    saved.append(str(p8.relative_to(PROJECT_ROOT)).replace("\\", "/"))

    # 9. Scaled target distributions & sequence counts across splits
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5))
    for name in ("train", "val", "test"):
        ys = seq_arrays[name]["y_scaled"].ravel()
        sns.kdeplot(ys, ax=axes[0], color=colors[name], linewidth=2.0,
                    label=f"{labels[name]} (min={ys.min():.3f}, max={ys.max():.3f})")
    axes[0].axvline(0.0, color="gray", linestyle=":", linewidth=1.1)
    axes[0].axvline(1.0, color="gray", linestyle=":", linewidth=1.1)
    axes[0].set_title("Scaled Target Distribution (MinMaxScaler Fit on Train Only)",
                      fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Scaled Global Active Power")
    axes[0].set_ylabel("Density")
    axes[0].legend(fontsize=8.5)

    split_names = ["Train (70%)", "Validation (15%)", "Test (15%)"]
    seq_counts = [seq_stats[k]["total_sequences"] for k in ("train", "val", "test")]
    seg_counts = [seq_stats[k]["used_segments"] for k in ("train", "val", "test")]
    bars = axes[1].bar(split_names, seq_counts, color=[colors["train"], colors["val"], colors["test"]], alpha=0.88)
    axes[1].set_title("Generated Sequences (Lookback=60h, Horizon=1h)", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Number of Sequences (Samples)")
    axes[1].set_ylim(0, max(seq_counts) * 1.15)
    for bar, cnt, seg_c in zip(bars, seq_counts, seg_counts):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(seq_counts) * 0.02,
            f"{cnt:,}\n({seg_c} seg)",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
        )
    fig.tight_layout()
    p9 = FIGURES_DIR / "09_preprocessing_scaling_and_sequences.png"
    fig.savefig(p9, dpi=300, bbox_inches="tight")
    plt.close(fig)
    saved.append(str(p9.relative_to(PROJECT_ROOT)).replace("\\", "/"))

    return saved


def main() -> None:
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"Raw dataset missing at {RAW_DATA_PATH}")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    SCALERS_DIR.mkdir(parents=True, exist_ok=True)

    sha256_before = compute_sha256(RAW_DATA_PATH)

    # 1. Load raw 1-minute dataset (convert '?' and blank strings to NaN)
    raw_df = pd.read_csv(
        RAW_DATA_PATH,
        sep=";",
        na_values=["?", ""],
        low_memory=False,
    )
    raw_df["Datetime"] = pd.to_datetime(
        raw_df["Date"] + " " + raw_df["Time"],
        format="%d/%m/%Y %H:%M:%S",
        errors="raise",
    )
    assert raw_df["Datetime"].is_monotonic_increasing, "Raw timestamps are not monotonic increasing"

    raw_total_rows = int(len(raw_df))
    raw_missing_mask = raw_df[RAW_MEASUREMENT_COLS].isna().any(axis=1).to_numpy()
    raw_missing_rows = int(raw_missing_mask.sum())
    raw_valid_rows = raw_total_rows - raw_missing_rows

    long_raw_gap_hours, long_raw_gaps_list = detect_long_raw_gap_hours(
        raw_missing_mask=raw_missing_mask,
        raw_datetimes=raw_df["Datetime"],
        max_gap_minutes=MAX_INTERPOLATION_GAP_MINUTES,
    )

    raw_df = raw_df.set_index("Datetime")[RAW_MEASUREMENT_COLS]

    # 2. Hourly aggregation (min 45 valid minutes out of 60)
    counts_per_col = {col: raw_df[col].resample("1h").count() for col in RAW_MEASUREMENT_COLS}
    valid_min_count = counts_per_col[TARGET_COL]
    for col in RAW_MEASUREMENT_COLS[1:]:
        assert valid_min_count.equals(counts_per_col[col]), f"Count mismatch in column {col}"

    hourly_mean = raw_df[MEAN_AGG_COLS].resample("1h").mean()
    # Scale sub-meter per-minute Wh mean by 60 -> unbiased hourly Wh energy when count >= 45
    hourly_sub = raw_df[ENERGY_SUM_COLS].resample("1h").mean() * 60.0
    hourly_df = pd.concat([hourly_mean, hourly_sub], axis=1)[RAW_MEASUREMENT_COLS]
    hourly_df["raw_valid_minutes"] = valid_min_count.astype(np.int32)

    invalid_hour_mask = hourly_df["raw_valid_minutes"] < MIN_VALID_MINUTES_PER_HOUR
    hourly_df.loc[invalid_hour_mask, RAW_MEASUREMENT_COLS] = np.nan

    pre_trim_total_hours = int(len(hourly_df))
    pre_trim_valid_hours = int((~invalid_hour_mask).sum())
    pre_trim_invalid_hours = int(invalid_hour_mask.sum())

    # Trim leading/trailing partial boundary hours (< 45 valid min)
    valid_timestamps = hourly_df.index[~invalid_hour_mask]
    trimmed_leading = int((hourly_df.index < valid_timestamps[0]).sum())
    trimmed_trailing = int((hourly_df.index > valid_timestamps[-1]).sum())
    hourly_df = hourly_df.loc[valid_timestamps[0] : valid_timestamps[-1]].copy()

    post_trim_total_hours = int(len(hourly_df))
    post_trim_invalid_hours = int(hourly_df[TARGET_COL].isna().sum())

    # 3. Chronological split (70% / 15% / 15%, no shuffle)
    n_hours = len(hourly_df)
    train_end = int(n_hours * TRAIN_RATIO)
    val_end = int(n_hours * (TRAIN_RATIO + VAL_RATIO))

    raw_splits = {
        "train": hourly_df.iloc[:train_end].copy(),
        "val": hourly_df.iloc[train_end:val_end].copy(),
        "test": hourly_df.iloc[val_end:].copy(),
    }

    # 4. Split-local short-gap interpolation (<= 60 min) + temporal features
    splits: dict[str, pd.DataFrame] = {}
    split_interp_info: dict[str, dict[str, Any]] = {}
    for name, s_df in raw_splits.items():
        s_interp, interp_info = interpolate_short_gaps_within_split(
            split_df=s_df,
            long_raw_gap_hours=long_raw_gap_hours,
            max_gap_minutes=MAX_INTERPOLATION_GAP_MINUTES,
        )
        s_feat = add_temporal_features(s_interp)
        s_feat["split"] = name
        splits[name] = s_feat
        split_interp_info[name] = interp_info

    # 5. Leakage-free MinMaxScaler (fit ONLY on Train valid rows)
    train_valid_mask = splits["train"]["is_valid"].to_numpy()
    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    target_scaler = MinMaxScaler(feature_range=(0, 1))

    feature_scaler.fit(
        splits["train"].loc[train_valid_mask, CONTINUOUS_FEATURES].to_numpy(dtype=np.float64)
    )
    target_scaler.fit(
        splits["train"].loc[train_valid_mask, [TARGET_COL]].to_numpy(dtype=np.float64)
    )

    scaled_continuous_by_split: dict[str, np.ndarray] = {}
    scaled_target_by_split: dict[str, np.ndarray] = {}
    for name, s_df in splits.items():
        v_mask = s_df["is_valid"].to_numpy()
        cont_scaled = np.full((len(s_df), len(CONTINUOUS_FEATURES)), np.nan, dtype=np.float64)
        targ_scaled = np.full(len(s_df), np.nan, dtype=np.float64)
        cont_scaled[v_mask] = feature_scaler.transform(
            s_df.loc[v_mask, CONTINUOUS_FEATURES].to_numpy(dtype=np.float64)
        )
        targ_scaled[v_mask] = target_scaler.transform(
            s_df.loc[v_mask, [TARGET_COL]].to_numpy(dtype=np.float64)
        ).ravel()
        scaled_continuous_by_split[name] = cont_scaled
        scaled_target_by_split[name] = targ_scaled

    # 6. Segment-aware sliding window sequences (lookback=60, horizon=1)
    seq_arrays: dict[str, dict[str, np.ndarray]] = {}
    seq_stats: dict[str, dict[str, Any]] = {}
    for name in ("train", "val", "test"):
        arrays, s_stats = build_sequences_by_segment(
            split_df=splits[name],
            scaled_continuous=scaled_continuous_by_split[name],
            scaled_target=scaled_target_by_split[name],
            lookback=LOOKBACK_HOURS,
            horizon=FORECAST_HORIZON_HOURS,
        )
        seq_arrays[name] = arrays
        seq_stats[name] = s_stats

    sha256_after = compute_sha256(RAW_DATA_PATH)

    # 7. Automated validation suite
    validation = run_validation_checks(
        splits=splits,
        seq_arrays=seq_arrays,
        split_interp_info=split_interp_info,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        sha256_before=sha256_before,
        sha256_after=sha256_after,
    )
    if not validation["all_passed"]:
        raise RuntimeError(f"Validation failed: {json.dumps(validation, indent=2)}")

    # 8. Persist outputs to data/processed/
    combined_hourly = pd.concat([splits["train"], splits["val"], splits["test"]], axis=0)
    combined_hourly.index.name = "Datetime"
    ordered_cols = (
        ALL_FEATURES
        + ["raw_valid_minutes", "is_interpolated", "is_valid", "segment_id", "split"]
    )
    combined_hourly[ordered_cols].to_csv(HOURLY_CSV_PATH, index=True)

    for name, arrays in seq_arrays.items():
        np.savez_compressed(PROCESSED_DIR / f"sequences_{name}.npz", **arrays)

    joblib.dump(feature_scaler, SCALERS_DIR / "feature_scaler.joblib")
    joblib.dump(target_scaler, SCALERS_DIR / "target_scaler.joblib")

    # 9. Generate preprocessing visualizations
    saved_figures = generate_preprocessing_figures(
        raw_df=raw_df,
        splits=splits,
        seq_arrays=seq_arrays,
        seq_stats=seq_stats,
    )

    # 10. Persist preprocessing metadata JSON
    metadata = {
        "task_definition": {
            "target": TARGET_COL,
            "target_meaning": (
                "Konsumsi daya aktif rata-rata rumah tangga (kW) pada 1 jam berikutnya "
                "(mean Global_active_power pada interval [t+1h, t+2h)), bukan menit berikutnya."
            ),
            "resolution": "1h",
            "lookback_hours": LOOKBACK_HOURS,
            "forecast_horizon_hours": FORECAST_HORIZON_HOURS,
        },
        "feature_configuration": {
            "all_features": ALL_FEATURES,
            "num_features": len(ALL_FEATURES),
            "continuous_scaled_features": CONTINUOUS_FEATURES,
            "temporal_unscaled_features": TEMPORAL_FEATURES,
            "excluded_raw_features": EXCLUDED_RAW_COLS,
            "exclusion_rationale": (
                "Global_intensity dikecualikan karena multikolinearitas ekstrem dengan target "
                "Global_active_power (Pearson r = 0.9989, Spearman rho = 0.9954)."
            ),
            "aggregation_rules": {
                "Global_active_power": "mean per 1h (kW)",
                "Global_reactive_power": "mean per 1h (kVAR/kW)",
                "Voltage": "mean per 1h (V)",
                "Sub_metering_1_to_3": "mean_per_min * 60 (Wh per 1h, unbiased untuk >=45/60 menit valid)",
                "min_valid_minutes_per_hour": MIN_VALID_MINUTES_PER_HOUR,
            },
        },
        "counts_before_and_after_aggregation": {
            "raw_1min_total_rows": raw_total_rows,
            "raw_1min_valid_rows": raw_valid_rows,
            "raw_1min_missing_rows": raw_missing_rows,
            "raw_1min_missing_pct": round(raw_missing_rows / raw_total_rows * 100.0, 4),
            "hourly_bins_pre_trim": pre_trim_total_hours,
            "hourly_valid_pre_trim": pre_trim_valid_hours,
            "hourly_invalid_pre_trim": pre_trim_invalid_hours,
            "trimmed_boundary_hours": {"leading": trimmed_leading, "trailing": trimmed_trailing},
            "hourly_bins_post_trim": post_trim_total_hours,
            "hourly_start": hourly_df.index[0].strftime("%Y-%m-%d %H:%M:%S"),
            "hourly_end": hourly_df.index[-1].strftime("%Y-%m-%d %H:%M:%S"),
            "hourly_invalid_before_interp": post_trim_invalid_hours,
            "hourly_interpolated_total": sum(v["interpolated_hours_count"] for v in split_interp_info.values()),
            "hourly_remaining_invalid_segment_breaks": sum(
                v["missing_hours_after_interp"] for v in split_interp_info.values()
            ),
            "hourly_valid_after_preprocessing": sum(
                v["valid_hours_after_interp"] for v in split_interp_info.values()
            ),
            "long_raw_outages_gt_60min": long_raw_gaps_list,
        },
        "splits_and_interpolation": split_interp_info,
        "scaler_summary": {
            "scaler_type": "MinMaxScaler(feature_range=(0, 1))",
            "fit_split": "train_only_valid_rows",
            "n_samples_seen": int(feature_scaler.n_samples_seen_),
            "continuous_features": CONTINUOUS_FEATURES,
            "data_min": [round(float(x), 6) for x in feature_scaler.data_min_],
            "data_max": [round(float(x), 6) for x in feature_scaler.data_max_],
            "target_data_min_kw": round(float(target_scaler.data_min_[0]), 6),
            "target_data_max_kw": round(float(target_scaler.data_max_[0]), 6),
        },
        "sequence_summary": seq_stats,
        "validation_checks": validation,
        "saved_artifacts": {
            "hourly_dataset_csv": str(HOURLY_CSV_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "sequences_train_npz": "data/processed/sequences_train.npz",
            "sequences_val_npz": "data/processed/sequences_val.npz",
            "sequences_test_npz": "data/processed/sequences_test.npz",
            "feature_scaler_joblib": "data/processed/scalers/feature_scaler.joblib",
            "target_scaler_joblib": "data/processed/scalers/target_scaler.joblib",
            "figures": saved_figures,
        },
    }

    with METADATA_PATH.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("Preprocessing & validation complete!")
    print(f"All validation checks passed: {validation['all_passed']}")
    for name in ("train", "val", "test"):
        print(
            f"  {name:5s}: hours={split_interp_info[name]['total_hours']}, "
            f"valid={split_interp_info[name]['valid_hours_after_interp']}, "
            f"interp={split_interp_info[name]['interpolated_hours_count']}, "
            f"gaps_left={split_interp_info[name]['missing_hours_after_interp']}, "
            f"sequences={seq_stats[name]['total_sequences']}, "
            f"X_shape={seq_stats[name]['X_shape']}"
        )


if __name__ == "__main__":
    main()
