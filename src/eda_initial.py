"""Initial Exploratory Data Analysis (EDA) for UCI Household Electric Power Consumption.

Performs read-only integrity, missingness, temporal, statistical, and forecasting diagnostics
on the raw 1-minute dataset without modifying raw files, imputing, scaling, or resampling.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "household_power_consumption.txt"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
METRICS_PATH = PROJECT_ROOT / "reports" / "eda_metrics.json"

NUMERIC_COLS = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
]

LOOKBACK_STEPS = 60


def compute_sha256(file_path: Path) -> str:
    """Compute SHA-256 hex digest of a file in streaming chunks."""
    hasher = hashlib.sha256()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def inspect_raw_file(file_path: Path) -> dict[str, Any]:
    """Audit raw file encoding, BOM, line endings, delimiter, line count, and sample lines."""
    size_bytes = file_path.stat().st_size
    has_utf8_bom = False
    has_crlf = False
    has_lf_only = False
    line_count = 0
    first_lines: list[str] = []
    last_lines: list[str] = []

    with file_path.open("rb") as f:
        head = f.read(3)
        has_utf8_bom = head == b"\xef\xbb\xbf"
        f.seek(0)
        for raw_line in f:
            line_count += 1
            if raw_line.endswith(b"\r\n"):
                has_crlf = True
            elif raw_line.endswith(b"\n"):
                has_lf_only = True

            decoded = raw_line.decode("utf-8", errors="strict").rstrip("\r\n")
            if line_count <= 6:
                first_lines.append(decoded)
            last_lines.append(decoded)
            if len(last_lines) > 5:
                last_lines.pop(0)

    # Verify strict ASCII compatibility as well
    is_ascii = True
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            try:
                chunk.decode("ascii")
            except UnicodeDecodeError:
                is_ascii = False
                break

    header_cols = first_lines[0].split(";") if first_lines else []
    return {
        "path": str(file_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "size_bytes": size_bytes,
        "size_mb": round(size_bytes / (1024 ** 2), 2),
        "physical_lines": line_count,
        "data_rows_expected_from_lines": line_count - 1,
        "encoding": "ASCII (valid UTF-8)" if is_ascii else "UTF-8",
        "has_utf8_bom": has_utf8_bom,
        "line_ending": "CRLF (\\r\\n)" if has_crlf and not has_lf_only else (
            "Mixed CRLF/LF" if has_crlf and has_lf_only else "LF (\\n)"
        ),
        "delimiter": ";",
        "columns": header_cols,
        "column_count": len(header_cols),
        "first_5_data_lines": first_lines[1:6],
        "last_5_data_lines": last_lines,
    }


def analyze_contiguous_gaps(
    missing_mask: np.ndarray,
    datetimes: pd.Series,
) -> dict[str, Any]:
    """Detect contiguous missing runs and summarize durations and buckets."""
    n = len(missing_mask)
    padded = np.concatenate(([False], missing_mask, [False]))
    starts = np.where((~padded[:-1]) & padded[1:])[0]
    ends = np.where(padded[:-1] & (~padded[1:]))[0] - 1

    gaps = []
    for s, e in zip(starts, ends):
        row_len = int(e - s + 1)
        start_dt = datetimes.iloc[s]
        end_dt = datetimes.iloc[e]
        duration_min = int((end_dt - start_dt).total_seconds() // 60) + 1
        gaps.append({
            "start_idx": int(s),
            "end_idx": int(e),
            "start_datetime": start_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "end_datetime": end_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "row_length": row_len,
            "duration_minutes": duration_min,
            "duration_hours": round(duration_min / 60.0, 2),
            "duration_days": round(duration_min / 1440.0, 2),
        })

    lengths = np.array([g["duration_minutes"] for g in gaps], dtype=np.int64)
    buckets = {
        "1 min": int((lengths == 1).sum()),
        "2-5 min": int(((lengths >= 2) & (lengths <= 5)).sum()),
        "6-60 min": int(((lengths >= 6) & (lengths <= 60)).sum()),
        "61 min - 24h": int(((lengths >= 61) & (lengths <= 1440)).sum()),
        "> 24h": int((lengths > 1440).sum()),
    }
    top_10 = sorted(gaps, key=lambda g: g["duration_minutes"], reverse=True)[:10]

    return {
        "total_contiguous_gaps": len(gaps),
        "total_missing_rows": int(missing_mask.sum()),
        "gap_length_minutes_summary": {
            "min": int(lengths.min()) if len(lengths) else 0,
            "q25": float(np.percentile(lengths, 25)) if len(lengths) else 0.0,
            "median": float(np.median(lengths)) if len(lengths) else 0.0,
            "mean": round(float(lengths.mean()), 2) if len(lengths) else 0.0,
            "q75": float(np.percentile(lengths, 75)) if len(lengths) else 0.0,
            "max": int(lengths.max()) if len(lengths) else 0,
        },
        "buckets": buckets,
        "top_10_longest_gaps": top_10,
    }


def compute_descriptive_stats(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Compute descriptive statistics and outlier counts for each numeric column."""
    result: dict[str, dict[str, Any]] = {}
    for col in NUMERIC_COLS:
        s = df[col].dropna()
        arr = s.to_numpy(dtype=np.float64)
        mean_val = float(arr.mean())
        std_val = float(arr.std(ddof=1))
        q25, median_val, q75 = np.percentile(arr, [25, 50, 75])
        iqr_val = float(q75 - q25)
        lower_bound = q25 - 1.5 * iqr_val
        upper_bound = q75 + 1.5 * iqr_val
        iqr_outliers = int(((arr < lower_bound) | (arr > upper_bound)).sum())

        z_scores = np.abs((arr - mean_val) / std_val) if std_val > 0 else np.zeros_like(arr)
        z_outliers = int((z_scores > 3.0).sum())

        result[col] = {
            "count": int(len(arr)),
            "missing": int(df[col].isna().sum()),
            "mean": round(mean_val, 4),
            "std": round(std_val, 4),
            "min": round(float(arr.min()), 4),
            "q25": round(float(q25), 4),
            "median": round(float(median_val), 4),
            "q75": round(float(q75), 4),
            "max": round(float(arr.max()), 4),
            "iqr": round(iqr_val, 4),
            "cv_pct": round((std_val / mean_val) * 100.0, 2) if mean_val != 0 else None,
            "skewness": round(float(s.skew()), 4),
            "kurtosis_excess": round(float(s.kurtosis()), 4),
            "zero_count": int((arr == 0.0).sum()),
            "negative_count": int((arr < 0.0).sum()),
            "iqr_lower_bound": round(float(lower_bound), 4),
            "iqr_upper_bound": round(float(upper_bound), 4),
            "iqr_outlier_count": iqr_outliers,
            "iqr_outlier_pct": round(iqr_outliers / len(arr) * 100.0, 2),
            "zscore_gt3_count": z_outliers,
            "zscore_gt3_pct": round(z_outliers / len(arr) * 100.0, 2),
        }
    return result


def find_complete_seven_day_window(df: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Find a clean 7-day Monday-to-Sunday window with zero missing values for plotting."""
    valid_per_day = (
        df.assign(date_only=df["Datetime"].dt.floor("D"))
        .groupby("date_only")["Global_active_power"]
        .agg(count="count", missing=lambda s: int(s.isna().sum()))
    )
    dates = valid_per_day.index
    for d in dates:
        if d.dayofweek == 0 and d >= pd.Timestamp("2007-01-01"):
            week_end = d + pd.Timedelta(days=6)
            sub = valid_per_day.loc[d:week_end]
            if len(sub) == 7 and int(sub["missing"].sum()) == 0 and int(sub["count"].sum()) == 7 * 1440:
                return d, week_end + pd.Timedelta(hours=23, minutes=59, seconds=59)
    # Fallback to first 7*1440 rows if no Monday-Sunday block found
    return df["Datetime"].iloc[0], df["Datetime"].iloc[min(len(df) - 1, 7 * 1440)]


def generate_figures(
    df: pd.DataFrame,
    monthly_missing: pd.DataFrame,
    gap_buckets: dict[str, int],
    stats_dict: dict[str, dict[str, Any]],
    pearson_corr: pd.DataFrame,
    acf_1_to_120: list[float],
    key_lag_acf: dict[str, float],
) -> list[str]:
    """Create initial EDA figures in reports/figures/ and return saved relative paths."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_style("whitegrid")
    plt.rcParams["figure.dpi"] = 150
    saved_paths: list[str] = []

    # 1. Missing values overview (monthly missing count + gap duration distribution)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    month_labels = monthly_missing["year_month"].tolist()
    missing_counts = monthly_missing["missing_rows"].tolist()
    x_pos = np.arange(len(month_labels))
    axes[0].bar(x_pos, missing_counts, color="#d95f02", alpha=0.85)
    tick_step = 4
    axes[0].set_xticks(x_pos[::tick_step])
    axes[0].set_xticklabels(month_labels[::tick_step], rotation=45, ha="right", fontsize=8)
    axes[0].set_title("Missing Observations per Month (Dec 2006 - Nov 2010)", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Year-Month")
    axes[0].set_ylabel("Missing 1-Min Rows")

    bucket_names = list(gap_buckets.keys())
    bucket_vals = list(gap_buckets.values())
    bars = axes[1].bar(bucket_names, bucket_vals, color="#7570b3", alpha=0.85)
    axes[1].set_title("Distribution of Contiguous Missing Gap Durations", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Gap Duration Bucket")
    axes[1].set_ylabel("Number of Contiguous Gaps")
    for bar, val in zip(bars, bucket_vals):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, str(val),
                     ha="center", va="bottom", fontsize=9, fontweight="bold")
    fig.tight_layout()
    p1 = FIGURES_DIR / "01_missing_values_overview.png"
    fig.savefig(p1, dpi=300, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(str(p1.relative_to(PROJECT_ROOT)).replace("\\", "/"))

    # 2. Global_active_power raw 1-minute timeline (full 4 years + 7-day zoom)
    w_start, w_end = find_complete_seven_day_window(df)
    zoom_mask = (df["Datetime"] >= w_start) & (df["Datetime"] <= w_end)
    df_zoom = df.loc[zoom_mask]

    fig, axes = plt.subplots(2, 1, figsize=(15, 8))
    axes[0].plot(df["Datetime"], df["Global_active_power"], color="#1b9e77",
                 linewidth=0.2, alpha=0.55, rasterized=True)
    axes[0].set_title("Raw 1-Minute Global Active Power (Dec 2006 - Nov 2010)", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Global Active Power (kW)")
    axes[0].set_xlabel("Date")
    axes[0].xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    axes[1].plot(df_zoom["Datetime"], df_zoom["Global_active_power"], color="#2b8cbe",
                 linewidth=0.6, alpha=0.9)
    axes[1].set_title(
        f"Zoomed 7-Day Raw 1-Minute Profile ({w_start.strftime('%Y-%m-%d')} to {w_end.strftime('%Y-%m-%d')}, 0 Missing)",
        fontsize=12, fontweight="bold",
    )
    axes[1].set_ylabel("Global Active Power (kW)")
    axes[1].set_xlabel("Timestamp")
    axes[1].xaxis.set_major_locator(mdates.DayLocator())
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%a %Y-%m-%d"))
    fig.tight_layout()
    p2 = FIGURES_DIR / "02_global_active_power_timeline.png"
    fig.savefig(p2, dpi=300, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(str(p2.relative_to(PROJECT_ROOT)).replace("\\", "/"))

    # 3. Target distribution & outliers
    gap_valid = df["Global_active_power"].dropna()
    t_stats = stats_dict["Global_active_power"]
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), gridspec_kw={"height_ratios": [3, 1]})
    axes[0].hist(gap_valid, bins=80, density=True, color="#386cb0", alpha=0.65, edgecolor="white", linewidth=0.3)
    sns.kdeplot(x=gap_valid.iloc[::10], ax=axes[0], color="#e41a1c", linewidth=1.8, label="KDE (1:10 sample)")
    axes[0].axvline(t_stats["mean"], color="#d95f02", linestyle="--", linewidth=1.6,
                    label=f"Mean = {t_stats['mean']:.3f} kW")
    axes[0].axvline(t_stats["median"], color="#1b9e77", linestyle="-.", linewidth=1.6,
                    label=f"Median = {t_stats['median']:.3f} kW")
    axes[0].set_title(
        f"Global Active Power Distribution (Skewness = {t_stats['skewness']:.2f}, IQR Outliers = {t_stats['iqr_outlier_pct']:.2f}%)",
        fontsize=12, fontweight="bold",
    )
    axes[0].set_xlabel("Global Active Power (kW)")
    axes[0].set_ylabel("Density")
    axes[0].legend()

    axes[1].boxplot(gap_valid, vert=False, flierprops={"marker": ".", "markersize": 2, "alpha": 0.25})
    axes[1].set_xlabel("Global Active Power (kW)")
    axes[1].set_yticks([])
    axes[1].set_title("Box Plot & Upper-Tail Outliers", fontsize=10)
    fig.tight_layout()
    p3 = FIGURES_DIR / "03_target_distribution_and_outliers.png"
    fig.savefig(p3, dpi=300, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(str(p3.relative_to(PROJECT_ROOT)).replace("\\", "/"))

    # 4. Temporal patterns (Hour of Day, Day of Week, Month of Year)
    valid_df = df.loc[df["Global_active_power"].notna(), ["Datetime", "Global_active_power"]].copy()
    valid_df["hour"] = valid_df["Datetime"].dt.hour
    valid_df["dow"] = valid_df["Datetime"].dt.dayofweek
    valid_df["month"] = valid_df["Datetime"].dt.month

    hourly = valid_df.groupby("hour")["Global_active_power"].agg(
        mean="mean",
        q25=lambda s: np.percentile(s, 25),
        median="median",
        q75=lambda s: np.percentile(s, 75),
    )
    dow_stats = valid_df.groupby("dow")["Global_active_power"].agg(
        mean="mean",
        q25=lambda s: np.percentile(s, 25),
        median="median",
        q75=lambda s: np.percentile(s, 75),
    )
    monthly = valid_df.groupby("month")["Global_active_power"].agg(
        mean="mean",
        q25=lambda s: np.percentile(s, 25),
        median="median",
        q75=lambda s: np.percentile(s, 75),
    )

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    # Hour of day
    axes[0].plot(hourly.index, hourly["mean"], marker="o", markersize=4, color="#1f78b4", label="Mean")
    axes[0].plot(hourly.index, hourly["median"], linestyle="--", color="#33a02c", label="Median")
    axes[0].fill_between(hourly.index, hourly["q25"], hourly["q75"], color="#1f78b4", alpha=0.18, label="IQR (Q25-Q75)")
    axes[0].set_title("By Hour of Day (Diurnal Cycle)", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Hour (0-23)")
    axes[0].set_ylabel("Global Active Power (kW)")
    axes[0].set_xticks(range(0, 24, 3))
    axes[0].legend(fontsize=8)

    # Day of week
    dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    axes[1].bar(range(7), dow_stats["mean"], color="#386cb0", alpha=0.8, label="Mean")
    axes[1].plot(range(7), dow_stats["median"], marker="s", color="#e41a1c", linewidth=1.5, label="Median")
    axes[1].set_title("By Day of Week", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Day of Week")
    axes[1].set_ylabel("Global Active Power (kW)")
    axes[1].set_xticks(range(7))
    axes[1].set_xticklabels(dow_labels)
    axes[1].legend(fontsize=8)

    # Month of year
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    axes[2].plot(monthly.index, monthly["mean"], marker="o", markersize=4, color="#d95f02", label="Mean")
    axes[2].plot(monthly.index, monthly["median"], linestyle="--", color="#7570b3", label="Median")
    axes[2].fill_between(monthly.index, monthly["q25"], monthly["q75"], color="#d95f02", alpha=0.18, label="IQR (Q25-Q75)")
    axes[2].set_title("By Month of Year (Seasonality)", fontsize=11, fontweight="bold")
    axes[2].set_xlabel("Month")
    axes[2].set_ylabel("Global Active Power (kW)")
    axes[2].set_xticks(range(1, 13))
    axes[2].set_xticklabels(month_names, rotation=30)
    axes[2].legend(fontsize=8)

    fig.tight_layout()
    p4 = FIGURES_DIR / "04_temporal_patterns.png"
    fig.savefig(p4, dpi=300, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(str(p4.relative_to(PROJECT_ROOT)).replace("\\", "/"))

    # 5. Correlation heatmap
    fig, ax = plt.subplots(figsize=(9.5, 8))
    sns.heatmap(
        pearson_corr,
        annot=True,
        fmt=".3f",
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0,
        center=0.0,
        square=True,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Pearson Correlation Matrix (7 Raw Numeric Variables)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    p5 = FIGURES_DIR / "05_correlation_heatmap.png"
    fig.savefig(p5, dpi=300, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(str(p5.relative_to(PROJECT_ROOT)).replace("\\", "/"))

    # 6. Autocorrelation (ACF) across lags
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2))
    lags_1_120 = np.arange(1, len(acf_1_to_120) + 1)
    axes[0].plot(lags_1_120, acf_1_to_120, color="#1b9e77", linewidth=1.8)
    axes[0].axvline(LOOKBACK_STEPS, color="#e41a1c", linestyle="--", linewidth=1.5,
                    label=f"Planned Lookback = {LOOKBACK_STEPS} min (ACF = {acf_1_to_120[LOOKBACK_STEPS - 1]:.3f})")
    axes[0].set_title("Autocorrelation (ACF) of Raw 1-Min Target: Lags 1-120 Min", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Lag (minutes)")
    axes[0].set_ylabel("Autocorrelation")
    axes[0].set_ylim(0, 1.02)
    axes[0].legend()

    key_labels = list(key_lag_acf.keys())
    key_vals = list(key_lag_acf.values())
    bars = axes[1].bar(range(len(key_labels)), key_vals, color="#386cb0", alpha=0.85)
    axes[1].set_xticks(range(len(key_labels)))
    axes[1].set_xticklabels(key_labels, rotation=40, ha="right", fontsize=9)
    axes[1].set_title("Multi-Scale ACF: Short-Term Inertia vs. 24h/7d Seasonality", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Lag Horizon")
    axes[1].set_ylabel("Autocorrelation")
    axes[1].set_ylim(0, 1.05)
    for bar, val in zip(bars, key_vals):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015, f"{val:.2f}",
                     ha="center", va="bottom", fontsize=7.5)
    fig.tight_layout()
    p6 = FIGURES_DIR / "06_autocorrelation_lags.png"
    fig.savefig(p6, dpi=300, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(str(p6.relative_to(PROJECT_ROOT)).replace("\\", "/"))

    return saved_paths


def main() -> None:
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"Raw dataset not found at {RAW_DATA_PATH}")

    sha256_before = compute_sha256(RAW_DATA_PATH)
    raw_audit = inspect_raw_file(RAW_DATA_PATH)
    raw_audit["sha256_before"] = sha256_before

    # Inspect pandas dtypes when loaded without na_values (strictly raw)
    df_raw_types = pd.read_csv(RAW_DATA_PATH, sep=";", low_memory=False)
    raw_dtypes_no_na = {col: str(dtype) for col, dtype in df_raw_types.dtypes.items()}
    del df_raw_types

    # Load as strings to count '?' and check for any other non-numeric tokens cleanly
    df = pd.read_csv(RAW_DATA_PATH, sep=";", dtype=str, low_memory=False)
    row_count = int(len(df))

    token_audit: dict[str, dict[str, int]] = {}
    for col in NUMERIC_COLS:
        qmark_count = int((df[col] == "?").sum())
        num_parsed = pd.to_numeric(df[col], errors="coerce")
        total_non_numeric = int(num_parsed.isna().sum())
        other_non_numeric = total_non_numeric - qmark_count
        token_audit[col] = {
            "question_mark_count": qmark_count,
            "other_non_numeric_count": other_non_numeric,
            "total_missing": total_non_numeric,
            "missing_pct": round(total_non_numeric / row_count * 100.0, 4),
        }
        df[col] = num_parsed.astype(np.float64)

    # Parse datetime explicitly with dd/mm/yyyy HH:MM:SS
    df["Datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"], format="%d/%m/%Y %H:%M:%S", errors="raise")

    min_dt = df["Datetime"].min()
    max_dt = df["Datetime"].max()
    expected_1min_rows = int((max_dt - min_dt).total_seconds() // 60) + 1
    dt_diffs = df["Datetime"].diff().dropna().dt.total_seconds().value_counts()

    temporal_audit = {
        "row_count": row_count,
        "min_datetime": min_dt.strftime("%Y-%m-%d %H:%M:%S"),
        "max_datetime": max_dt.strftime("%Y-%m-%d %H:%M:%S"),
        "span_days": round((max_dt - min_dt).total_seconds() / 86400.0, 2),
        "expected_1min_rows_inclusive": expected_1min_rows,
        "row_count_matches_expected_1min_grid": bool(row_count == expected_1min_rows),
        "is_monotonic_increasing": bool(df["Datetime"].is_monotonic_increasing),
        "duplicate_timestamps": int(df["Datetime"].duplicated().sum()),
        "duplicate_full_rows": int(df.duplicated(subset=["Date", "Time"] + NUMERIC_COLS).sum()),
        "timestamp_diff_seconds_counts": {str(int(k)): int(v) for k, v in dt_diffs.items()},
    }

    # Missingness alignment across the 7 measurement columns
    any_missing = df[NUMERIC_COLS].isna().any(axis=1)
    all_missing = df[NUMERIC_COLS].isna().all(axis=1)
    always_missing_simultaneously = bool((any_missing == all_missing).all())

    # Monthly & yearly missingness
    df["year"] = df["Datetime"].dt.year
    df["year_month"] = df["Datetime"].dt.to_period("M").astype(str)

    yearly_summary_df = (
        df.groupby("year")
        .agg(
            total_rows=("Datetime", "count"),
            valid_rows=("Global_active_power", "count"),
            missing_rows=("Global_active_power", lambda s: int(s.isna().sum())),
            mean_kw=("Global_active_power", "mean"),
            std_kw=("Global_active_power", "std"),
            median_kw=("Global_active_power", "median"),
        )
        .reset_index()
    )
    yearly_summary_df["missing_pct"] = (
        yearly_summary_df["missing_rows"] / yearly_summary_df["total_rows"] * 100.0
    ).round(4)

    monthly_missing_df = (
        df.groupby("year_month")
        .agg(
            total_rows=("Datetime", "count"),
            missing_rows=("Global_active_power", lambda s: int(s.isna().sum())),
        )
        .reset_index()
    )
    monthly_missing_df["missing_pct"] = (
        monthly_missing_df["missing_rows"] / monthly_missing_df["total_rows"] * 100.0
    ).round(4)

    gap_analysis = analyze_contiguous_gaps(any_missing.to_numpy(), df["Datetime"])

    # Window impact for lookback=60 (+1 target = 61 contiguous steps)
    miss_int = any_missing.to_numpy(dtype=np.int32)
    csum = np.concatenate(([0], np.cumsum(miss_int)))
    win60_miss = csum[LOOKBACK_STEPS:] - csum[:-LOOKBACK_STEPS]
    win61_miss = csum[LOOKBACK_STEPS + 1:] - csum[:-(LOOKBACK_STEPS + 1)]
    window_impact = {
        "lookback_steps": LOOKBACK_STEPS,
        "total_possible_lookback60_plus_1_windows": int(len(df) - LOOKBACK_STEPS),
        "valid_lookback60_plus_1_windows_without_imputation": int((win61_miss == 0).sum()),
        "affected_lookback60_plus_1_windows": int((win61_miss > 0).sum()),
        "affected_lookback60_plus_1_pct": round(float((win61_miss > 0).mean() * 100.0), 2),
        "affected_lookback60_input_only_windows": int((win60_miss > 0).sum()),
    }

    # Descriptive statistics & domain physical check
    descriptive_stats = compute_descriptive_stats(df)
    valid_rows_df = df.loc[~any_missing, NUMERIC_COLS]
    unmetered_wh = (
        valid_rows_df["Global_active_power"] * 1000.0 / 60.0
        - (valid_rows_df["Sub_metering_1"] + valid_rows_df["Sub_metering_2"] + valid_rows_df["Sub_metering_3"])
    )
    domain_check = {
        "formula": "Global_active_power * 1000 / 60 - (Sub_metering_1 + Sub_metering_2 + Sub_metering_3)",
        "valid_count": int(len(unmetered_wh)),
        "min_wh": round(float(unmetered_wh.min()), 4),
        "q01_wh": round(float(np.percentile(unmetered_wh, 1)), 4),
        "median_wh": round(float(unmetered_wh.median()), 4),
        "mean_wh": round(float(unmetered_wh.mean()), 4),
        "max_wh": round(float(unmetered_wh.max()), 4),
        "negative_count": int((unmetered_wh < 0).sum()),
        "negative_pct": round(float((unmetered_wh < 0).mean() * 100.0), 4),
    }

    # Correlations (Pearson & Spearman)
    pearson_corr = valid_rows_df.corr(method="pearson")
    spearman_corr = valid_rows_df.corr(method="spearman")

    # Autocorrelation on strictly consecutive 1-minute series
    target_series = df["Global_active_power"]
    acf_1_to_120 = [round(float(target_series.autocorr(lag=k)), 4) for k in range(1, 121)]
    key_lags = [
        ("1 min", 1),
        ("2 min", 2),
        ("5 min", 5),
        ("10 min", 10),
        ("15 min", 15),
        ("30 min", 30),
        ("45 min", 45),
        ("60 min (1h)", 60),
        ("120 min (2h)", 120),
        ("360 min (6h)", 360),
        ("720 min (12h)", 720),
        ("1440 min (24h)", 1440),
        ("2880 min (48h)", 2880),
        ("10080 min (7d)", 10080),
    ]
    key_lag_acf = {label: round(float(target_series.autocorr(lag=lag)), 4) for label, lag in key_lags}

    # Calendar breakdowns (in-memory only)
    valid_cal = df.loc[~any_missing, ["Datetime", "Global_active_power"]].copy()
    valid_cal["hour"] = valid_cal["Datetime"].dt.hour
    valid_cal["dow"] = valid_cal["Datetime"].dt.dayofweek
    valid_cal["month"] = valid_cal["Datetime"].dt.month

    hourly_summary = (
        valid_cal.groupby("hour")["Global_active_power"]
        .agg(mean="mean", std="std", median="median")
        .round(4)
        .to_dict(orient="index")
    )
    dow_summary = (
        valid_cal.groupby("dow")["Global_active_power"]
        .agg(mean="mean", std="std", median="median")
        .round(4)
        .to_dict(orient="index")
    )
    monthly_seasonal_summary = (
        valid_cal.groupby("month")["Global_active_power"]
        .agg(mean="mean", std="std", median="median")
        .round(4)
        .to_dict(orient="index")
    )

    # Analytical memory footprint estimates for lookback=60 (float32) — no resampling performed
    def estimate_tensor_mb(n_steps: int, n_features: int) -> float:
        n_windows = max(n_steps - LOOKBACK_STEPS, 0)
        return round((n_windows * LOOKBACK_STEPS * n_features * 4) / (1024 ** 2), 2)

    footprint_estimates = {
        "1min_raw_steps": row_count,
        "1min_materialized_X_univariate_mb": estimate_tensor_mb(row_count, 1),
        "1min_materialized_X_7features_mb": estimate_tensor_mb(row_count, 7),
        "15min_approx_steps": row_count // 15,
        "15min_materialized_X_univariate_mb": estimate_tensor_mb(row_count // 15, 1),
        "15min_materialized_X_7features_mb": estimate_tensor_mb(row_count // 15, 7),
        "1hour_approx_steps": row_count // 60,
        "1hour_materialized_X_univariate_mb": estimate_tensor_mb(row_count // 60, 1),
        "1hour_materialized_X_7features_mb": estimate_tensor_mb(row_count // 60, 7),
    }

    saved_figures = generate_figures(
        df=df,
        monthly_missing=monthly_missing_df,
        gap_buckets=gap_analysis["buckets"],
        stats_dict=descriptive_stats,
        pearson_corr=pearson_corr,
        acf_1_to_120=acf_1_to_120,
        key_lag_acf=key_lag_acf,
    )

    sha256_after = compute_sha256(RAW_DATA_PATH)
    assert sha256_before == sha256_after, "Raw file hash changed during read-only EDA!"
    raw_audit["sha256_after"] = sha256_after
    raw_audit["raw_file_untouched"] = True

    metrics = {
        "raw_file_audit": raw_audit,
        "raw_dtypes_without_na_values": raw_dtypes_no_na,
        "missing_token_audit": token_audit,
        "always_missing_simultaneously_across_7_cols": always_missing_simultaneously,
        "temporal_audit": temporal_audit,
        "gap_analysis": gap_analysis,
        "window_impact_lookback_60": window_impact,
        "descriptive_statistics": descriptive_stats,
        "domain_physical_check": domain_check,
        "pearson_correlation": pearson_corr.round(4).to_dict(),
        "spearman_correlation": spearman_corr.round(4).to_dict(),
        "autocorrelation_key_lags": key_lag_acf,
        "autocorrelation_lags_1_to_60": acf_1_to_120[:60],
        "yearly_summary": yearly_summary_df.round(4).to_dict(orient="records"),
        "monthly_missing_summary": monthly_missing_df.to_dict(orient="records"),
        "hourly_summary": hourly_summary,
        "day_of_week_summary": dow_summary,
        "month_of_year_summary": monthly_seasonal_summary,
        "footprint_estimates_float32": footprint_estimates,
        "saved_figures": saved_figures,
    }

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with METRICS_PATH.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"EDA complete. Saved metrics to {METRICS_PATH}")
    print(f"Saved {len(saved_figures)} figures to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
