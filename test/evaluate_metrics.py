#!/usr/bin/env python3
"""Compare timing CSV metrics from f_vigs_slam runs.

Usage:
  ./test/evaluate_metrics.py --baseline /path/to/baseline.csv --candidate /path/to/candidate.csv

The script prints numeric summaries and a simple verdict for timing metrics.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence


# ---------------------------------------------------------------------------
# Metric composition reference (CSV columns)
#
# Identification / frame context:
#   baseline_tag, frame_index, rgb_stamp_s, imu_interval_n, imu_dt_min_s,
#   imu_dt_max_s, imu_lag_s, gaussians_count, frame_width, frame_height
#
# Core timings (inside GSSlam compute):
#   core_total_ms, core_lock_wait_ms, core_gpu_total_ms, core_init_copy_ms,
#   core_predict_ms, core_optimize_track_ms, core_remove_outliers_ms,
#   core_keyframe_ms, core_keyframe_refine_ms, core_keyframe_prune_ms,
#   core_keyframe_add_ms, core_keyframe_densify_ms, core_marginalization_ms
#
# Node-level timings:
#   node_total_ms, node_imu_apply_ms, node_compute_call_ms, node_odom_pub_ms,
#   node_pointcloud_pub_ms, node_unaccounted_ms, node_unaccounted_minus_lock_ms
#
# State jump diagnostics:
#   state_jump_dpos_m, state_jump_drot_deg, jump_imu_apply_dpos_m,
#   jump_imu_apply_drot_deg, jump_compute_dpos_m, jump_compute_drot_deg,
#   state_jump_source_stage, keyframe_added, jump_source_stage,
#   jump_source_dpos_m, jump_source_drot_deg
#
# Bias / validity diagnostics:
#   bias_ba_norm, bias_bg_norm, bias_ba_delta_norm, bias_bg_delta_norm,
#   invalid_gaussian_pos_count, invalid_gaussian_scale_count,
#   invalid_gaussian_opacity_count
#
# To analyze custom columns, add names to --columns when invoking this script.
# ---------------------------------------------------------------------------


@dataclass
class MetricSummary:
    count: int
    avg: float
    median: float
    p95: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare f_vigs_slam timing CSV metrics")
    parser.add_argument("--baseline", required=True, help="Path to baseline CSV")
    parser.add_argument("--candidate", required=True, help="Path to candidate/new CSV")
    parser.add_argument(
        "--columns",
        nargs="+",
        default=[
            "core_total_ms",
            "core_gpu_total_ms",
            "core_lock_wait_ms",
            "core_init_copy_ms",
            "core_optimize_track_ms",
            "core_keyframe_densify_ms",
            "core_marginalization_ms",
            "node_total_ms",
            "node_compute_call_ms",
            "gaussians_count",
        ],
        help="Columns to compare (must exist in both CSVs)",
    )
    return parser.parse_args()


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError(f"CSV has no data rows: {path}")
    return rows


def _to_float(values: Sequence[str]) -> List[float]:
    out: List[float] = []
    for v in values:
        if v is None or v == "":
            continue
        try:
            out.append(float(v))
        except ValueError:
            continue
    return out


def _p95(sorted_values: Sequence[float]) -> float:
    if not sorted_values:
        return float("nan")
    idx = int((len(sorted_values) - 1) * 0.95)
    return sorted_values[idx]


def summarize(values: Sequence[float]) -> MetricSummary:
    if not values:
        return MetricSummary(count=0, avg=float("nan"), median=float("nan"), p95=float("nan"))
    ordered = sorted(values)
    return MetricSummary(
        count=len(values),
        avg=statistics.fmean(values),
        median=statistics.median(values),
        p95=_p95(ordered),
    )


def fmt(value: float) -> str:
    if math.isnan(value):
        return "nan"
    return f"{value:.3f}"


def verdict_for_metric(metric: str, delta_pct: float) -> str:
    if math.isnan(delta_pct):
        return "n/a"
    # For timing metrics (*_ms): lower is better.
    if metric.endswith("_ms"):
        if delta_pct < 0:
            return "improved"
        if delta_pct > 0:
            return "regressed"
        return "same"
    return "numeric-only"


def compare_metric(rows_a: Sequence[Dict[str, str]], rows_b: Sequence[Dict[str, str]], metric: str) -> Dict[str, object]:
    vals_a = _to_float([r.get(metric, "") for r in rows_a])
    vals_b = _to_float([r.get(metric, "") for r in rows_b])
    s_a = summarize(vals_a)
    s_b = summarize(vals_b)

    delta_avg = s_b.avg - s_a.avg
    if s_a.avg == 0 or math.isnan(s_a.avg) or math.isnan(s_b.avg):
        delta_pct = float("nan")
    else:
        delta_pct = (delta_avg / s_a.avg) * 100.0

    return {
        "metric": metric,
        "is_timing": metric.endswith("_ms"),
        "baseline": s_a,
        "candidate": s_b,
        "delta_avg": delta_avg,
        "delta_pct": delta_pct,
        "verdict": verdict_for_metric(metric, delta_pct),
    }


def _is_non_keyframe_row(row: Dict[str, str]) -> bool:
    raw = (row.get("keyframe_added", "") or "").strip().lower()
    return raw in ("0", "false", "no")


def filter_non_keyframes(rows: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    return [r for r in rows if _is_non_keyframe_row(r)]


def compare_metric_per_1k_gaussians(
    rows_a: Sequence[Dict[str, str]],
    rows_b: Sequence[Dict[str, str]],
    metric: str,
) -> Dict[str, object]:
    def normalized_values(rows: Sequence[Dict[str, str]]) -> List[float]:
        vals: List[float] = []
        for r in rows:
            try:
                g = float(r.get("gaussians_count", ""))
                v = float(r.get(metric, ""))
            except (TypeError, ValueError):
                continue
            if g > 0:
                vals.append((v * 1000.0) / g)
        return vals

    s_a = summarize(normalized_values(rows_a))
    s_b = summarize(normalized_values(rows_b))

    delta_avg = s_b.avg - s_a.avg
    if s_a.avg == 0 or math.isnan(s_a.avg) or math.isnan(s_b.avg):
        delta_pct = float("nan")
    else:
        delta_pct = (delta_avg / s_a.avg) * 100.0

    metric_name = f"{metric}/1k_g"
    return {
        "metric": metric_name,
        "is_timing": metric.endswith("_ms"),
        "baseline": s_a,
        "candidate": s_b,
        "delta_avg": delta_avg,
        "delta_pct": delta_pct,
        "verdict": verdict_for_metric(metric, delta_pct),
    }


def compare_metric_weighted_per_1k_gaussians(
    rows_a: Sequence[Dict[str, str]],
    rows_b: Sequence[Dict[str, str]],
    metric: str,
) -> Dict[str, object]:
    def weighted_value(rows: Sequence[Dict[str, str]]) -> float:
        sum_ms = 0.0
        sum_g = 0.0
        for r in rows:
            try:
                g = float(r.get("gaussians_count", ""))
                ms = float(r.get(metric, ""))
            except (TypeError, ValueError):
                continue
            if g > 0:
                sum_ms += ms
                sum_g += g
        if sum_g <= 0.0:
            return float("nan")
        return (1000.0 * sum_ms) / sum_g

    base_w = weighted_value(rows_a)
    cand_w = weighted_value(rows_b)
    delta = cand_w - base_w
    if base_w == 0 or math.isnan(base_w) or math.isnan(cand_w):
        delta_pct = float("nan")
    else:
        delta_pct = (delta / base_w) * 100.0

    return {
        "metric": f"{metric}/W1k_g",
        "base": base_w,
        "cand": cand_w,
        "delta_pct": delta_pct,
        "verdict": verdict_for_metric(metric, delta_pct),
    }


def print_weighted_report(results: Sequence[Dict[str, object]]) -> None:
    print("metric                 base_w1k cand_w1k  delta_%    verdict")
    print("--------------------------------------------------------------")

    for item in results:
        metric_name = str(item["metric"])
        if len(metric_name) > 22:
            metric_name = metric_name[:21] + "~"
        print(
            f"{metric_name:<22}"
            f" {fmt(float(item['base'])):>8}"
            f" {fmt(float(item['cand'])):>8}"
            f" {fmt(float(item['delta_pct'])):>8}"
            f" {str(item['verdict']):>10}"
        )

    timing_items = [r for r in results if str(r["verdict"]) in ("improved", "regressed", "same")]
    improved = sum(1 for r in timing_items if r["verdict"] == "improved")
    regressed = sum(1 for r in timing_items if r["verdict"] == "regressed")
    print()
    print(
        "weighted timing verdict summary: "
        f"improved={improved} regressed={regressed} total_timing_metrics={len(timing_items)}"
    )


def print_report(results: Sequence[Dict[str, object]], baseline_rows: int, candidate_rows: int) -> None:
    print("f_vigs_slam metrics comparison")
    print(f"rows: baseline={baseline_rows} candidate={candidate_rows}")
    print()

    def short_metric_name(name: str, max_len: int = 22) -> str:
        if len(name) <= max_len:
            return name
        return name[: max_len - 1] + "~"

    # Table 1: averages + verdict (fits narrow terminals).
    row_format_main = (
        "{metric:<22} "
        "{base_avg:>8} {cand_avg:>8} "
        "{delta_pct:>8} {verdict:>10}"
    )
    print(
        row_format_main.format(
            metric="metric",
            base_avg="base_avg",
            cand_avg="cand_avg",
            delta_pct="delta_%",
            verdict="verdict",
        )
    )
    print("-" * 62)

    for item in results:
        b = item["baseline"]
        c = item["candidate"]
        assert isinstance(b, MetricSummary)
        assert isinstance(c, MetricSummary)
        delta_pct = item["delta_pct"]
        print(
            row_format_main.format(
                metric=short_metric_name(str(item["metric"])),
                base_avg=fmt(b.avg),
                cand_avg=fmt(c.avg),
                delta_pct=fmt(delta_pct),
                verdict=str(item["verdict"]),
            )
        )

    print()

    # Table 2: p95 tail behavior.
    row_format_tail = "{metric:<22} {base_p95:>9} {cand_p95:>9}"
    print(row_format_tail.format(metric="metric", base_p95="base_p95", cand_p95="cand_p95"))
    print("-" * 44)
    for item in results:
        b = item["baseline"]
        c = item["candidate"]
        assert isinstance(b, MetricSummary)
        assert isinstance(c, MetricSummary)
        print(
            row_format_tail.format(
                metric=short_metric_name(str(item["metric"])),
                base_p95=fmt(b.p95),
                cand_p95=fmt(c.p95),
            )
        )

    timing_items = [r for r in results if bool(r.get("is_timing", False))]
    improved = sum(1 for r in timing_items if r["verdict"] == "improved")
    regressed = sum(1 for r in timing_items if r["verdict"] == "regressed")
    print()
    print(
        "timing verdict summary: "
        f"improved={improved} regressed={regressed} total_timing_metrics={len(timing_items)}"
    )


def main() -> int:
    args = parse_args()
    baseline_path = Path(args.baseline)
    candidate_path = Path(args.candidate)

    if not baseline_path.exists():
        raise FileNotFoundError(f"Baseline CSV not found: {baseline_path}")
    if not candidate_path.exists():
        raise FileNotFoundError(f"Candidate CSV not found: {candidate_path}")

    baseline_rows = _read_csv(baseline_path)
    candidate_rows = _read_csv(candidate_path)

    shared_cols = set(baseline_rows[0].keys()) & set(candidate_rows[0].keys())
    columns = [c for c in args.columns if c in shared_cols]
    missing = [c for c in args.columns if c not in shared_cols]

    if missing:
        print("warning: skipping missing columns:", ", ".join(missing))
    if not columns:
        raise ValueError("No comparable columns found. Check --columns and CSV headers.")

    results = [compare_metric(baseline_rows, candidate_rows, col) for col in columns]

    # Also report FPS proxy from node_total_ms if available.
    if "node_total_ms" in shared_cols:
        base_ms = summarize(_to_float([r.get("node_total_ms", "") for r in baseline_rows])).avg
        cand_ms = summarize(_to_float([r.get("node_total_ms", "") for r in candidate_rows])).avg
        if base_ms > 0 and cand_ms > 0:
            print(f"baseline_avg_fps={1000.0 / base_ms:.3f}")
            print(f"candidate_avg_fps={1000.0 / cand_ms:.3f}")
            print()

    print_report(results, len(baseline_rows), len(candidate_rows))

    # Extra block 1: non-keyframe-only comparison to isolate tracking behavior.
    if "keyframe_added" in shared_cols:
        base_nk = filter_non_keyframes(baseline_rows)
        cand_nk = filter_non_keyframes(candidate_rows)
        print()
        print("[non-keyframe-only analysis]")
        if base_nk and cand_nk:
            non_kf_results = [compare_metric(base_nk, cand_nk, col) for col in columns]
            print_report(non_kf_results, len(base_nk), len(cand_nk))
        else:
            print(
                "not enough non-keyframe rows: "
                f"baseline={len(base_nk)} candidate={len(cand_nk)}"
            )

    # Extra block 2: normalized timing per 1k gaussians to factor workload size.
    if "gaussians_count" in shared_cols:
        timing_cols = [c for c in columns if c.endswith("_ms")]
        if timing_cols:
            print()
            print("[normalized timing: ms per 1k gaussians]")
            norm_results = [
                compare_metric_per_1k_gaussians(baseline_rows, candidate_rows, c)
                for c in timing_cols
            ]
            print_report(norm_results, len(baseline_rows), len(candidate_rows))

            print()
            print("[weighted normalized timing: 1000*sum(ms)/sum(gaussians_count)]")
            weighted_results = [
                compare_metric_weighted_per_1k_gaussians(baseline_rows, candidate_rows, c)
                for c in timing_cols
            ]
            print_weighted_report(weighted_results)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
