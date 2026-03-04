#!/usr/bin/env python3
"""
Drift Detection Report (PSI + KL)

Usage:
  python drift_report.py --baseline baseline.csv --live live.csv --out drift_report.html

Expected inputs:
  baseline.csv and live.csv must both contain numeric columns for the same features
  (e.g., Time, V1..V28, Amount). Extra columns are ignored if not shared.
"""

from __future__ import annotations

import argparse
import math
import html
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


EPS = 1e-12


@dataclass
class Thresholds:
    psi_warn: float = 0.1
    psi_drift: float = 0.25
    kl_warn: float = 0.05
    kl_drift: float = 0.1
    overall_warn_ratio: float = 0.10   # if >=10% features drift/warn -> overall warn
    overall_drift_ratio: float = 0.20  # if >=20% features drift -> overall drift


def safe_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a / np.clip(b, EPS, None)


def get_shared_numeric_columns(baseline: pd.DataFrame, live: pd.DataFrame) -> List[str]:
    # keep only numeric shared columns
    shared = [c for c in baseline.columns if c in live.columns]
    shared_numeric = []
    for c in shared:
        if pd.api.types.is_numeric_dtype(baseline[c]) and pd.api.types.is_numeric_dtype(live[c]):
            shared_numeric.append(c)
    return shared_numeric


def quantile_bins(series: pd.Series, n_bins: int) -> np.ndarray:
    # Quantile-based edges; ensure unique and strictly increasing
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(series.dropna().values, qs)
    edges = np.unique(edges)
    if len(edges) < 3:
        # fallback to min/max with padding
        vmin = float(series.min())
        vmax = float(series.max())
        if math.isclose(vmin, vmax):
            vmin -= 1.0
            vmax += 1.0
        edges = np.linspace(vmin, vmax, n_bins + 1)
    # Expand ends slightly to include all values
    edges[0] = edges[0] - 1e-9
    edges[-1] = edges[-1] + 1e-9
    return edges


def hist_probs(series: pd.Series, edges: np.ndarray) -> np.ndarray:
    counts, _ = np.histogram(series.dropna().values, bins=edges)
    probs = counts.astype(np.float64)
    probs = probs / max(probs.sum(), 1.0)
    return probs


def psi(expected_p: np.ndarray, actual_p: np.ndarray) -> float:
    e = np.clip(expected_p, EPS, None)
    a = np.clip(actual_p, EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    # KL(p || q)
    p = np.clip(p, EPS, None)
    q = np.clip(q, EPS, None)
    return float(np.sum(p * np.log(p / q)))


def classify(value: float, warn: float, drift: float) -> str:
    if value >= drift:
        return "DRIFT"
    if value >= warn:
        return "WARN"
    return "OK"


def compute_feature_metrics(
    baseline: pd.Series,
    live: pd.Series,
    n_bins: int
) -> Tuple[float, float]:
    edges = quantile_bins(baseline, n_bins=n_bins)
    p_base = hist_probs(baseline, edges)
    p_live = hist_probs(live, edges)
    psi_v = psi(p_base, p_live)
    kl_v = kl_divergence(p_live, p_base)  # live vs baseline
    return psi_v, kl_v


def overall_status(rows: List[Dict], thr: Thresholds) -> str:
    n = len(rows)
    if n == 0:
        return "OK"
    drift_count = sum(1 for r in rows if r["psi_status"] == "DRIFT" or r["kl_status"] == "DRIFT")
    warn_or_drift = sum(1 for r in rows if r["psi_status"] != "OK" or r["kl_status"] != "OK")

    drift_ratio = drift_count / n
    warn_ratio = warn_or_drift / n

    if drift_ratio >= thr.overall_drift_ratio:
        return "DRIFT"
    if warn_ratio >= thr.overall_warn_ratio:
        return "WARN"
    return "OK"


def drift_score(rows: List[Dict]) -> float:
    """
    Simple scoring:
      - normalize PSI and KL by capping
      - average per-feature and then overall
    """
    if not rows:
        return 0.0
    per = []
    for r in rows:
        psi_n = min(float(r["psi"]), 1.0)          # cap PSI at 1.0
        kl_n = min(float(r["kl"]), 1.0)            # cap KL at 1.0
        per.append(0.7 * psi_n + 0.3 * kl_n)       # weighted
    return float(np.mean(per))


def render_html(
    rows: List[Dict],
    out_path: str,
    thr: Thresholds,
    baseline_n: int,
    live_n: int,
    overall: str,
    score: float,
) -> None:
    def badge(status: str) -> str:
        color = {"OK": "#1a7f37", "WARN": "#9a6700", "DRIFT": "#cf222e"}.get(status, "#57606a")
        return f'<span style="padding:2px 8px;border-radius:999px;background:{color};color:white;font-weight:600;">{html.escape(status)}</span>'

    table_rows = []
    for r in rows:
        table_rows.append(
            "<tr>"
            f"<td>{html.escape(r['feature'])}</td>"
            f"<td>{r['psi']:.6f}</td>"
            f"<td>{badge(r['psi_status'])}</td>"
            f"<td>{r['kl']:.6f}</td>"
            f"<td>{badge(r['kl_status'])}</td>"
            "</tr>"
        )

    doc = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Drift Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; }}
    .card {{ border: 1px solid #e5e7eb; border-radius: 12px; padding: 16px; margin-bottom: 16px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border-bottom: 1px solid #eee; padding: 10px; text-align: left; }}
    th {{ background: #fafafa; }}
    .muted {{ color: #6b7280; }}
    code {{ background: #f3f4f6; padding: 2px 6px; border-radius: 6px; }}
  </style>
</head>
<body>
  <h1>Drift Detection Report</h1>
  <div class="card">
    <p><b>Overall Status:</b> {badge(overall)}</p>
    <p class="muted">Baseline rows: <code>{baseline_n}</code> | Live rows: <code>{live_n}</code></p>
    <p class="muted">Drift Score (0-1 capped): <code>{score:.4f}</code></p>
  </div>

  <div class="card">
    <h2>Thresholds</h2>
    <ul>
      <li>PSI warn: <code>{thr.psi_warn}</code>, PSI drift: <code>{thr.psi_drift}</code></li>
      <li>KL warn: <code>{thr.kl_warn}</code>, KL drift: <code>{thr.kl_drift}</code></li>
      <li>Overall warn ratio: <code>{thr.overall_warn_ratio}</code>, Overall drift ratio: <code>{thr.overall_drift_ratio}</code></li>
    </ul>
  </div>

  <div class="card">
    <h2>Feature-level Metrics</h2>
    <table>
      <thead>
        <tr>
          <th>Feature</th>
          <th>PSI</th>
          <th>PSI Status</th>
          <th>KL (live || baseline)</th>
          <th>KL Status</th>
        </tr>
      </thead>
      <tbody>
        {''.join(table_rows)}
      </tbody>
    </table>
  </div>
</body>
</html>
"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(doc)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True, help="Path to baseline CSV (training distribution)")
    ap.add_argument("--live", required=True, help="Path to live CSV (recent prediction features)")
    ap.add_argument("--out", default="drift_report.html", help="Output HTML report path")
    ap.add_argument("--bins", type=int, default=10, help="Number of bins for histograms")
    ap.add_argument("--psi-warn", type=float, default=0.1)
    ap.add_argument("--psi-drift", type=float, default=0.25)
    ap.add_argument("--kl-warn", type=float, default=0.05)
    ap.add_argument("--kl-drift", type=float, default=0.1)
    args = ap.parse_args()

    thr = Thresholds(
        psi_warn=args.psi_warn,
        psi_drift=args.psi_drift,
        kl_warn=args.kl_warn,
        kl_drift=args.kl_drift,
    )

    baseline = pd.read_csv(args.baseline)
    live = pd.read_csv(args.live)

    features = get_shared_numeric_columns(baseline, live)
    if not features:
        raise SystemExit("No shared numeric columns found between baseline and live CSVs.")

    rows: List[Dict] = []
    for feat in features:
        psi_v, kl_v = compute_feature_metrics(baseline[feat], live[feat], n_bins=args.bins)
        rows.append({
            "feature": feat,
            "psi": psi_v,
            "kl": kl_v,
            "psi_status": classify(psi_v, thr.psi_warn, thr.psi_drift),
            "kl_status": classify(kl_v, thr.kl_warn, thr.kl_drift),
        })

    # sort most drifted first (by psi then kl)
    rows.sort(key=lambda r: (r["psi"], r["kl"]), reverse=True)

    overall = overall_status(rows, thr)
    score = drift_score(rows)

    render_html(
        rows=rows,
        out_path=args.out,
        thr=thr,
        baseline_n=len(baseline),
        live_n=len(live),
        overall=overall,
        score=score,
    )

    print(f"[OK] Wrote report: {args.out}")
    print(f"Overall status: {overall} | Drift score: {score:.4f}")
    top = rows[:5]
    print("Top 5 drifted features:")
    for r in top:
        print(f"  {r['feature']}: PSI={r['psi']:.4f}({r['psi_status']}), KL={r['kl']:.4f}({r['kl_status']})")


if __name__ == "__main__":
    main()