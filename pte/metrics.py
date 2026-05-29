"""Statistical metrics used across evaluators.

Pure-numpy implementations (no sklearn dependency):
  - expected_calibration_error / reliability_curve
  - auroc (confidence vs correctness)
  - brier_score
  - rate + wilson_interval (binomial CIs for small samples)
  - bootstrap_ci (generic mean CI)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ReliabilityCurve:
    bin_centers: list[float]
    bin_confidence: list[float]
    bin_accuracy: list[float]
    bin_counts: list[int]


def expected_calibration_error(
    confidences: list[float], correct: list[bool], n_bins: int = 10
) -> tuple[float, ReliabilityCurve]:
    """ECE = sum_b (n_b / N) * |acc_b - conf_b|, plus the reliability curve."""
    conf = np.asarray(confidences, dtype=float)
    acc = np.asarray(correct, dtype=float)
    if len(conf) == 0:
        return 0.0, ReliabilityCurve([], [], [], [])
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    centers, bconf, bacc, bcount = [], [], [], []
    n = len(conf)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        count = int(mask.sum())
        centers.append(float((lo + hi) / 2))
        if count == 0:
            bconf.append(0.0)
            bacc.append(0.0)
            bcount.append(0)
            continue
        c_mean = float(conf[mask].mean())
        a_mean = float(acc[mask].mean())
        ece += (count / n) * abs(a_mean - c_mean)
        bconf.append(c_mean)
        bacc.append(a_mean)
        bcount.append(count)
    return ece, ReliabilityCurve(centers, bconf, bacc, bcount)


def brier_score(confidences: list[float], correct: list[bool]) -> float:
    conf = np.asarray(confidences, dtype=float)
    acc = np.asarray(correct, dtype=float)
    if len(conf) == 0:
        return 0.0
    return float(np.mean((conf - acc) ** 2))


def auroc(scores: list[float], labels: list[bool]) -> float:
    """AUROC via the rank-sum (Mann-Whitney) identity. Returns 0.5 if degenerate."""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=bool)
    n_pos = int(y.sum())
    n_neg = int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)
    # Average ranks for ties.
    _, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    cum = np.cumsum(counts)
    start = cum - counts
    avg_rank = (start + cum + 1) / 2.0
    ranks = avg_rank[inv]
    sum_pos = ranks[y].sum()
    return float((sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def rate(successes: int, total: int) -> float:
    return successes / total if total else 0.0


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (good for small n)."""
    if total == 0:
        return (0.0, 0.0)
    p = successes / total
    denom = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denom
    margin = (z * np.sqrt(p * (1 - p) / total + z**2 / (4 * total**2))) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def bootstrap_ci(
    values: list[float], n_boot: int = 2000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float, float]:
    """Return (mean, lo, hi) percentile bootstrap CI for the mean of `values`."""
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return (0.0, 0.0, 0.0)
    rng = np.random.default_rng(seed)
    means = arr[rng.integers(0, len(arr), size=(n_boot, len(arr)))].mean(axis=1)
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return (float(arr.mean()), lo, hi)
