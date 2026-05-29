import numpy as np

from pte import metrics


def test_ece_perfect_calibration_is_zero():
    # Confidence exactly matches accuracy within each bin (0 conf -> wrong, 1 conf -> right).
    conf = [0.0, 1.0, 0.0, 1.0]
    correct = [False, True, False, True]
    ece, curve = metrics.expected_calibration_error(conf, correct, n_bins=10)
    assert ece < 1e-6
    assert len(curve.bin_centers) == 10


def test_ece_detects_overconfidence():
    conf = [0.99] * 10
    correct = [True, False, False, False, False, True, False, False, False, False]  # 20% acc
    ece, _ = metrics.expected_calibration_error(conf, correct)
    assert ece > 0.5


def test_auroc_perfect_and_random():
    assert metrics.auroc([0.9, 0.8, 0.2, 0.1], [True, True, False, False]) == 1.0
    # Degenerate (all one class) -> 0.5
    assert metrics.auroc([0.5, 0.6], [True, True]) == 0.5


def test_auroc_handles_ties():
    au = metrics.auroc([0.5, 0.5, 0.5, 0.5], [True, False, True, False])
    assert abs(au - 0.5) < 1e-9


def test_brier_bounds():
    assert metrics.brier_score([1.0, 0.0], [True, False]) == 0.0
    assert metrics.brier_score([0.0, 1.0], [True, False]) == 1.0


def test_wilson_interval_contains_point_estimate():
    lo, hi = metrics.wilson_interval(5, 10)
    assert lo < 0.5 < hi
    assert 0.0 <= lo <= hi <= 1.0


def test_wilson_interval_zero_total():
    assert metrics.wilson_interval(0, 0) == (0.0, 0.0)


def test_bootstrap_ci_brackets_mean():
    vals = list(np.random.default_rng(0).normal(0.3, 0.1, 200))
    mean, lo, hi = metrics.bootstrap_ci(vals, n_boot=500, seed=1)
    assert lo <= mean <= hi
