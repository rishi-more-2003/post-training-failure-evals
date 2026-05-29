"""Reporting: render plots and a markdown report from a completed run.

Generates:
  plots/composite_failure.png  -- composite failure score per variant
  plots/failure_heatmap.png    -- normalized failure matrix (model x suite)
  plots/reliability.png        -- calibration reliability diagrams per variant
  plots/verbosity_length.png   -- mean output length per variant
  plots/safety.png             -- harmful compliance vs over-refusal
  report.md                    -- human-readable summary with tables + plots
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from pte.utils import setup_logging  # noqa: E402

logger = setup_logging()

_SUITE_ORDER = [
    "sycophancy",
    "verbosity",
    "reward_hacking",
    "calibration",
    "safety",
    "dist_shift",
]


def _models_in_order(summary: dict) -> list[str]:
    return summary["models"]


def _matrix(summary: dict, models: list[str], suites: list[str]) -> np.ndarray:
    lookup = {(r["model"], r["suite"]): r.get("failure_score") for r in summary["rows"]}
    mat = np.full((len(models), len(suites)), np.nan)
    for i, m in enumerate(models):
        for j, s in enumerate(suites):
            v = lookup.get((m, s))
            if v is not None:
                mat[i, j] = v
    return mat


def _plot_composite(summary: dict, path: Path) -> None:
    comp = summary["composite_failure"]
    models = [m for m in _models_in_order(summary) if m in comp]
    vals = [comp[m] for m in models]
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.85, len(models)))
    ax.bar(models, vals, color=colors)
    ax.set_ylabel("Composite failure score (higher = worse)")
    ax.set_title("Aggregate failure-mode risk by model variant")
    ax.set_ylim(0, max(0.5, max(vals) * 1.2 if vals else 0.5))
    for i, v in enumerate(vals):
        ax.text(i, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _plot_heatmap(summary: dict, path: Path) -> None:
    models = _models_in_order(summary)
    suites = [s for s in _SUITE_ORDER if s in summary["suites"]]
    mat = _matrix(summary, models, suites)
    fig, ax = plt.subplots(figsize=(1.4 * len(suites) + 2, 0.8 * len(models) + 2))
    im = ax.imshow(mat, cmap="RdYlGn_r", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(suites)))
    ax.set_xticklabels(suites, rotation=30, ha="right")
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)
    for i in range(len(models)):
        for j in range(len(suites)):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=9)
    ax.set_title("Normalized failure score (0 = safe, 1 = failing)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _plot_reliability(metrics_all: dict, models: list[str], path: Path) -> bool:
    have = [(m, metrics_all[m]["calibration"]) for m in models if "calibration" in metrics_all[m]]
    have = [(m, c) for m, c in have if c.get("reliability_curve")]
    if not have:
        return False
    n = len(have)
    fig, axes = plt.subplots(1, n, figsize=(3.4 * n, 3.4), squeeze=False)
    for ax, (m, cal) in zip(axes[0], have):
        rc = cal["reliability_curve"]
        centers = rc["bin_centers"]
        acc = rc["bin_accuracy"]
        counts = rc["bin_counts"]
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
        xs = [c for c, n_ in zip(centers, counts) if n_ > 0]
        ys = [a for a, n_ in zip(acc, counts) if n_ > 0]
        ax.plot(xs, ys, "o-", color="#d6604d", label="model")
        ax.set_title(f"{m}\nECE={cal['ece']:.3f}")
        ax.set_xlabel("confidence")
        ax.set_ylabel("accuracy")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=7)
    fig.suptitle("Calibration reliability (log-prob confidence)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return True


def _plot_verbosity(metrics_all: dict, models: list[str], path: Path) -> bool:
    rows = [(m, metrics_all[m]["verbosity"]) for m in models if "verbosity" in metrics_all[m]]
    if not rows:
        return False
    labels = [m for m, _ in rows]
    tokens = [v["mean_tokens"] for _, v in rows]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, tokens, color="#4575b4")
    ax.set_ylabel("Mean output length (tokens)")
    ax.set_title("Verbosity: default answer length by variant")
    for i, v in enumerate(tokens):
        ax.text(i, v, f"{v:.0f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return True


def _plot_safety(metrics_all: dict, models: list[str], path: Path) -> bool:
    rows = [(m, metrics_all[m]["safety"]) for m in models if "safety" in metrics_all[m]]
    if not rows:
        return False
    labels = [m for m, _ in rows]
    comply = [v["harmful_compliance_rate"] for _, v in rows]
    over = [v["over_refusal_rate"] for _, v in rows]
    x = np.arange(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - w / 2, comply, w, label="harmful compliance", color="#d73027")
    ax.bar(x + w / 2, over, w, label="benign over-refusal", color="#fc8d59")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("rate")
    ax.set_title("Safety boundary: compliance vs over-refusal")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return True


def _fmt_table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for r in rows:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


def build_report(out_dir: Path, summary: dict, metrics_all: dict) -> Path:
    plots = out_dir / "plots"
    plots.mkdir(exist_ok=True)
    models = _models_in_order(summary)
    suites = summary["suites"]

    _plot_composite(summary, plots / "composite_failure.png")
    _plot_heatmap(summary, plots / "failure_heatmap.png")
    has_rel = _plot_reliability(metrics_all, models, plots / "reliability.png")
    has_verb = _plot_verbosity(metrics_all, models, plots / "verbosity_length.png")
    has_safe = _plot_safety(metrics_all, models, plots / "safety.png")

    comp = summary["composite_failure"]
    ranked = sorted(comp.items(), key=lambda kv: kv[1])

    lines: list[str] = []
    lines.append(f"# Post-Training Failure Eval Report\n")
    lines.append(f"**Run:** `{summary['run_id']}`  ")
    lines.append(f"**Baseline:** `{summary['baseline']}`  ")
    lines.append(f"**Models:** {', '.join(models)}  ")
    lines.append(f"**Suites:** {', '.join(suites)}\n")

    lines.append("## Composite failure ranking\n")
    lines.append("Lower is safer. Composite = mean of normalized per-suite failure scores.\n")
    lines.append(
        _fmt_table(
            ["rank", "model", "composite failure score"],
            [[str(i + 1), m, f"{v:.3f}"] for i, (m, v) in enumerate(ranked)],
        )
    )
    lines.append("\n![composite](plots/composite_failure.png)\n")
    lines.append("![heatmap](plots/failure_heatmap.png)\n")

    # Per-suite headline tables.
    lines.append("## Headline metrics by suite\n")
    by_suite: dict[str, list[dict]] = {}
    for r in summary["rows"]:
        by_suite.setdefault(r["suite"], []).append(r)
    for suite in suites:
        rows = by_suite.get(suite, [])
        if not rows:
            continue
        metric = rows[0]["headline_metric"]
        lines.append(f"### {suite}  \n*headline: `{metric}`*\n")
        body = []
        for r in rows:
            extra = ""
            if suite == "verbosity" and "length_growth_vs_base" in r:
                extra = f" (x{r['length_growth_vs_base']:.2f} vs base)"
            fs = r.get("failure_score")
            body.append(
                [r["model"], f"{r['headline_value']:.3f}{extra}", f"{fs:.3f}" if fs is not None else "-"]
            )
        lines.append(_fmt_table(["model", metric, "norm. failure"], body))
        lines.append("")

    if has_rel:
        lines.append("## Calibration\n![reliability](plots/reliability.png)\n")
    if has_verb:
        lines.append("## Verbosity\n![verbosity](plots/verbosity_length.png)\n")
    if has_safe:
        lines.append("## Safety\n![safety](plots/safety.png)\n")

    lines.append("## How to read this\n")
    lines.append(
        "- **sycophancy** -- agreement with false premises + caving under user pushback.\n"
        "- **verbosity** -- default length, padding rate, and the judge's length preference.\n"
        "- **reward_hacking** -- judge win-rate vs base exceeding factual improvement.\n"
        "- **calibration** -- ECE / overconfidence between confidence and correctness.\n"
        "- **safety** -- harmful compliance (and benign over-refusal as a side signal).\n"
        "- **dist_shift** -- judged quality drop from in-domain to out-of-domain prompts.\n"
    )
    report_path = out_dir / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Report written: {report_path}")
    return report_path
