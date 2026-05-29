"""Orchestration: run the selected evaluators across all model variants.

Outputs (under <output_dir>/<run_id>/):
  config.json          -- snapshot of the resolved config
  raw/<model>__<suite>.jsonl  -- per-example records for auditing
  metrics.json         -- full metric dict for every (model, suite)
  summary.json         -- headline + normalized failure matrix + composite scores
  report.md, plots/    -- written by pte.report
"""

from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import tinker
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn

from pte.config import EvalConfig
from pte.evaluators import EvalContext, all_evaluator_names, get_evaluator
from pte.judge import Judge
from pte.models import ModelClient
from pte.utils import setup_logging, write_json, write_jsonl

logger = setup_logging()


def normalized_failure(suite: str, metrics: dict) -> float | None:
    """Map a suite's metrics to a [0,1] 'failure' value (higher = worse).

    Used only for the composite leaderboard; raw metrics remain authoritative.
    """
    if metrics.get("skipped"):
        return None
    if suite == "sycophancy":
        return float(metrics["sycophancy_rate"])
    if suite == "calibration":
        return float(min(1.0, metrics["ece"] * 2.0))  # ECE rarely exceeds 0.5
    if suite == "verbosity":
        return float(metrics["padding_rate"])
    if suite == "reward_hacking":
        return float(min(1.0, max(0.0, metrics["reward_hacking_score"])))
    if suite == "safety":
        return float(metrics["harmful_compliance_rate"])
    if suite == "dist_shift":
        return float(metrics["quality_drop"])
    return None


def _select_baseline(models: list[ModelClient]) -> ModelClient | None:
    for m in models:
        if m.spec.stage == "base":
            return m
    return models[0] if models else None


def run_eval(cfg: EvalConfig) -> dict:
    suites = cfg.suites or all_evaluator_names()
    run_id = time.strftime("%Y%m%d-%H%M%S")
    out_dir = Path(cfg.output_dir) / run_id
    (out_dir / "raw").mkdir(parents=True, exist_ok=True)

    logger.info(f"Run {run_id}: {len(cfg.models)} models x {len(suites)} suites -> {out_dir}")

    service = tinker.ServiceClient()
    models = [ModelClient(spec, service_client=service) for spec in cfg.models]
    baseline = _select_baseline(models)
    judge = Judge(cfg.judge, service_client=service)
    logger.info(
        f"Models: {[m.name for m in models]} | baseline={baseline.name if baseline else None} "
        f"| judge={cfg.judge.base_model}"
    )

    metrics_all: dict[str, dict[str, dict]] = {}
    summary_rows: list[dict] = []

    total_tasks = len(models) * len(suites)
    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("Evaluating", total=total_tasks)
        for model in models:
            metrics_all.setdefault(model.name, {})
            for suite in suites:
                evaluator = get_evaluator(suite)
                ctx = EvalContext(
                    model=model,
                    judge=judge,
                    cfg=cfg,
                    baseline=baseline,
                    rng=np.random.default_rng(cfg.seed),
                )
                progress.update(task, description=f"{model.name} / {suite}")
                try:
                    result = evaluator.run(ctx)
                except Exception as e:  # keep the matrix filled even if one cell fails
                    logger.exception(f"{model.name}/{suite} failed: {e}")
                    progress.advance(task)
                    continue

                metrics_all[model.name][suite] = result.metrics
                if result.records:
                    write_jsonl(out_dir / "raw" / f"{model.name}__{suite}.jsonl", result.records)
                nf = normalized_failure(suite, result.metrics)
                summary_rows.append(
                    {
                        "model": model.name,
                        "stage": result.stage,
                        "suite": suite,
                        "headline_metric": result.headline_metric,
                        "headline_value": result.headline_value,
                        "failure_score": nf,
                    }
                )
                progress.advance(task)

    # Length growth vs base (verbosity context).
    base_tokens = None
    for r in summary_rows:
        if r["suite"] == "verbosity" and r["stage"] == "base":
            base_tokens = metrics_all[r["model"]]["verbosity"]["mean_tokens"]
    if base_tokens:
        for r in summary_rows:
            if r["suite"] == "verbosity":
                mt = metrics_all[r["model"]]["verbosity"]["mean_tokens"]
                r["length_growth_vs_base"] = (mt / base_tokens) if base_tokens else 1.0

    composites = _composite_scores(summary_rows)

    write_json(out_dir / "config.json", _config_snapshot(cfg))
    write_json(out_dir / "metrics.json", metrics_all)
    summary = {
        "run_id": run_id,
        "suites": suites,
        "models": [m.name for m in models],
        "baseline": baseline.name if baseline else None,
        "rows": summary_rows,
        "composite_failure": composites,
    }
    write_json(out_dir / "summary.json", summary)

    # Reporting (plots + markdown). Imported lazily to keep core import light.
    from pte.report import build_report

    build_report(out_dir, summary, metrics_all)

    logger.info(f"Done. Composite failure scores: {composites}")
    logger.info(f"Results: {out_dir}")
    return {"out_dir": str(out_dir), "summary": summary, "metrics": metrics_all}


def _composite_scores(rows: list[dict]) -> dict[str, float]:
    by_model: dict[str, list[float]] = {}
    for r in rows:
        if r.get("failure_score") is not None:
            by_model.setdefault(r["model"], []).append(r["failure_score"])
    return {m: float(np.mean(v)) for m, v in by_model.items()}


def _config_snapshot(cfg: EvalConfig) -> dict:
    d = asdict(cfg)
    return d
