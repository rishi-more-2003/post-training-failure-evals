"""Optional Gradio dashboard for browsing a completed eval run.

Usage:
    pip install gradio
    python scripts/dashboard.py --run results/<run_id>

Shows the composite ranking, the failure heatmap, per-suite metrics, and lets you
drill into the raw per-example records (the receipts behind every number).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pte.utils import read_json, read_jsonl


def _latest_run(results_dir: Path) -> Path:
    runs = sorted(p for p in results_dir.iterdir() if p.is_dir())
    if not runs:
        raise SystemExit(f"No runs found in {results_dir}")
    return runs[-1]


def build_app(run_dir: Path):
    import gradio as gr

    summary = read_json(run_dir / "summary.json")
    metrics = read_json(run_dir / "metrics.json")
    rows = summary["rows"]
    suites = summary["suites"]
    models = summary["models"]

    comp = summary["composite_failure"]
    comp_df = pd.DataFrame(
        sorted(({"model": m, "composite_failure": v} for m, v in comp.items()),
               key=lambda d: d["composite_failure"])
    )
    matrix_df = pd.DataFrame(
        [
            {"model": m, **{s: _failure(rows, m, s) for s in suites}}
            for m in models
        ]
    )

    def suite_table(suite: str) -> pd.DataFrame:
        recs = [r for r in rows if r["suite"] == suite]
        return pd.DataFrame(
            [{"model": r["model"], "stage": r["stage"], r["headline_metric"]: r["headline_value"],
              "failure_score": r.get("failure_score")} for r in recs]
        )

    def raw_table(model: str, suite: str) -> pd.DataFrame:
        path = run_dir / "raw" / f"{model}__{suite}.jsonl"
        if not path.exists():
            return pd.DataFrame([{"info": "no records for this cell"}])
        return pd.DataFrame(read_jsonl(path))

    with gr.Blocks(title="Post-Training Failure Evals") as app:
        gr.Markdown(f"# Post-Training Failure Evals\n**Run:** `{summary['run_id']}` "
                    f"| **Baseline:** `{summary['baseline']}`")
        with gr.Tab("Overview"):
            gr.Markdown("### Composite failure ranking (lower = safer)")
            gr.Dataframe(comp_df)
            gr.Markdown("### Normalized failure matrix")
            gr.Dataframe(matrix_df)
            for png in ["composite_failure", "failure_heatmap", "reliability",
                        "verbosity_length", "safety"]:
                p = run_dir / "plots" / f"{png}.png"
                if p.exists():
                    gr.Image(str(p), label=png, show_label=True)
        with gr.Tab("By suite"):
            suite_dd = gr.Dropdown(suites, value=suites[0], label="Suite")
            suite_out = gr.Dataframe()
            suite_dd.change(suite_table, suite_dd, suite_out)
            app.load(suite_table, suite_dd, suite_out)
        with gr.Tab("Raw records"):
            with gr.Row():
                m_dd = gr.Dropdown(models, value=models[0], label="Model")
                s_dd = gr.Dropdown(suites, value=suites[0], label="Suite")
            raw_out = gr.Dataframe()
            m_dd.change(raw_table, [m_dd, s_dd], raw_out)
            s_dd.change(raw_table, [m_dd, s_dd], raw_out)
            app.load(raw_table, [m_dd, s_dd], raw_out)
    return app


def _failure(rows, model, suite):
    for r in rows:
        if r["model"] == model and r["suite"] == suite:
            return r.get("failure_score")
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None, help="Path to a results/<run_id> dir (default: latest).")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()
    run_dir = Path(args.run) if args.run else _latest_run(Path(args.results_dir))
    print(f"Loading run: {run_dir}")
    app = build_app(run_dir)
    app.launch(share=args.share)


if __name__ == "__main__":
    main()
