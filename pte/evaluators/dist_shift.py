"""Distribution shift: quality drop on out-of-domain prompts.

The model answers an in-domain set (everyday general-knowledge prompts) and an
out-of-domain set (formal proofs, niche programming, strict output formats,
graduate-level technical prompts). A judge scores each answer 1-10. We report the
absolute quality drop from in-domain to OOD, normalized to [0, 1].

A model that overfit to its post-training distribution shows a large drop even if
in-domain numbers look strong. Headline `quality_drop` (higher = worse).
"""

from __future__ import annotations

import numpy as np

from pte.data import load_ood
from pte.evaluators.base import EvalContext, Evaluator, SuiteResult, register


@register("dist_shift")
class DistShiftEvaluator(Evaluator):
    direction = "higher quality_drop = worse out-of-domain generalization"

    def run(self, ctx: EvalContext) -> SuiteResult:
        items = load_ood(limit=ctx.cfg.limit, use_hf=ctx.cfg.use_hf_datasets)
        convs = [[{"role": "user", "content": r["prompt"]}] for r in items]
        gens = ctx.model.generate_batch(convs)

        records: list[dict] = []
        in_scores: list[int] = []
        ood_scores: list[int] = []
        for r, g in zip(items, gens):
            score = ctx.judge.score(r["prompt"], g.text)
            (in_scores if r["split"] == "in" else ood_scores).append(score)
            records.append(
                {
                    "id": r["id"],
                    "domain": r["domain"],
                    "split": r["split"],
                    "prompt": r["prompt"],
                    "response": g.text,
                    "judge_score": score,
                }
            )

        in_mean = float(np.mean(in_scores)) if in_scores else 0.0
        ood_mean = float(np.mean(ood_scores)) if ood_scores else 0.0
        quality_drop = max(0.0, (in_mean - ood_mean) / 10.0)

        metrics = {
            "quality_drop": quality_drop,
            "in_domain_score": in_mean,
            "ood_score": ood_mean,
            "in_domain_score_norm": in_mean / 10.0,
            "ood_score_norm": ood_mean / 10.0,
            "n_in": len(in_scores),
            "n_ood": len(ood_scores),
        }
        return SuiteResult(
            suite=self.name,
            model=ctx.model.name,
            stage=ctx.model.spec.stage,
            headline_metric="quality_drop",
            headline_value=quality_drop,
            metrics=metrics,
            records=records,
            notes=self.direction,
        )
