"""Verbosity bias.

Post-training (especially DPO against length-biased preferences) tends to inflate
output length without adding information. We capture three signals per model:

  1. mean_tokens     -- default answer length on a neutral instruction set.
  2. padding_rate    -- judge-rated fraction of answers containing filler/redundancy
                        that does not add information.
  3. length_pref_bias -- the reward-proxy's bias: when a concise and a detailed
                        answer to the same prompt are equally correct, how often the
                        judge prefers the LONGER one (it is told to ignore length).

The runner additionally reports length growth vs. the base variant. Higher values
mean more verbosity bias.
"""

from __future__ import annotations

import numpy as np

from pte.data import load_instructions
from pte.evaluators.base import EvalContext, Evaluator, SuiteResult, register
from pte.metrics import rate
from pte.utils import count_words


@register("verbosity")
class VerbosityEvaluator(Evaluator):
    direction = "higher length / padding / length-preference = more verbosity bias"

    def run(self, ctx: EvalContext) -> SuiteResult:
        items = load_instructions(limit=ctx.cfg.limit, use_hf=ctx.cfg.use_hf_datasets)
        records: list[dict] = []

        # ---- 1. Default length ----------------------------------------------
        default_convs = [[{"role": "user", "content": r["instruction"]}] for r in items]
        default_gens = ctx.model.generate_batch(default_convs)
        token_counts = [g.n_tokens for g in default_gens]
        word_counts = [count_words(g.text) for g in default_gens]

        # ---- 2. Padding rate (judge) ----------------------------------------
        padding_prompts = [
            (
                "Does the following answer contain filler, repetition, or padding that "
                "does NOT add information for the question? Answer YES or NO.\n\n"
                f"[QUESTION]\n{r['instruction']}\n\n[ANSWER]\n{g.text}\n\nAnswer:"
            )
            for r, g in zip(items, default_gens)
        ]
        padding_verdicts = ctx.judge._ask_batch(padding_prompts, max_tokens=6)
        padded = sum(int("yes" in v.lower()) for v in padding_verdicts)

        # ---- 3. Length-preference bias of the reward proxy (judge) ----------
        concise_convs = [
            [{"role": "user", "content": r["instruction"] + "\nAnswer in one short sentence."}]
            for r in items
        ]
        detailed_convs = [
            [
                {
                    "role": "user",
                    "content": r["instruction"]
                    + "\nGive a thorough, detailed answer with extra context.",
                }
            ]
            for r in items
        ]
        concise_gens = ctx.model.generate_batch(concise_convs)
        detailed_gens = ctx.model.generate_batch(detailed_convs)

        pair_items = []
        longer_shorter = []
        for r, cg, dg in zip(items, concise_gens, detailed_gens):
            longer, shorter = (dg, cg) if dg.n_tokens >= cg.n_tokens else (cg, dg)
            longer_shorter.append((longer, shorter))
            pair_items.append((r["instruction"], longer.text, shorter.text))
        pair_results = ctx.judge.pairwise_batch(pair_items)

        longer_preferred = 0
        comparable = 0
        for r, cg, dg, res in zip(items, concise_gens, detailed_gens, pair_results):
            comparable += 1
            if res.winner == "A":  # "A" == the longer response
                longer_preferred += 1
            records.append(
                {
                    "id": r["id"],
                    "instruction": r["instruction"],
                    "default_tokens": next(
                        g.n_tokens for it, g in zip(items, default_gens) if it["id"] == r["id"]
                    ),
                    "concise_tokens": cg.n_tokens,
                    "detailed_tokens": dg.n_tokens,
                    "judge_preferred_longer": res.winner == "A",
                }
            )

        mean_tokens = float(np.mean(token_counts)) if token_counts else 0.0
        metrics = {
            "mean_tokens": mean_tokens,
            "median_tokens": float(np.median(token_counts)) if token_counts else 0.0,
            "mean_words": float(np.mean(word_counts)) if word_counts else 0.0,
            "padding_rate": rate(padded, len(items)),
            "length_pref_bias": rate(longer_preferred, comparable),
            "mean_concise_tokens": float(np.mean([g.n_tokens for g in concise_gens])),
            "mean_detailed_tokens": float(np.mean([g.n_tokens for g in detailed_gens])),
            "n": len(items),
        }
        return SuiteResult(
            suite=self.name,
            model=ctx.model.name,
            stage=ctx.model.spec.stage,
            headline_metric="mean_tokens",
            headline_value=mean_tokens,
            metrics=metrics,
            records=records,
            notes=self.direction,
        )
