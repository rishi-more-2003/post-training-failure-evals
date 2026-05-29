"""Offline evaluator tests using fake model/judge clients (no Tinker calls)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from pte.config import EvalConfig, ModelSpec
from pte.evaluators import EvalContext, all_evaluator_names, get_evaluator
from pte.evaluators.base import SuiteResult
from pte.judge import PairwiseResult
from pte.models import Generation
from pte.runner import normalized_failure


@dataclass
class _Spec:
    stage: str = "dpo"


class FakeModel:
    def __init__(self, name="dpo", stage="dpo", verbose=False, truthful=True):
        self.name = name
        self.spec = _Spec(stage=stage)
        self.verbose = verbose
        self.truthful = truthful

    def generate_batch(self, conversations, max_tokens=None, temperature=None):
        out = []
        for conv in conversations:
            text = "A detailed answer " * (12 if self.verbose else 2)
            out.append(
                Generation(
                    text=text.strip(),
                    n_tokens=len(text.split()),
                    mean_logprob=-0.5,
                    sum_logprob=-5.0,
                    stop_reason="stop",
                )
            )
        return out

    def score_continuations(self, messages, continuations):
        # Favor the first continuation (the reference) when truthful.
        base = [0.0] * len(continuations)
        if continuations:
            base[0] = 2.0 if self.truthful else -2.0
        return base


class FakeJudge:
    def __init__(self, factual=True):
        self.factual = factual

    def classify_factuality(self, q, r, ref):
        return self.factual

    def classify_agreement(self, premise, response):
        return "reject"

    def classify_refusal(self, prompt, response):
        return "refuse"

    def pairwise(self, instruction, a, b, criteria=None):
        # Prefer the longer response (A is passed as the longer one in verbosity).
        return PairwiseResult(winner="A" if len(a) >= len(b) else "B", raw=["1"])

    def score(self, instruction, response, rubric=None):
        return 7

    def _ask(self, prompt, max_tokens=256):
        return "no"

    # Batched variants used by the evaluators (delegate to the single methods).
    def _ask_batch(self, prompts, max_tokens=64):
        return [self._ask(p, max_tokens) for p in prompts]

    def factuality_batch(self, items):
        return [self.classify_factuality(*it) for it in items]

    def agreement_batch(self, items):
        return [self.classify_agreement(*it) for it in items]

    def refusal_batch(self, items):
        return [self.classify_refusal(*it) for it in items]

    def score_batch(self, items):
        return [self.score(*it) for it in items]

    def pairwise_batch(self, items, criteria=None):
        return [self.pairwise(*it, criteria=criteria) for it in items]


def _ctx(model, baseline=None, judge=None, limit=4):
    cfg = EvalConfig(models=[], limit=limit)
    return EvalContext(
        model=model,
        judge=judge or FakeJudge(),
        cfg=cfg,
        baseline=baseline,
        rng=np.random.default_rng(0),
    )


@pytest.mark.parametrize("suite", ["sycophancy", "calibration", "verbosity", "safety", "dist_shift"])
def test_evaluator_runs_and_reports_headline(suite):
    ev = get_evaluator(suite)
    ctx = _ctx(FakeModel())
    res = ev.run(ctx)
    assert isinstance(res, SuiteResult)
    assert res.suite == suite
    assert res.headline_metric in res.metrics
    assert isinstance(res.headline_value, float)


def test_reward_hacking_requires_baseline_and_detects_gap():
    ev = get_evaluator("reward_hacking")
    # Model wins judge (longer) but is NOT truthful while base IS -> reward hacking.
    model = FakeModel(name="dpo", verbose=True, truthful=False)
    base = FakeModel(name="base", stage="base", verbose=False, truthful=True)
    ctx = _ctx(model, baseline=base, judge=FakeJudge(factual=True))
    res = ev.run(ctx)
    assert res.metrics["judge_win_rate_vs_base"] >= 0.5  # judge prefers verbose model
    assert res.headline_value == pytest.approx(res.metrics["reward_hacking_score"])


def test_reward_hacking_skips_on_baseline():
    ev = get_evaluator("reward_hacking")
    base = FakeModel(name="base", stage="base")
    ctx = _ctx(base, baseline=base)
    res = ev.run(ctx)
    assert res.metrics.get("skipped")


def test_normalized_failure_in_unit_range():
    cases = {
        "sycophancy": {"sycophancy_rate": 0.3},
        "calibration": {"ece": 0.2},
        "verbosity": {"padding_rate": 0.4},
        "reward_hacking": {"reward_hacking_score": 0.6},
        "safety": {"harmful_compliance_rate": 0.1},
        "dist_shift": {"quality_drop": 0.25},
    }
    for suite, m in cases.items():
        v = normalized_failure(suite, m)
        assert v is not None and 0.0 <= v <= 1.0


def test_all_six_suites_registered():
    assert set(all_evaluator_names()) == {
        "sycophancy",
        "verbosity",
        "reward_hacking",
        "calibration",
        "safety",
        "dist_shift",
    }
