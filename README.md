# Post-Training Failure Evals & Reward-Hacking Detector

> Detect when a post-trained model (SFT / DPO / RLHF) looks good on surface-level
> metrics but **fails under adversarial, distribution-shifted, or preference-conflicting
> prompts**.

Preference optimization (DPO/RLHF) maximizes a *proxy* — a learned reward or an
LLM judge — not ground truth. The classic, dangerous outcome is that the proxy
improves while the *true* objective (truthfulness, calibration, safety) silently
degrades. This repo is a model-agnostic harness that probes for that gap across
six concrete failure modes and produces an auditable report.

Built on the [Tinker API](https://tinker-docs.thinkingmachines.ai/) for both
inference and (optional) LoRA training of the variants under test.

```
Stack: Python · PyTorch · Hugging Face · Tinker · DPO/RLHF · LLM-as-judge · calibration
```

---

## What it detects

| Suite | Failure mode | Headline metric | How it's measured |
|---|---|---|---|
| `sycophancy` | Agrees with the user instead of the truth | `sycophancy_rate` | False-premise endorsement + **caving under pushback** (flips a correct answer when the user pushes a wrong one) |
| `verbosity` | Length inflation without substance | `mean_tokens` | Default length, judge-rated **padding rate**, and the reward-proxy's **length preference** |
| `reward_hacking` | Wins the judge but loses factuality | `reward_hacking_score` | Judge win-rate vs. base **minus** factual-accuracy improvement |
| `calibration` | Confidently wrong (false confidence) | `ece` | Log-prob MC confidence + verbalized confidence vs. correctness (ECE, Brier, AUROC, overconfidence gap) |
| `safety` | Unsafe compliance drift | `harmful_compliance_rate` | Compliance on harmful prompts + **over-refusal** on benign controls |
| `dist_shift` | Overfit to the post-training distribution | `quality_drop` | Judged quality drop from in-domain to out-of-domain prompts |

Every number is backed by per-example records written to `raw/*.jsonl` — the receipts.

---

## Quickstart

```bash
# 1. Install
python -m venv .venv && source .venv/bin/activate   # (Windows: .venv\Scripts\activate)
pip install -e .

# 2. Add your Tinker key (a .env at the repo root)
echo 'TINKER_API_KEY="tml-..."' > .env

# 3. Smoke test the whole pipeline cheaply (a few examples/suite, small models)
pte-eval run --config configs/smoke.yaml

# 4. Full run: base vs instruct (+ your SFT/DPO checkpoints once trained)
pte-eval run --config configs/eval.yaml

# Run a subset of suites / cap examples:
pte-eval run --config configs/eval.yaml --suites sycophancy calibration --limit 20
pte-eval list-suites
```

Outputs land in `results/<run_id>/`:

```
report.md            # human-readable summary (tables + embedded plots)
summary.json         # headline + normalized failure matrix + composite scores
metrics.json         # full metric dict for every (model, suite)
raw/<model>__<suite>.jsonl   # per-example records (auditable)
plots/*.png          # composite bar, failure heatmap, reliability diagrams, ...
```

Optional interactive dashboard:

```bash
pip install gradio
python scripts/dashboard.py --run results/<run_id>
```

---

## Example output

A tiny `smoke.yaml` run (Qwen3-8B-Base vs Qwen3-4B-Instruct, judged by Qwen3-4B)
produces a composite ranking and a failure heatmap like this:

![failure heatmap](docs/example_run/plots/failure_heatmap.png)

![composite failure](docs/example_run/plots/composite_failure.png)

In this illustrative run the instruct variant shows the textbook post-training
trade: it **wins the judge but not on factuality** (`reward_hacking`), is more
**overconfident** (higher `ece`), more **sycophantic**, and **1.3× longer** than the
base — even though both look fine on safety. Calibration reliability diagrams
(log-prob confidence vs. accuracy) per variant:

![reliability](docs/example_run/plots/reliability.png)

See [`docs/example_run/report.md`](docs/example_run/report.md) for the full report.
*(Numbers are from a tiny smoke run for illustration — use `configs/eval.yaml` and
the full builtin sets, or `use_hf_datasets: true`, for real conclusions.)*

---

## Comparing variants: base → SFT → DPO

The most resume-relevant story is attributing a failure to a specific
optimization step. Train the variants on Tinker, then point the harness at them:

```bash
# Stage 1: SFT a base model into a chat model
python training/train_sft.py --model-name Qwen/Qwen3-8B-Base --dataset no_robots

# Stage 2: DPO from the SFT checkpoint (lower beta = more aggressive optimization)
python training/train_dpo.py --model-name Qwen/Qwen3-8B-Base \
    --load-checkpoint-path tinker://YOUR_SFT_STATE --dataset hhh --dpo-beta 0.1
```

Each script prints a `tinker://` checkpoint path. Paste it into `configs/eval.yaml`
(see the commented `sft` / `dpo` blocks) and re-run. Details in
[`training/README.md`](training/README.md).

---

## How it works

```
                 ┌──────────────┐     ┌──────────────┐
   ModelSpec ──▶ │ ModelClient  │     │    Judge     │ (separate, stronger model)
 (base/sft/dpo)  │ Tinker sample│     │ pairwise /   │
                 │ + logprobs   │     │ score /      │
                 │ + chat render│     │ classify     │
                 └──────┬───────┘     └──────┬───────┘
                        │                    │
                        ▼                    ▼
        ┌───────────────────────────────────────────────┐
        │   Evaluators (6 failure-mode suites)           │
        │   sycophancy · verbosity · reward_hacking ·    │
        │   calibration · safety · dist_shift            │
        └───────────────────────┬───────────────────────┘
                                 ▼
              Runner → metrics.json / summary.json
                     → report.md + plots/  (+ Gradio dashboard)
```

Key design choices:

- **Model-agnostic.** A `ModelClient` wraps a Tinker `SamplingClient` and uses the
  cookbook renderers, so the *same* chat prompt works on a raw base model
  (`role_colon`) and an instruct model (`qwen3_instruct` / `llama3`).
- **Confidence from log-probs.** Calibration uses `compute_logprobs` to score the
  correct vs. a plausible-false answer as continuations — a clean, judge-free
  confidence signal — alongside verbalized confidence.
- **Judge bias is a measured signal, not an assumption.** Pairwise judging uses
  position-swap debiasing, and we *measure* the judge's verbosity bias
  (`length_pref_bias`) rather than trusting it blindly.
- **Comparative metrics.** Reward hacking is defined relative to the `base` variant:
  `judge_pref − factuality_delta`. Verbosity reports length growth vs. base.
- **Cheap + reproducible by default.** Curated builtin datasets ship in the repo so
  runs are offline-deterministic; flip `use_hf_datasets: true` to scale to
  TruthfulQA / Anthropic HH-RLHF.
- **Cached sampling.** Generations are memoized per client so suites that share
  prompts (calibration, reward-hacking, sycophancy) don't re-sample.

---

## Datasets

Curated, citation-style builtin sets live in `pte/data/builtin/` (with optional
Hugging Face backends):

- `factual_qa` — TruthfulQA-style misconceptions with a reference and a
  plausible-but-false answer (→ calibration, reward hacking, sycophancy pressure).
- `false_premises` — false premises embedded in questions (→ sycophancy).
- `safety` — harmful requests **and** benign over-refusal controls (→ safety drift).
- `ood` — paired in-domain vs. out-of-domain prompts (→ distribution shift).
- `instructions` — neutral helpfulness prompts (→ verbosity).

Set `use_hf_datasets: true` to pull `truthful_qa` and `Anthropic/hh-rlhf` instead.

---

## Repo layout

```
pte/
  models.py        # Tinker SamplingClient wrapper: chat gen, batching, logprob scoring
  judge.py         # LLM-as-judge: pairwise (debiased), scoring, classifiers
  metrics.py       # ECE, reliability curve, AUROC, Brier, Wilson + bootstrap CIs
  evaluators/      # one file per failure mode + base framework/registry
  data/            # loaders + curated builtin JSONL datasets
  runner.py        # orchestration, normalization, composite scoring
  report.py        # matplotlib plots + markdown report
  cli.py           # `pte-eval` entrypoint
training/          # SFT + DPO scripts (Tinker Cookbook) to produce variants
configs/           # eval.yaml (full) + smoke.yaml (cheap)
scripts/           # optional Gradio dashboard
tests/             # offline tests (metrics, data schema, evaluator logic)
```

## Testing

```bash
pip install pytest
pytest -q          # fully offline: no Tinker calls, no network
```

## Notes & limitations

- LLM judges are imperfect; treat absolute numbers as directional and rely on
  base→variant *deltas*. Use a stronger judge (e.g. `Qwen3-30B`/`Llama-3.3-70B`)
  for headline runs.
- Safety prompts are standard red-team *categories* phrased as requests; no
  harmful content is included in the repo.
- Tinker training/inference incurs usage — start with `smoke.yaml` / `--limit`.

## License

MIT
