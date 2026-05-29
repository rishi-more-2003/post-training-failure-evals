"""Dataset loaders for each failure-mode suite.

Two backends:
  - builtin  (default): curated JSONL sets shipped with the repo. Free, offline,
    deterministic -- ideal for CI and reproducible portfolio results.
  - hf       (use_hf_datasets=True): pull larger public datasets
    (TruthfulQA, Anthropic/hh-rlhf, ...) for scaled-up runs.

All loaders return a list[dict] with a documented schema (see each function).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pte.utils import read_jsonl, setup_logging

logger = setup_logging()

_BUILTIN = Path(__file__).resolve().parent / "builtin"


def _load_builtin(name: str, limit: int | None = None) -> list[dict[str, Any]]:
    rows = read_jsonl(_BUILTIN / f"{name}.jsonl")
    return rows[:limit] if limit else rows


def _balanced_limit(rows: list[dict[str, Any]], key: str, limit: int | None) -> list[dict[str, Any]]:
    """Take `limit` rows round-robin across the distinct values of `rows[key]`.

    Keeps small/capped runs representative of every class (e.g. both safety
    harmful + benign, both in-domain + OOD splits) instead of slicing the first
    class only. Preserves original order within each group.
    """
    if not limit or limit >= len(rows):
        return rows
    groups: dict[Any, list[dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault(r[key], []).append(r)
    out: list[dict[str, Any]] = []
    idx = 0
    order = list(groups)
    while len(out) < limit:
        progressed = False
        for g in order:
            if idx < len(groups[g]):
                out.append(groups[g][idx])
                progressed = True
                if len(out) >= limit:
                    break
        if not progressed:
            break
        idx += 1
    return out


# --------------------------------------------------------------------------
# factual_qa: confidence/calibration, reward hacking, sycophancy-under-pressure
#   schema: id, question, reference, false_answer, category
# --------------------------------------------------------------------------
def load_factual_qa(limit: int | None = None, use_hf: bool = False) -> list[dict[str, Any]]:
    if use_hf:
        try:
            from datasets import load_dataset

            ds = load_dataset("truthful_qa", "generation", split="validation")
            rows = []
            for i, r in enumerate(ds):
                incorrect = r.get("incorrect_answers") or [""]
                rows.append(
                    {
                        "id": f"tqa-{i}",
                        "question": r["question"],
                        "reference": r["best_answer"],
                        "false_answer": incorrect[0],
                        "category": r.get("category", "truthfulqa"),
                    }
                )
                if limit and len(rows) >= limit:
                    break
            return rows
        except Exception as e:  # pragma: no cover - network/optional path
            logger.warning(f"HF TruthfulQA load failed ({e}); falling back to builtin.")
    return _load_builtin("factual_qa", limit)


# --------------------------------------------------------------------------
# false_premises: sycophancy (agreeing with a false user-stated premise)
#   schema: id, premise (false), question (presupposes premise), correction, category
# --------------------------------------------------------------------------
def load_false_premises(limit: int | None = None, use_hf: bool = False) -> list[dict[str, Any]]:
    return _load_builtin("false_premises", limit)


# --------------------------------------------------------------------------
# safety: safety boundary drift (harmful requests + benign over-refusal controls)
#   schema: id, prompt, category, should_refuse (bool)
# --------------------------------------------------------------------------
def load_safety(limit: int | None = None, use_hf: bool = False) -> list[dict[str, Any]]:
    if use_hf:
        try:
            from datasets import load_dataset

            ds = load_dataset("Anthropic/hh-rlhf", data_dir="red-team-attempts", split="train")
            rows = []
            for i, r in enumerate(ds):
                txt = r.get("transcript", "")
                first = txt.split("\n\nAssistant:")[0].replace("\n\nHuman:", "").strip()
                if not first:
                    continue
                rows.append(
                    {
                        "id": f"hh-rt-{i}",
                        "prompt": first[:1000],
                        "category": "red-team",
                        "should_refuse": True,
                    }
                )
                if limit and len(rows) >= limit:
                    break
            return rows
        except Exception as e:  # pragma: no cover
            logger.warning(f"HF hh-rlhf red-team load failed ({e}); falling back to builtin.")
    return _balanced_limit(_load_builtin("safety"), "should_refuse", limit)


# --------------------------------------------------------------------------
# ood: distribution shift (in-domain vs out-of-domain instruction prompts)
#   schema: id, prompt, domain, split ("in" | "ood")
# --------------------------------------------------------------------------
def load_ood(limit: int | None = None, use_hf: bool = False) -> list[dict[str, Any]]:
    return _balanced_limit(_load_builtin("ood"), "split", limit)


# --------------------------------------------------------------------------
# instructions: general helpfulness prompts (verbosity bias, head-to-head)
#   schema: id, instruction, category
# --------------------------------------------------------------------------
def load_instructions(limit: int | None = None, use_hf: bool = False) -> list[dict[str, Any]]:
    return _load_builtin("instructions", limit)
