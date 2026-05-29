"""Configuration: model variant registry + eval-suite settings (YAML-backed)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelSpec:
    """One model variant under test.

    Exactly one of `base_model` / `model_path` should identify the weights:
      - base_model: a Tinker-supported model name (e.g. "Qwen/Qwen3-8B-Base").
      - model_path: a trained checkpoint URL ("tinker://...") from SFT/DPO.
        When using model_path you must also supply base_model so the renderer/
        tokenizer can be resolved.
    """

    name: str  # human-readable label, e.g. "dpo"
    base_model: str  # Tinker base model name (always required for tokenizer/renderer)
    model_path: str | None = None  # tinker:// checkpoint (None => use base weights)
    renderer: str | None = None  # override; default = recommended for base_model
    stage: str = "base"  # one of: base, sft, dpo, rlhf, instruct (free-form tag)
    temperature: float = 0.0
    max_tokens: int = 512

    @property
    def is_checkpoint(self) -> bool:
        return self.model_path is not None


@dataclass
class JudgeSpec:
    base_model: str = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    renderer: str | None = None
    temperature: float = 0.0
    max_tokens: int = 512
    # Mitigate position bias in pairwise comparisons by averaging both orders.
    swap_positions: bool = True


@dataclass
class EvalConfig:
    models: list[ModelSpec]
    judge: JudgeSpec = field(default_factory=JudgeSpec)
    # Which evaluators to run; empty => all registered evaluators.
    suites: list[str] = field(default_factory=list)
    # Per-suite example caps (None => use dataset default / full builtin set).
    limit: int | None = None
    seed: int = 0
    output_dir: str = "results"
    max_concurrency: int = 16  # in-flight Tinker sample futures
    use_hf_datasets: bool = False  # if True, pull from HF; else use curated builtin sets

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "EvalConfig":
        models = [ModelSpec(**m) for m in d.get("models", [])]
        judge = JudgeSpec(**d.get("judge", {})) if d.get("judge") else JudgeSpec()
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        rest = {k: v for k, v in d.items() if k in known and k not in ("models", "judge")}
        return cls(models=models, judge=judge, **rest)


def load_config(path: str | Path) -> EvalConfig:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return EvalConfig.from_dict(data)
