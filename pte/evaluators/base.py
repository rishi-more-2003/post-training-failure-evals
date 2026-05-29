"""Evaluator framework: shared context, result types, and registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

import numpy as np

from pte.config import EvalConfig
from pte.judge import Judge
from pte.models import ModelClient


@dataclass
class EvalContext:
    """Everything an evaluator needs to run against one model variant."""

    model: ModelClient
    judge: Judge
    cfg: EvalConfig
    baseline: ModelClient | None = None  # the "base" variant, for comparative metrics
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))


@dataclass
class SuiteResult:
    """Result of running one evaluator (suite) on one model variant."""

    suite: str
    model: str
    stage: str
    # headline_metric: the single number to rank variants (higher = more failure,
    # unless noted). Each evaluator documents its direction in `notes`.
    headline_metric: str
    headline_value: float
    metrics: dict[str, Any]
    records: list[dict[str, Any]] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Evaluator(ABC):
    """Base class for all failure-mode evaluators."""

    name: str = "base"
    #: True if this evaluator needs the baseline variant (comparative metrics).
    requires_baseline: bool = False
    #: Human-readable failure direction, e.g. "higher = more sycophantic".
    direction: str = ""

    @abstractmethod
    def run(self, ctx: EvalContext) -> SuiteResult:
        ...


_REGISTRY: dict[str, Callable[[], Evaluator]] = {}


def register(name: str):
    def deco(cls):
        cls.name = name
        _REGISTRY[name] = cls
        return cls

    return deco


def get_evaluator(name: str) -> Evaluator:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown evaluator '{name}'. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[name]()


def all_evaluator_names() -> list[str]:
    return sorted(_REGISTRY)
