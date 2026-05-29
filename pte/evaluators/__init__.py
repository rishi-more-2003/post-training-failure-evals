"""Failure-mode evaluators. Importing this package registers all suites."""

from pte.evaluators.base import (
    EvalContext,
    Evaluator,
    SuiteResult,
    all_evaluator_names,
    get_evaluator,
    register,
)

# Import side effects register each evaluator in the global registry.
from pte.evaluators import (  # noqa: E402,F401
    calibration,
    dist_shift,
    reward_hacking,
    safety,
    sycophancy,
    verbosity,
)

__all__ = [
    "EvalContext",
    "Evaluator",
    "SuiteResult",
    "all_evaluator_names",
    "get_evaluator",
    "register",
]
