"""Shared utilities: env loading, logging, JSON(L) IO, light parsing helpers."""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Iterable

from dotenv import load_dotenv
from rich.logging import RichHandler

_LOG_CONFIGURED = False


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure rich logging once and return the package logger."""
    global _LOG_CONFIGURED
    if not _LOG_CONFIGURED:
        logging.basicConfig(
            level=level,
            format="%(message)s",
            datefmt="[%X]",
            handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
        )
        _LOG_CONFIGURED = True
    return logging.getLogger("pte")


def load_env(dotenv_path: str | os.PathLike | None = None) -> None:
    """Load .env so TINKER_API_KEY is available to the Tinker client."""
    load_dotenv(dotenv_path=dotenv_path, override=False)
    if not os.environ.get("TINKER_API_KEY"):
        raise RuntimeError(
            "TINKER_API_KEY not found. Add it to a .env file at the repo root "
            'as TINKER_API_KEY="tml-..." or export it in your shell.'
        )


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def read_jsonl(path: str | os.PathLike) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def write_jsonl(path: str | os.PathLike, rows: Iterable[dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: str | os.PathLike, obj: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def read_json(path: str | os.PathLike) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


_YES_RE = re.compile(r"\b(yes|agree|correct|true|right|i agree|that'?s right)\b", re.I)
_NO_RE = re.compile(r"\b(no|disagree|incorrect|false|wrong|that'?s not|actually)\b", re.I)


def first_int(text: str, lo: int | None = None, hi: int | None = None) -> int | None:
    """Extract the first integer in `text`, optionally clamped to [lo, hi]."""
    m = re.search(r"-?\d+", text)
    if not m:
        return None
    val = int(m.group())
    if lo is not None:
        val = max(lo, val)
    if hi is not None:
        val = min(hi, val)
    return val


def first_float(text: str) -> float | None:
    m = re.search(r"-?\d+(?:\.\d+)?", text)
    return float(m.group()) if m else None


def extract_label(text: str, labels: list[str]) -> str | None:
    """Return the first label (case-insensitive, word-boundary) found in text."""
    low = text.lower()
    best: tuple[int, str] | None = None
    for lab in labels:
        idx = low.find(lab.lower())
        if idx != -1 and (best is None or idx < best[0]):
            best = (idx, lab)
    return best[1] if best else None


def count_words(text: str) -> int:
    return len(re.findall(r"\S+", text))
