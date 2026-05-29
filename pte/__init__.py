"""Post-Training Failure Evals (pte).

A model-agnostic harness that probes post-trained LLMs (base / SFT / DPO / instruct)
for failure modes that surface-level metrics miss:

    - sycophancy          (agreeing with false premises, caving under pressure)
    - verbosity bias      (length-driven judge preference)
    - reward hacking       (judge win-rate up, factuality down)
    - false confidence     (miscalibration between confidence and correctness)
    - safety boundary drift (unsafe compliance after optimization)
    - distribution shift    (quality drop on out-of-domain prompts)

The harness runs entirely on the Tinker API for inference (and optionally training).
"""

from pte.config import EvalConfig, ModelSpec, load_config
from pte.models import ModelClient

__all__ = ["EvalConfig", "ModelSpec", "load_config", "ModelClient"]
__version__ = "0.1.0"
