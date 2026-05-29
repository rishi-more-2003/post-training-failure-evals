"""Model abstraction over the Tinker SamplingClient.

Provides a uniform interface across base / SFT / DPO / instruct variants:
  - chat generation (single + batched via Tinker futures)
  - sequence log-probability of a model's own generation (for confidence)
  - scoring a fixed continuation under the model (for MC calibration)

Renderers handle the bidirectional token<->chat conversion so the same chat
prompt works on a base model (role_colon) and an instruct model (qwen3/llama3).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import tinker
from tinker import types
from tinker_cookbook import model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

from pte.config import ModelSpec
from pte.utils import setup_logging

logger = setup_logging()

Message = dict[str, str]  # {"role": ..., "content": ...}


@dataclass
class Generation:
    """Result of a single chat generation."""

    text: str
    n_tokens: int
    # Mean per-token log-prob of the generated sequence (a proxy for model confidence).
    mean_logprob: float
    sum_logprob: float
    stop_reason: str

    @property
    def perplexity(self) -> float:
        return math.exp(-self.mean_logprob) if self.mean_logprob else float("inf")


class ModelClient:
    """Uniform wrapper around a Tinker sampling client for one model variant."""

    def __init__(self, spec: ModelSpec, service_client: tinker.ServiceClient | None = None):
        self.spec = spec
        self.name = spec.name
        sc = service_client or tinker.ServiceClient()
        if spec.is_checkpoint:
            self.sampling = sc.create_sampling_client(
                base_model=spec.base_model, model_path=spec.model_path
            )
        else:
            self.sampling = sc.create_sampling_client(base_model=spec.base_model)
        self.tokenizer = get_tokenizer(spec.base_model)
        renderer_name = spec.renderer or model_info.get_recommended_renderer_name(spec.base_model)
        self.renderer = renderers.get_renderer(renderer_name, self.tokenizer)
        self.renderer_name = renderer_name
        self._stops = self.renderer.get_stop_sequences()
        self._cache: dict[tuple, Generation] = {}

    # ----- generation -------------------------------------------------------
    def _params(self, max_tokens: int | None, temperature: float | None) -> types.SamplingParams:
        return types.SamplingParams(
            max_tokens=max_tokens if max_tokens is not None else self.spec.max_tokens,
            temperature=temperature if temperature is not None else self.spec.temperature,
            stop=self._stops,
        )

    def _decode_generation(self, seq) -> Generation:
        msg, stop_reason = self.renderer.parse_response(seq.tokens)
        text = (msg or {}).get("content", "") if isinstance(msg, dict) else str(msg)
        logprobs = [lp for lp in (seq.logprobs or []) if lp is not None]
        n = len(seq.tokens)
        sum_lp = float(sum(logprobs))
        mean_lp = sum_lp / len(logprobs) if logprobs else 0.0
        return Generation(
            text=(text or "").strip(),
            n_tokens=n,
            mean_logprob=mean_lp,
            sum_logprob=sum_lp,
            stop_reason=str(stop_reason),
        )

    def generate(
        self,
        messages: Sequence[Message],
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> Generation:
        return self.generate_batch([messages], max_tokens, temperature)[0]

    @staticmethod
    def _key(conv: Sequence[Message], max_tokens, temperature) -> tuple:
        return (
            tuple((m["role"], m["content"]) for m in conv),
            max_tokens,
            temperature,
        )

    def generate_batch(
        self,
        conversations: Sequence[Sequence[Message]],
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> list[Generation]:
        """Generate for many conversations, pipelining requests via Tinker futures.

        Results are memoized per client so evaluators that reuse the same prompt
        (e.g. factual QA shared by calibration + reward-hacking) avoid re-sampling.
        Deterministic (temperature 0) requests are the common case and cache cleanly.
        """
        params = self._params(max_tokens, temperature)
        mt = params.max_tokens
        temp = params.temperature
        results: list[Generation | None] = [None] * len(conversations)
        futures = []
        pending_idx = []
        for i, conv in enumerate(conversations):
            key = self._key(conv, mt, temp)
            if key in self._cache:
                results[i] = self._cache[key]
                continue
            mi = self.renderer.build_generation_prompt(list(conv))
            futures.append(self.sampling.sample(prompt=mi, num_samples=1, sampling_params=params))
            pending_idx.append(i)
        for i, fut in zip(pending_idx, futures):
            res = fut.result()
            gen = self._decode_generation(res.sequences[0])
            self._cache[self._key(conversations[i], mt, temp)] = gen
            results[i] = gen
        return results  # type: ignore[return-value]

    # ----- scoring ----------------------------------------------------------
    def score_continuation(self, messages: Sequence[Message], continuation: str) -> float:
        """Mean per-token log-prob the model assigns to `continuation` after `messages`.

        Used for multiple-choice calibration: score each option and compare.
        """
        return self.score_continuations(messages, [continuation])[0]

    def score_continuations(
        self, messages: Sequence[Message], continuations: Sequence[str]
    ) -> list[float]:
        prompt_mi = self.renderer.build_generation_prompt(list(messages))
        prompt_len = prompt_mi.length
        futures = []
        cont_token_counts = []
        for cont in continuations:
            cont_tokens = self.tokenizer.encode(cont, add_special_tokens=False)
            cont_token_counts.append(len(cont_tokens))
            full = types.ModelInput.from_ints(list(prompt_mi.to_ints()) + list(cont_tokens))
            futures.append(self.sampling.compute_logprobs(prompt=full))
        scores: list[float] = []
        for fut, n_cont in zip(futures, cont_token_counts):
            logprobs = fut.result()
            cont_lps = [lp for lp in logprobs[prompt_len:] if lp is not None]
            scores.append(sum(cont_lps) / max(1, len(cont_lps)) if cont_lps else float("-inf"))
        return scores
