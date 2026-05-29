"""Supervised fine-tuning (SFT) on Tinker, producing a chat-capable checkpoint.

This is stage 1 of the base -> SFT -> DPO pipeline whose variants the eval harness
compares. It wraps the Tinker Cookbook supervised trainer with project-friendly
defaults and prints the resulting `tinker://` sampler path to paste into your
eval config (`configs/eval.yaml`).

Example:
    python training/train_sft.py \
        --model-name Qwen/Qwen3-8B-Base \
        --dataset no_robots \
        --lora-rank 32 --lr 1e-4 --epochs 1

Cost/time note: this runs real LoRA training on remote GPUs and will incur Tinker
usage. Start small (Llama-3.2-1B or a few hundred steps via --max-steps) to iterate.
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime

from dotenv import load_dotenv


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="SFT a base model on Tinker (LoRA).")
    ap.add_argument("--model-name", default="Qwen/Qwen3-8B-Base")
    ap.add_argument(
        "--dataset",
        default="no_robots",
        help="'no_robots', 'tulu3', or a path to a chat .jsonl file.",
    )
    ap.add_argument("--renderer-name", default=None, help="Override; default = recommended.")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-rank", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--max-length", type=int, default=16384)
    ap.add_argument("--max-steps", type=int, default=None, help="Cap steps for a quick run.")
    ap.add_argument("--log-path", default=None)
    ap.add_argument("--save-every", type=int, default=50)
    return ap.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()

    from tinker_cookbook import checkpoint_utils, cli_utils, renderers  # noqa: F401
    from tinker_cookbook.recipes.chat_sl.train import get_dataset_builder
    from tinker_cookbook.supervised import train

    renderer_name = args.renderer_name or checkpoint_utils.resolve_renderer_name_from_checkpoint_or_default(
        model_name=args.model_name, explicit_renderer_name=args.renderer_name, load_checkpoint_path=None
    )
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    safe = args.model_name.replace("/", "-")
    log_path = args.log_path or f"results/training/sft-{safe}-{stamp}"
    cli_utils.check_log_dir(log_path, behavior_if_exists="overwrite")

    config = train.Config(
        log_path=log_path,
        model_name=args.model_name,
        renderer_name=renderer_name,
        dataset_builder=get_dataset_builder(
            args.dataset, args.model_name, renderer_name, args.max_length, args.batch_size
        ),
        evaluator_builders=[],
        infrequent_evaluator_builders=[],
        learning_rate=args.lr,
        lr_schedule="linear",
        num_epochs=args.epochs,
        lora_rank=args.lora_rank,
        save_every=args.save_every,
        eval_every=args.save_every,
        max_steps=args.max_steps,
    )
    print(f"[SFT] model={args.model_name} renderer={renderer_name} -> {log_path}")
    asyncio.run(train.main(config))

    rec = checkpoint_utils.get_last_checkpoint(log_path, required_key="sampler_path")
    if rec is not None:
        print("\n================ SFT checkpoint ================")
        print(f"model_path (paste into configs/eval.yaml): {rec.sampler_path}")
        print(f"renderer_name: {renderer_name}")
        print(f"state_path (for DPO --load-checkpoint-path): {rec.state_path}")
        print("===============================================")
    else:
        print(f"No checkpoint with sampler_path found in {log_path}.")


if __name__ == "__main__":
    main()
