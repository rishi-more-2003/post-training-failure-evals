"""Direct Preference Optimization (DPO) on Tinker.

Stage 2 of the base -> SFT -> DPO pipeline. Ideally start from your SFT
state checkpoint (`--load-checkpoint-path tinker://...state...`) so the DPO
variant is comparable to the SFT one; you can also DPO directly from a base model.

It wraps the Tinker Cookbook DPO trainer and prints the resulting `tinker://`
sampler path for `configs/eval.yaml`.

Example (after SFT):
    python training/train_dpo.py \
        --model-name Qwen/Qwen3-8B-Base \
        --load-checkpoint-path tinker://YOUR_SFT_STATE \
        --dataset hhh --dpo-beta 0.1 --lr 1e-5

DPO intentionally over-optimizes a preference signal -- exactly the regime where
this repo's harness should reveal sycophancy / verbosity / reward-hacking drift.
"""

from __future__ import annotations

import argparse
from datetime import datetime

from dotenv import load_dotenv


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="DPO-train a model on Tinker (LoRA).")
    ap.add_argument("--model-name", default="Qwen/Qwen3-8B-Base")
    ap.add_argument(
        "--dataset",
        default="hhh",
        choices=["hhh", "helpsteer3", "ultrafeedback"],
        help="Preference dataset (chosen/rejected pairs).",
    )
    ap.add_argument(
        "--load-checkpoint-path",
        default=None,
        help="SFT state checkpoint to start from (recommended).",
    )
    ap.add_argument("--renderer-name", default=None)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--dpo-beta", type=float, default=0.1)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--max-length", type=int, default=8192)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--reference-model-name", default=None)
    ap.add_argument("--log-path", default=None)
    return ap.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()

    from tinker_cookbook import checkpoint_utils, cli_utils
    from tinker_cookbook.preference import train_dpo
    from tinker_cookbook.recipes.preference.dpo.train import get_dataset_builder

    renderer_name = args.renderer_name or checkpoint_utils.resolve_renderer_name_from_checkpoint_or_default(
        model_name=args.model_name,
        explicit_renderer_name=args.renderer_name,
        load_checkpoint_path=args.load_checkpoint_path,
    )
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    safe = args.model_name.replace("/", "-")
    log_path = args.log_path or f"results/training/dpo-{safe}-{stamp}"
    cli_utils.check_log_dir(log_path, behavior_if_exists="overwrite")

    config = train_dpo.Config(
        log_path=log_path,
        model_name=args.model_name,
        renderer_name=renderer_name,
        dataset_builder=get_dataset_builder(
            args.dataset, args.model_name, renderer_name, args.max_length, args.batch_size
        ),
        load_checkpoint_path=args.load_checkpoint_path,
        evaluator_builders=[],
        learning_rate=args.lr,
        lr_schedule="linear",
        dpo_beta=args.dpo_beta,
        reference_model_name=args.reference_model_name,
        max_steps=args.max_steps,
    )
    print(f"[DPO] model={args.model_name} beta={args.dpo_beta} renderer={renderer_name} -> {log_path}")
    train_dpo.main(config)

    rec = checkpoint_utils.get_last_checkpoint(log_path, required_key="sampler_path")
    if rec is not None:
        print("\n================ DPO checkpoint ================")
        print(f"model_path (paste into configs/eval.yaml): {rec.sampler_path}")
        print(f"renderer_name: {renderer_name}")
        print("===============================================")
    else:
        print(f"No checkpoint with sampler_path found in {log_path}.")


if __name__ == "__main__":
    main()
