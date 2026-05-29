# Training the model variants

The eval harness compares variants along the post-training pipeline:

```
base  ->  SFT  ->  DPO/RLHF
```

You don't need to train anything to try the harness — `configs/eval.yaml` runs a
`base` vs `instruct` comparison out of the box. Training lets you study the
*transition* (e.g. "what did DPO break that SFT didn't?"), which is the most
resume-relevant story.

> These scripts run **real LoRA training on Tinker's remote GPUs** and incur
> usage. Start small (`--max-steps 50`, a 1B model) to validate, then scale.

## 1. SFT (base → chat-capable)

```bash
python training/train_sft.py \
    --model-name Qwen/Qwen3-8B-Base \
    --dataset no_robots \
    --lora-rank 32 --lr 1e-4 --epochs 1
```

Prints a `sampler_path` (`tinker://...`) and a `state_path`. Use the `state_path`
as the DPO starting point.

## 2. DPO (SFT → preference-optimized)

```bash
python training/train_dpo.py \
    --model-name Qwen/Qwen3-8B-Base \
    --load-checkpoint-path tinker://YOUR_SFT_STATE \
    --dataset hhh --dpo-beta 0.1 --lr 1e-5
```

Preference datasets supported: `hhh` (Anthropic HHH), `helpsteer3`, `ultrafeedback`.
Lower `--dpo-beta` = more aggressive optimization = more likely to surface
reward-hacking / sycophancy / verbosity drift.

## 3. Wire variants into the eval config

Paste the printed `sampler_path` values into `configs/eval.yaml`:

```yaml
  - name: sft
    base_model: "Qwen/Qwen3-8B-Base"   # tokenizer/renderer source
    model_path: "tinker://...SFT..."
    renderer: "qwen3_instruct"
    stage: sft
  - name: dpo
    base_model: "Qwen/Qwen3-8B-Base"
    model_path: "tinker://...DPO..."
    renderer: "qwen3_instruct"
    stage: dpo
```

Then run:

```bash
pte-eval run --config configs/eval.yaml
```

## Why this design

DPO/RLHF optimize a *proxy* (a preference/reward signal), not ground truth. The
classic failure is that the proxy improves while the true objective (truthfulness,
calibration, safety) silently degrades. By holding the base fixed and training SFT
then DPO from it, the harness can attribute each failure mode to a specific
optimization step.
