# Task-Aligned Draft-KV

This repository implements three-stage latent communication for a frozen
Sharer/Receiver model pair. A small Draft-KV bridge projects selected Sharer
KV states into the Receiver through gated cross-attention.

## Training

1. **Stage 1, reconstruction.** The Sharer generates a complete answer and the
   Receiver reconstructs that answer from the projected generated-token KV.
2. **Stage 2, answer alignment.** The bridge is optimized on OpenHermes
   answers while periodically replaying the reconstruction objective.
3. **Stage 3, task alignment.** The bridge is optimized on ARC-Easy and
   ARC-Challenge option logits. A one-sided protection term limits the harm
   caused by wrongly paired Sharer KV without rewarding deliberately worse
   negative controls.

The Stage 3 objective is

```text
CE(Matched, gold)
+ lambda * relu(NLL(Deranged) - stopgrad(NLL(Zero)) - tolerance)
```

Defaults are `lambda=0.1` and `tolerance=0.1` nats. The Sharer and Receiver
language models remain frozen. Only the existing KV projections and per-head
gates are trained.

## Entry Points

Run the complete pipeline from the repository root:

```bash
bash script/draft_kv/run_three_stage_training.sh --gpu-id 0
```

Run Stage 3 independently after Stage 2 has completed:

```bash
bash script/draft_kv/run_stage3_train_and_eval.sh --gpu-id 0
```

Individual stages remain available through `run_stage3_training.sh`,
`run_stage3_eval_all.sh`, and `run_stage3_eval.sh`.

See [the Stage 3 protocol](docs/STAGE3.md),
[the method](docs/METHOD.md), and
[the three-stage protocol](docs/THREE_STAGE_TRAINING.md) for details.

Model weights, datasets, checkpoints, and generated caches are intentionally
not stored in this repository.
