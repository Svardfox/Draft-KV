# Stage 3: task alignment

Stage 3 starts from the OpenHermes Stage-2 `last.pt` checkpoint and keeps the
Sharer and Receiver frozen. Only the existing KV projections and per-head
gates are updated. ARC-Easy and ARC-Challenge train/validation rows are split
deterministically into training and calibration partitions; ARC test rows are
opened only by the final evaluator after calibration selection.

## Objective

For each example the Receiver is scored in paired conditions:

- `Zero`: no communication;
- `Matched`: the same example's Sharer packet;
- `Deranged`: another example's Sharer packet.

Training minimizes

```text
CE(Matched, gold)
+ lambda * relu(NLL(Deranged) - stopgrad(NLL(Zero)) - tolerance)
```

The default protection weight is `0.1` and the tolerance is `0.1` nats. The
term acts only when the wrongly paired packet is worse than the frozen
no-communication baseline by more than the tolerance, so the objective does
not reward deliberately degrading the negative control. Four option updates
are followed by one OpenHermes reconstruction replay update.

A checkpoint is eligible for selection only when reconstruction preservation
passes. `best.pt` maximizes calibration Matched accuracy, with lower Matched
NLL as the tie-breaker.

## Entry Points

```bash
bash script/draft_kv/run_stage3_train_and_eval.sh --gpu-id 0
```

Individual stages are available as:

```bash
bash script/draft_kv/run_stage3_training.sh --gpu-id 0
DRAFT_KV_GPU_ID=0 bash script/draft_kv/run_stage3_eval_all.sh
```
