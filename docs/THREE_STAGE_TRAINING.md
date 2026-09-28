# Three-stage training and legal layer mappings

## Layer mapping contract

The syntax is `Receiver-layer:Sharer-layer`, and indices are zero-based.  A
pair is legal only when the Receiver index is within
`[0, receiver.num_hidden_layers)` and the Sharer index is within
`[0, sharer.num_hidden_layers)`.

For the requested reverse pair:

| role | model | depth |
|---|---|---:|
| Receiver | Qwen3-0.6B | 28 (`0..27`) |
| Sharer | Qwen2.5-0.5B-Instruct | 24 (`0..23`) |

The default starting mapping is:

```text
18:14,20:16,22:18,24:20
```

The mapping is validated against both model depths. Legal only means that the
indices and tensor shapes are constructible; it does not claim that this is
the scientifically optimal layer sweep. The one-click wrapper validates the
bounds before loading either model.

## One-command run

From the clean `Draft-KV` checkout:

```bash
bash script/draft_kv/run_three_stage_training.sh \
  --allow-stage1-gate-no-go \
  --gpu-id 0
```

The default paths point to the mounted shared storage under
`/workspace`.  Override paths when needed:

```bash
bash script/draft_kv/run_three_stage_training.sh \
  --receiver /path/to/qwen3 \
  --sharer /path/to/qwen25 \
  --layer-mapping 18:14,20:16,22:18,24:20 \
  --openhermes /path/to/openhermes.json \
  --data-root /path/to/datasets \
  --output-root /path/to/run \
  --gpu-id 0
```

`--allow-stage1-gate-no-go` is an explicit exploratory continuation switch.  It
does not bypass the mandatory Stage-1 Overfit GO check: the Stage-1 pilot
trainer requires the Overfit evaluation to be GO.  Without the switch, a
pilot Gate-val NO_GO stops before Stage 2.

## What the wrapper runs

All values below are passed explicitly so the run is auditable.

1. **Stage 1 — textual reconstruction**
   - OpenHermes: 4,096 train / 512 Gate-val / 512 reserve / 32 Overfit.
   - Overfit: 300 updates, microbatch 8, accumulation 4, projector LR `1e-3`,
     gate LR `1e-2`.
   - Pilot: 16,000 updates, microbatch 16, accumulation 4, evaluation every
     1,000 updates, same learning rates.
   - Runs Overfit evaluation and pilot Gate-val evaluation.  Reserve-test is
     never opened by this wrapper.

2. **Stage 2 — OpenHermes answer training**
   - Builds an immutable 9,216-row candidate pool excluding Stage-1 IDs,
     repairs non-EOS rows with the deterministic EOS-pool protocol, and then
     uses 8,192 train / 512 Gate-val / 512 reserve rows.
   - Trains 4,000 updates with microbatch 2, accumulation 16, projector LR
     `2e-4`, gate LR `1e-3`, and reconstruction replay every 5 updates.

3. **Stage 3 — MC option training**
   - Prepares ARC-E/ARC-C train and validation rows and a frozen Sharer cache;
     test rows remain reserved for downstream evaluation.
   - Trains 4,000 MC updates with one OpenHermes replay update after every four
     MC updates, microbatch 2, accumulation 8, projector LR `1e-4`, and gate
     LR `5e-4`.

The two language models remain frozen in all stages. Only the four KV
projectors (eight K/V matrices) and the per-Receiver-KV-head gates are
optimized.

## Outputs and reruns

The default output root is:

```text
/workspace/draft-kv/
  three_stage_qwen25_sharer_to_qwen3_receiver/
```

Important files are:

```text
run_manifest.json
stage1_reconstruction/data/
stage1_reconstruction/overfit_eval/eval_result.json
stage1_reconstruction/pilot_train/best.pt
stage1_reconstruction/gate_eval/eval_result.json
stage2_openhermes/eos_pool/
stage2_openhermes/train/last.pt
stage3_mc/data/
stage3_mc/train/last.pt
stage3_mc/train/best.pt
logs/
```

Completed directories are reused on a rerun.  Partial training directories are
not silently resumed by the underlying training programs; use a new output
root after an interrupted training job.  The Stage-2 candidate program may
return non-zero when it intentionally finds truncated drafts.  The wrapper
accepts that only when the complete candidate records and cache exist, then
hands them to the EOS-pool repair.  Any other failure stops the run.
