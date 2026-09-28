# Draft-KV method

## Interface

The sharer runs on the full prompt and exposes unrotated key/value states from
selected decoder layers.  `DraftKVProjector` maps each selected sharer KV
tensor to the receiver's KV-head shape.  The receiver's corresponding layers
consume those projected KVs through gated cross-attention.  Draft masks ensure
that only generated sharer tokens are communicated; prompt tokens are not
silently reused as a second input channel.

The bridge is the only trainable part.  The receiver and sharer language-model
weights are frozen, and the zero-gate setting is exactly the native receiver
baseline (a required parity invariant).

## Three-stage training

Stage 1 uses OpenHermes conversations.  The sharer generates a complete
assistant answer, its generated-token KV packet is translated, and the
receiver is teacher-forced on that same answer.  The reconstruction objective
is deliberately content-bearing: it tests whether the receiver can decode the
message carried by the draft KV.

Stage 2 starts from that communication-capable bridge and optimizes the
OpenHermes answer objective with a fixed reconstruction replay mixture.

Stage 3 starts from the Stage-2 bridge and optimizes ARC-Easy/ARC-Challenge
option-token cross-entropy.  A one-sided protection term penalizes a wrongly
paired packet only when it is worse than the no-communication baseline by more
than a tolerance.  The baseline is detached, so the objective never rewards
deliberately degrading the negative control.

## Controls and reproducibility

Every evaluator emits native/zero parity, matched and deranged conditions,
per-example records, a manifest and SHA-256 digests.  A deranged condition
uses a no-fixed-point donor permutation, so a matched-versus-deranged gap is a
direct donor-sensitivity check.  The scripts accept explicit model paths,
layer mapping, device and checkpoint paths; no model or dataset artifact is
vendored in this repository.
