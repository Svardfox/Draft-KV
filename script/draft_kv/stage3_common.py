#!/usr/bin/env python3
"""Shared Stage 3 datasets, packet helpers, validation, and replay updates."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from draft_kv.train.draft_kv_mc_training_data import (  # noqa: E402
    DraftKVMCOptionCollator,
    DraftKVMCOptionDataset,
    option_classification_stats,
    option_distribution_kl,
    option_token_ids,
)
from draft_kv.train.draft_kv_reconstruction_data import (  # noqa: E402
    OpenHermesDraftKVReconstructionDataset,
)
from script.draft_kv.draft_kv_reconstruction_common import (  # noqa: E402
    token_reconstruction_stats,
)


class _CyclingLoader:
    def __init__(self, loader: DataLoader) -> None:
        self.loader = loader
        self.iterator: Iterator[Mapping[str, Any]] = iter(loader)

    def next(self) -> Mapping[str, Any]:
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            return next(self.iterator)


def _canonical_mapping(value: Mapping[Any, Any]) -> Dict[int, int]:
    return {int(target): int(source) for target, source in value.items()}


def _validate_initial_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    path: Path,
    manifest: Mapping[str, Any],
) -> None:
    if path.name != "last.pt":
        raise RuntimeError(
            "Stage 3 initialization must be an existing Stage-2 last.pt"
        )
    if not str(checkpoint.get("protocol", "")).startswith(
        "draft_kv_openhermes_stage2"
    ):
        raise RuntimeError(
            "initial last.pt is not an OpenHermes Stage-2 checkpoint"
        )
    if "communication_state" not in checkpoint:
        raise RuntimeError("initial checkpoint has no communication state")
    for name in ("receiver", "sharer"):
        if Path(str(checkpoint.get(name, ""))).resolve() != Path(
            str(manifest[name])
        ).resolve():
            raise RuntimeError(f"initial checkpoint {name} differs from MC manifest")
    if _canonical_mapping(checkpoint.get("layer_mapping", {})) != _canonical_mapping(
        manifest["layer_mapping"]
    ):
        raise RuntimeError("initial checkpoint layer mapping differs from MC data")
    if "stage2_data_manifest_sha256" not in checkpoint:
        raise RuntimeError("initial last.pt is not an OpenHermes Stage-2 checkpoint")


def _mc_dataset(
    manifest: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    drafts: Mapping[str, Mapping[str, Any]],
    receiver_tokenizer: Any,
    sharer_tokenizer: Any,
    *,
    split: str,
) -> DraftKVMCOptionDataset:
    return DraftKVMCOptionDataset(
        records,
        drafts,
        receiver_tokenizer,
        sharer_tokenizer,
        split=split,
        max_receiver_length=int(manifest["lengths"]["max_receiver_length"]),
        max_sharer_length=int(manifest["lengths"]["max_sharer_length"]),
    )


def _reconstruction_dataset(
    manifest: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    receiver_tokenizer: Any,
    sharer_tokenizer: Any,
    *,
    split: str,
) -> OpenHermesDraftKVReconstructionDataset:
    lengths = manifest["lengths"]
    return OpenHermesDraftKVReconstructionDataset(
        records,
        receiver_tokenizer,
        sharer_tokenizer,
        split=split,
        min_message_tokens=int(lengths["min_message_tokens"]),
        max_message_tokens=int(lengths["max_message_tokens"]),
        max_receiver_length=int(lengths["max_receiver_length"]),
        max_sharer_length=int(lengths["max_sharer_length"]),
    )


def _packet(
    model: Any,
    packet_batch: Mapping[str, torch.Tensor],
    *,
    detach: bool,
) -> Any:
    return model.make_packet(
        packet_batch["sharer_input_ids"],
        packet_batch["sharer_attention_mask"],
        packet_batch["sharer_draft_mask"],
        detach=detach,
    )


def _matched_deranged_packet_batch(
    batch: Mapping[str, Any],
    *,
    dataset: DraftKVMCOptionDataset,
    collator: DraftKVMCOptionCollator,
    derangement: Mapping[str, str],
) -> Dict[str, torch.Tensor]:
    matched = [dataset.by_id[str(value)] for value in batch["example_id"]]
    deranged = [
        dataset.by_id[str(derangement[str(value)])] for value in batch["example_id"]
    ]
    width = max(
        len(row["sharer_input_ids"]) for row in matched + deranged
    )
    return collator.packet_batch(matched + deranged, width=width)


def _condition_packet_batch(
    example_ids: Sequence[str],
    *,
    condition: str,
    dataset: DraftKVMCOptionDataset,
    collator: DraftKVMCOptionCollator,
    derangement: Mapping[str, str],
    common_width: int,
) -> Dict[str, torch.Tensor]:
    donor_ids = (
        list(example_ids)
        if condition == "matched"
        else [str(derangement[str(value)]) for value in example_ids]
    )
    return collator.packet_batch(
        [dataset.by_id[str(value)] for value in donor_ids],
        width=common_width,
    )


@torch.no_grad()
def mc_validation(
    model: Any,
    dataset: DraftKVMCOptionDataset,
    collator: DraftKVMCOptionCollator,
    *,
    derangement: Mapping[str, str],
    option_ids: Sequence[int],
    batch_size: int,
) -> Dict[str, Any]:
    """Evaluate calibration in paired Zero/Matched/Deranged conditions."""

    model.set_stage("eval")
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=0,
        collate_fn=collator,
    )
    aggregate = {
        condition: {"nll": [], "gold_log_probability": [], "correct": []}
        for condition in ("zero", "matched", "deranged")
    }
    zero_deranged_kl: list[float] = []
    rows: list[Dict[str, Any]] = []
    for batch in loader:
        ids = batch["receiver_input_ids"]
        attention = batch["receiver_attention_mask"]
        zero_logits, _, _ = model.forward_receiver(
            ids,
            attention,
            packet=None,
            disable_communication=True,
            use_cache=False,
        )
        matched_features = [
            dataset.by_id[str(value)] for value in batch["example_id"]
        ]
        deranged_features = [
            dataset.by_id[str(derangement[str(value)])]
            for value in batch["example_id"]
        ]
        common_width = max(
            len(row["sharer_input_ids"])
            for row in matched_features + deranged_features
        )
        condition_logits = {"zero": zero_logits}
        for condition in ("matched", "deranged"):
            packet_batch = _condition_packet_batch(
                batch["example_id"],
                condition=condition,
                dataset=dataset,
                collator=collator,
                derangement=derangement,
                common_width=common_width,
            )
            packet = _packet(model, packet_batch, detach=True)
            logits, _, _ = model.forward_receiver(
                ids,
                attention,
                packet=packet,
                disable_communication=False,
                use_cache=False,
            )
            condition_logits[condition] = logits

        stats_by_condition = {}
        for condition, logits in condition_logits.items():
            stats = option_classification_stats(
                logits[:, -1],
                option_ids=option_ids,
                option_counts=batch["option_count"],
                gold_indices=batch["answer_index"],
            )
            stats_by_condition[condition] = stats
            for key in aggregate[condition]:
                aggregate[condition][key].extend(
                    stats[key].detach().cpu().tolist()
                )
        zero_deranged_kl.extend(
            option_distribution_kl(
                stats_by_condition["zero"]["candidate_logits"],
                stats_by_condition["deranged"]["candidate_logits"],
                option_counts=batch["option_count"],
            )
            .detach()
            .cpu()
            .tolist()
        )
        for index, example_id in enumerate(batch["example_id"]):
            for condition in ("zero", "matched", "deranged"):
                stats = stats_by_condition[condition]
                prediction = int(stats["prediction"][index])
                rows.append(
                    {
                        "example_id": str(example_id),
                        "dataset": str(batch["dataset"][index]),
                        "condition": condition,
                        "donor_example_id": (
                            None
                            if condition == "zero"
                            else str(example_id)
                            if condition == "matched"
                            else str(derangement[str(example_id)])
                        ),
                        "gold_index": int(batch["answer_index"][index]),
                        "prediction_index": prediction,
                        "nll": float(stats["nll"][index]),
                        "gold_log_probability": float(
                            stats["gold_log_probability"][index]
                        ),
                        "correct": bool(stats["correct"][index]),
                    }
                )
    summary: Dict[str, Any] = {"count": len(dataset), "conditions": {}}
    for condition, values in aggregate.items():
        summary["conditions"][condition] = {
            "accuracy": float(np.mean(values["correct"])),
            "mean_nll": float(np.mean(values["nll"])),
            "mean_gold_log_probability": float(
                np.mean(values["gold_log_probability"])
            ),
        }
    summary["gaps"] = {
        "matched_minus_zero_accuracy": (
            summary["conditions"]["matched"]["accuracy"]
            - summary["conditions"]["zero"]["accuracy"]
        ),
        "matched_minus_deranged_accuracy": (
            summary["conditions"]["matched"]["accuracy"]
            - summary["conditions"]["deranged"]["accuracy"]
        ),
        "matched_minus_zero_gold_log_probability": (
            summary["conditions"]["matched"]["mean_gold_log_probability"]
            - summary["conditions"]["zero"]["mean_gold_log_probability"]
        ),
        "matched_minus_deranged_gold_log_probability": (
            summary["conditions"]["matched"]["mean_gold_log_probability"]
            - summary["conditions"]["deranged"]["mean_gold_log_probability"]
        ),
        "deranged_minus_zero_accuracy": (
            summary["conditions"]["deranged"]["accuracy"]
            - summary["conditions"]["zero"]["accuracy"]
        ),
        "deranged_minus_zero_gold_log_probability": (
            summary["conditions"]["deranged"]["mean_gold_log_probability"]
            - summary["conditions"]["zero"]["mean_gold_log_probability"]
        ),
    }
    summary["consistency"] = {
        "zero_to_deranged_mean_kl": float(np.mean(zero_deranged_kl)),
    }
    summary["rows"] = rows
    return summary


@torch.no_grad()
def reconstruction_validation(model: Any, loader: DataLoader) -> Dict[str, float]:
    model.set_stage("eval")
    values = {"zero": [], "matched": []}
    correct = {"zero": 0, "matched": 0}
    counts = {"zero": 0, "matched": 0}
    for batch in loader:
        packet = model.make_packet(
            batch["sharer_input_ids"],
            batch["sharer_attention_mask"],
            batch["sharer_draft_mask"],
            detach=True,
        )
        for condition in ("zero", "matched"):
            logits, _, _ = model.forward_receiver(
                batch["receiver_input_ids"],
                batch["receiver_attention_mask"],
                packet=packet if condition == "matched" else None,
                disable_communication=condition == "zero",
                use_cache=False,
            )
            stats = token_reconstruction_stats(logits, batch["labels"])
            values[condition].extend(float(value) for value in stats["mean"])
            correct[condition] += int(stats["correct"].sum())
            counts[condition] += int(stats["count"].sum())
    zero = float(np.mean(values["zero"]))
    matched = float(np.mean(values["matched"]))
    return {
        "zero_mean_nll": zero,
        "matched_mean_nll": matched,
        "zero_minus_matched_nll": zero - matched,
        "zero_token_accuracy": correct["zero"] / max(1, counts["zero"]),
        "matched_token_accuracy": correct["matched"] / max(1, counts["matched"]),
        "count": len(values["zero"]),
    }


def _train_replay_step(
    *,
    model: Any,
    optimizer: torch.optim.Optimizer,
    cycle: _CyclingLoader,
    grad_accum: int,
    trainable: Sequence[torch.nn.Parameter],
) -> Dict[str, float]:
    optimizer.zero_grad(set_to_none=True)
    loss_total = 0.0
    correct = 0
    count = 0
    for _ in range(int(grad_accum)):
        batch = cycle.next()
        packet = model.make_packet(
            batch["sharer_input_ids"],
            batch["sharer_attention_mask"],
            batch["sharer_draft_mask"],
            detach=False,
        )
        logits, _, _ = model.forward_receiver(
            batch["receiver_input_ids"],
            batch["receiver_attention_mask"],
            packet=packet,
            disable_communication=False,
            use_cache=False,
        )
        stats = token_reconstruction_stats(logits, batch["labels"])
        loss = stats["sum"].sum() / stats["count"].sum().clamp_min(1)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("non-finite reconstruction replay loss")
        (loss / float(grad_accum)).backward()
        loss_total += float(loss.detach())
        correct += int(stats["correct"].sum())
        count += int(stats["count"].sum())
    if any(
        parameter.grad is not None
        and not bool(torch.isfinite(parameter.grad).all())
        for parameter in trainable
    ):
        raise RuntimeError("non-finite replay communication gradient")
    gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
    optimizer.step()
    return {
        "loss": loss_total / float(grad_accum),
        "token_accuracy": correct / max(1, count),
        "gradient_norm_before_clip": float(gradient_norm),
    }
