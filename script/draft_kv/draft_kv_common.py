"""Shared utilities for Draft-KV preparation, training and evaluation."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from draft_kv.model.draft_kv import DraftKVModel


DEFAULT_RECEIVER = "/workspace/models/Qwen2.5-0.5B-Instruct"
DEFAULT_SHARER = "/workspace/models/Qwen3-0.6B"
DEFAULT_OUTPUT = (
    "/workspace/draft-kv"
)
DEFAULT_MAPPING = "14:18,16:20,18:22,20:24"


def seed_all(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def write_json(path: str | Path, payload: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=float) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_layer_mapping(value: str | Mapping[int, int]) -> Dict[int, int]:
    if isinstance(value, Mapping):
        mapping = {int(target): int(source) for target, source in value.items()}
    else:
        mapping = {}
        for item in str(value).split(","):
            target, source = item.strip().split(":", 1)
            mapping[int(target)] = int(source)
    if not mapping:
        raise ValueError("layer mapping cannot be empty")
    return mapping


def answer_nll(logits: Tensor, labels: Tensor) -> Dict[str, Tensor]:
    labels = labels.to(logits.device)
    shifted_logits = logits[:, :-1].float()
    shifted_labels = labels[:, 1:]
    losses = F.cross_entropy(
        shifted_logits.reshape(-1, shifted_logits.shape[-1]),
        shifted_labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).reshape(shifted_labels.shape)
    valid = shifted_labels.ne(-100)
    sums = (losses * valid).sum(dim=1)
    counts = valid.sum(dim=1)
    means = sums / counts.clamp_min(1)
    return {"sum": sums, "count": counts, "mean": means}


def bootstrap_ci(
    differences: Sequence[float],
    *,
    samples: int = 10000,
    seed: int = 91827,
) -> Dict[str, float]:
    values = np.asarray(list(differences), dtype=np.float64)
    if values.size == 0:
        raise ValueError("cannot bootstrap an empty difference vector")
    rng = np.random.default_rng(int(seed))
    means = np.empty(int(samples), dtype=np.float64)
    chunk = 256
    for start in range(0, int(samples), chunk):
        end = min(int(samples), start + chunk)
        selected = rng.integers(0, values.size, size=(end - start, values.size))
        means[start:end] = values[selected].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci95_lower": float(np.percentile(means, 2.5)),
        "ci95_upper": float(np.percentile(means, 97.5)),
        "positive_fraction": float(np.mean(values > 0)),
        "n": int(values.size),
    }


def prepare_tokenizer(path: str, *, padding_side: str) -> Any:
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    tokenizer.padding_side = padding_side
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_causal_lm(path: str, device: torch.device) -> Any:
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    return (
        AutoModelForCausalLM.from_pretrained(
            path,
            local_files_only=True,
            torch_dtype=dtype,
            attn_implementation="eager",
        )
        .eval()
        .to(device)
    )


def build_model(
    *,
    receiver_path: str,
    sharer_path: str,
    layer_mapping: str | Mapping[int, int],
    device_name: str,
) -> Tuple[DraftKVModel, Any, Any]:
    device = torch.device(device_name)
    receiver_tokenizer = prepare_tokenizer(receiver_path, padding_side="right")
    sharer_tokenizer = prepare_tokenizer(sharer_path, padding_side="right")
    receiver = load_causal_lm(receiver_path, device)
    sharer = load_causal_lm(sharer_path, device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = DraftKVModel(
        receiver,
        sharer,
        parse_layer_mapping(layer_mapping),
    ).to(device=device, dtype=dtype)
    return model, receiver_tokenizer, sharer_tokenizer


def trainable_state(model: DraftKVModel) -> Dict[str, Any]:
    return {
        "projection": model.projection.state_dict(),
        "gate_logits": {
            layer: module.gate_logits.detach().cpu()
            for layer, module in model.consumer.external.items()
        },
    }


def load_trainable_state(
    model: DraftKVModel, payload: Mapping[str, Any]
) -> None:
    state = payload.get("communication_state", payload)
    model.projection.load_state_dict(state["projection"], strict=True)
    expected = set(model.consumer.external.keys())
    observed = set(state["gate_logits"])
    if expected != observed:
        raise RuntimeError("checkpoint gate layers do not match the model")
    with torch.no_grad():
        for layer, values in state["gate_logits"].items():
            model.consumer.external[layer].gate_logits.copy_(
                values.to(model.device)
            )


def count_parameters(parameters: Sequence[torch.nn.Parameter]) -> int:
    return int(sum(parameter.numel() for parameter in parameters))


__all__ = [
    "DEFAULT_MAPPING",
    "DEFAULT_OUTPUT",
    "DEFAULT_RECEIVER",
    "DEFAULT_SHARER",
    "answer_nll",
    "bootstrap_ci",
    "build_model",
    "count_parameters",
    "load_causal_lm",
    "load_trainable_state",
    "parse_layer_mapping",
    "prepare_tokenizer",
    "seed_all",
    "sha256_file",
    "trainable_state",
    "write_json",
]
