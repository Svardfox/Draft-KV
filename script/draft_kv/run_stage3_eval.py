#!/usr/bin/env python3
"""Evaluate a Stage 3 checkpoint on multiple-choice downstream datasets.

The frozen Sharer first generates and caches a response for every question.
The frozen Receiver is then scored by next-option logits under paired Zero,
Matched, and full-packet Deranged conditions.  ``--latency-only`` provides a
separate Matched-only path with per-question timing.  The cache
and evaluation rows are resumable and bound to immutable manifests.  Use
``--conditions matched deranged`` to omit the invariant Zero path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
import torch
from transformers import AutoTokenizer

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from draft_kv.train.downstream_mc_data import (  # noqa: E402
    build_plain_multiple_choice_prompt,
    build_multiple_choice_prompt,
    canonical_dataset_name,
    extract_choice,
    validate_normalized_row,
)
from draft_kv.train.draft_kv_reconstruction_data import (  # noqa: E402
    make_no_fixed_point_id_mapping,
)
from script.draft_kv.draft_kv_common import (  # noqa: E402
    bootstrap_ci,
    build_model,
    load_trainable_state,
    parse_layer_mapping,
    seed_all,
    sha256_file,
)
from script.draft_kv.make_downstream_calibration_split import (  # noqa: E402
    example_ids_file_binding as _example_ids_file_binding,
    load_example_ids_file as _load_example_ids_file,
    select_examples_by_id as _select_examples_by_id,
)


PROTOCOL = "draft_kv_stage3_downstream_mc"
CONDITIONS = ("Zero", "Matched", "Deranged")
CONDITION_CHOICES = tuple(value.lower() for value in CONDITIONS)
DEFAULT_RECEIVER = "/workspace/models/Qwen2.5-0.5B-Instruct"
DEFAULT_SHARER = "/workspace/models/Qwen3-0.6B"
DEFAULT_MAPPING = "14:18,16:20,18:22,20:24"
DEFAULT_CHECKPOINT = (
    "/workspace/draft-kv/"
    "stage3_mc_training/train/best.pt"
)
DEFAULT_DATA_ROOT = "/workspace/datasets"
DEFAULT_OUTPUT_ROOT = (
    "/workspace/draft-kv/"
    "stage3_mc_training/downstream"
)
DEFAULT_PAIR = "qwen3-0.6b_to_qwen2.5-0.5b-instruct"
DEFAULT_MAX_SHARER_INPUT_TOKENS = 1280
DEFAULT_MAX_RECEIVER_INPUT_TOKENS = 1280
DEFAULT_DRAFT_BATCH_SIZE = 128
PROMPT_STYLES = ("reasoning", "plain")
PLAIN_ANSWER_PREFIX = "The correct answer is"
LATENCY_PROTOCOL = "draft_kv_stage3_matched_latency"


def _normalize_conditions(values: Sequence[str], *, latency_only: bool = False) -> tuple[str, ...]:
    if latency_only:
        return ("Matched",)
    result = []
    for value in values:
        name = str(value).strip().lower().capitalize()
        if name not in CONDITIONS:
            raise ValueError(f"unknown evaluation condition: {value}")
        if name not in result:
            result.append(name)
    if not result:
        raise ValueError("at least one evaluation condition is required")
    return tuple(result)


def _read_jsonl(path: Path) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(dict(json.loads(line)))
            except json.JSONDecodeError as error:
                raise RuntimeError(f"invalid JSONL at {path}:{line_number}") from error
    return rows


def _atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
    temporary.replace(path)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            dict(value),
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
            default=float,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _default_split(dataset: str) -> str:
    # C-EVAL's public, labeled benchmark partition is ``val``.  Its ``test``
    # labels are withheld in the standard release and cannot support a local
    # accuracy report.
    return "val" if dataset == "ceval" else "test"


def _data_path(data_root: Path, dataset: str, split: str) -> Path:
    if dataset == "mmlu-redux":
        return data_root / "mmlu-redux-2.0" / "adapted" / f"{split}.jsonl"
    if dataset == "arc-e" or dataset == "arc-c":
        return data_root / "ai2_arc" / "adapted" / dataset / f"{split}.jsonl"
    if dataset == "openbookqa":
        return data_root / "openbookqa" / "adapted" / f"{split}.jsonl"
    if dataset == "ceval":
        return data_root / "ceval" / "adapted" / f"{split}.jsonl"
    if dataset in {"gsm-mc", "math-mc"}:
        return data_root / "mc-evaluation" / dataset / "adapted" / f"{split}.jsonl"
    raise ValueError(f"unsupported dataset {dataset!r}")


def _dataset_manifest_entry(
    data_root: Path, dataset: str, data_path: Path
) -> Dict[str, Any]:
    if dataset == "mmlu-redux":
        dataset_root = data_root / "mmlu-redux-2.0"
        manifest_path = dataset_root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"missing normalized dataset manifest: {manifest_path}"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        files = manifest.get("files", [])
    elif dataset in {"arc-e", "arc-c"}:
        dataset_root = data_root / "ai2_arc"
        manifest_path = dataset_root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"missing normalized dataset manifest: {manifest_path}"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        files = manifest.get("datasets", {}).get(dataset, {}).get("files", [])
    elif dataset == "openbookqa":
        dataset_root = data_root / "openbookqa"
        manifest_path = dataset_root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"missing normalized dataset manifest: {manifest_path}"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        files = manifest.get("files", [])
    elif dataset == "ceval":
        dataset_root = data_root / "ceval"
        manifest_path = dataset_root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"missing normalized dataset manifest: {manifest_path}"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        files = manifest.get("datasets", {}).get("ceval", {}).get("files", [])
    elif dataset in {"gsm-mc", "math-mc"}:
        dataset_root = data_root / "mc-evaluation"
        manifest_path = dataset_root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"missing normalized dataset manifest: {manifest_path}"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        files = manifest.get("datasets", {}).get(dataset, {}).get("files", [])
    else:
        raise ValueError(f"unsupported dataset {dataset!r}")
    relative = str(data_path.relative_to(dataset_root))
    matches = [row for row in files if str(row.get("path")) == relative]
    if len(matches) != 1:
        raise RuntimeError(f"dataset manifest does not uniquely register {relative}")
    entry = dict(matches[0])
    if int(entry.get("count", -1)) <= 0:
        raise RuntimeError("dataset manifest registers an empty adapted file")
    observed_sha = sha256_file(data_path)
    if str(entry.get("sha256")) != observed_sha:
        raise RuntimeError("adapted dataset SHA256 differs from its download manifest")
    return {
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "adapted_file": entry,
    }


def _model_artifact_hashes(path: str) -> Dict[str, str]:
    root = Path(path)
    candidates: set[Path] = set(root.glob("*.safetensors"))
    for name in (
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "chat_template.jinja",
        "vocab.json",
        "merges.txt",
        "model.safetensors.index.json",
    ):
        candidate = root / name
        if candidate.is_file():
            candidates.add(candidate)
    if not any(path.suffix == ".safetensors" for path in candidates):
        raise RuntimeError(f"model directory has no safetensors weights: {root}")
    return {
        str(candidate.relative_to(root)): sha256_file(candidate)
        for candidate in sorted(candidates)
    }


def _selection_sha256(example_ids: Sequence[str]) -> str:
    payload = json.dumps(list(example_ids), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _configuration_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        dict(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _safe_tag(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value)).strip("-.")
    return result or "model"


def _chat_text(tokenizer: Any, prompt: str, *, assistant_prefix: str = "") -> str:
    messages = [{"role": "user", "content": str(prompt)}]
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    try:
        text = tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        text = tokenizer.apply_chat_template(messages, **kwargs)
    return text + str(assistant_prefix)


def _chat_ids(tokenizer: Any, prompt: str, *, assistant_prefix: str = "") -> list[int]:
    text = _chat_text(tokenizer, prompt, assistant_prefix=assistant_prefix)
    values = tokenizer.encode(text, add_special_tokens=False)
    if not values:
        raise RuntimeError("chat template produced an empty prompt")
    return [int(value) for value in values]


def _evaluation_prompt(
    row: Mapping[str, Any], *, prompt_style: str, use_cot: bool
) -> str:
    if prompt_style == "reasoning":
        return build_multiple_choice_prompt(row, use_cot=use_cot)
    if prompt_style == "plain":
        if use_cot:
            raise ValueError("plain prompt style does not support CoT")
        return build_plain_multiple_choice_prompt(row)
    raise ValueError(f"unknown prompt style: {prompt_style}")


def _sharer_assistant_prefix(prompt_style: str) -> str:
    return PLAIN_ANSWER_PREFIX if prompt_style == "plain" else ""


def _prompt_length_audit(
    tokenizer: Any,
    examples: Sequence[Mapping[str, Any]],
    *,
    use_cot: bool,
    assistant_prefix: str,
    limit: int,
    role: str,
    prompt_style: str,
) -> Dict[str, Any]:
    maximum = -1
    maximum_id = ""
    over_limit: list[tuple[int, str]] = []
    audit_batch = 256
    for start in range(0, len(examples), audit_batch):
        chunk = examples[start : start + audit_batch]
        texts = [
            _chat_text(
                tokenizer,
                _evaluation_prompt(
                    row,
                    prompt_style=prompt_style,
                    use_cot=use_cot,
                ),
                assistant_prefix=assistant_prefix,
            )
            for row in chunk
        ]
        encoded = tokenizer(
            texts,
            add_special_tokens=False,
            padding=False,
            truncation=False,
            return_length=True,
        )
        for row, raw_length in zip(chunk, encoded["length"]):
            length = int(raw_length)
            example_id = str(row["example_id"])
            if length > maximum:
                maximum = length
                maximum_id = example_id
            if length > int(limit):
                over_limit.append((length, example_id))
    over_limit.sort(reverse=True)
    result = {
        "role": role,
        "count": len(examples),
        "limit": int(limit),
        "maximum_tokens": maximum,
        "maximum_example_id": maximum_id,
        "over_limit_count": len(over_limit),
        "largest_over_limit": [
            {"tokens": length, "example_id": example_id}
            for length, example_id in over_limit[:10]
        ],
    }
    if over_limit:
        first = over_limit[0]
        raise RuntimeError(
            f"{role} prompt length preflight failed: "
            f"{len(over_limit)} examples exceed limit {limit}; "
            f"maximum {first[0]} at {first[1]}"
        )
    print(
        f"{role} prompt length preflight PASS: "
        f"max={maximum}/{limit} example={maximum_id}",
        flush=True,
    )
    return result


def _trim_generated(
    values: Sequence[int], *, eos_ids: set[int], pad_token_id: int
) -> tuple[list[int], bool]:
    result: list[int] = []
    terminated = False
    for raw in values:
        value = int(raw)
        if value == int(pad_token_id) and value not in eos_ids:
            break
        result.append(value)
        if value in eos_ids:
            terminated = True
            break
    return result, terminated


def _eos_ids(tokenizer: Any) -> set[int]:
    value = tokenizer.eos_token_id
    if isinstance(value, (list, tuple, set)):
        return {int(item) for item in value}
    return set() if value is None else {int(value)}


def _ordered_cache(
    cache: Mapping[str, Mapping[str, Any]], example_ids: Sequence[str]
) -> list[Mapping[str, Any]]:
    return [cache[example_id] for example_id in example_ids if example_id in cache]


@torch.no_grad()
def _generate_sharer_cache(
    model: Any,
    tokenizer: Any,
    examples: Sequence[Mapping[str, Any]],
    cache_path: Path,
    *,
    batch_size: int,
    max_input_tokens: int,
    max_new_tokens: int,
    save_every_batches: int,
    prompt_style: str,
) -> Dict[str, Dict[str, Any]]:
    existing_rows = _read_jsonl(cache_path) if cache_path.exists() else []
    cache = {str(row["example_id"]): row for row in existing_rows}
    if len(cache) != len(existing_rows):
        raise RuntimeError("Sharer cache contains duplicate example IDs")
    example_ids = [str(row["example_id"]) for row in examples]
    if not set(cache).issubset(example_ids):
        raise RuntimeError("Sharer cache contains IDs outside this manifest")
    cached_ids = [str(row["example_id"]) for row in existing_rows]
    if cached_ids != example_ids[: len(cached_ids)]:
        raise RuntimeError("Sharer cache is not an ordered manifest prefix")
    if len(cached_ids) != len(example_ids) and len(cached_ids) % int(batch_size):
        raise RuntimeError(
            "incomplete Sharer cache ends inside a fixed generation batch"
        )
    by_id = {str(row["example_id"]): row for row in examples}
    for example_id, row in cache.items():
        expected_prompt = _chat_ids(
            tokenizer,
            _evaluation_prompt(
                by_id[example_id],
                prompt_style=prompt_style,
                use_cot=prompt_style == "reasoning",
            ),
            assistant_prefix=_sharer_assistant_prefix(prompt_style),
        )
        prompt = [int(value) for value in row.get("prompt_token_ids", [])]
        draft = [int(value) for value in row.get("draft_token_ids", [])]
        response = tokenizer.decode(draft, skip_special_tokens=True).strip()
        if (
            not draft
            or not response
            or response != str(row.get("response", "")).strip()
        ):
            raise RuntimeError(
                f"cached Sharer row fails replay validation: {example_id}"
            )
    missing = [row for row in examples if str(row["example_id"]) not in cache]
    previous_padding = tokenizer.padding_side
    tokenizer.padding_side = "left"
    eos_ids = _eos_ids(tokenizer)
    try:
        for batch_index, start in enumerate(range(0, len(missing), int(batch_size))):
            chunk = missing[start : start + int(batch_size)]
            prompt_ids = [
                _chat_ids(
                    tokenizer,
                    _evaluation_prompt(
                        row,
                        prompt_style=prompt_style,
                        use_cot=prompt_style == "reasoning",
                    ),
                    assistant_prefix=_sharer_assistant_prefix(prompt_style),
                )
                for row in chunk
            ]
            longest = max(len(values) for values in prompt_ids)
            if longest > int(max_input_tokens):
                culprit = chunk[prompt_ids.index(max(prompt_ids, key=len))]
                raise RuntimeError(
                    f"Sharer prompt exceeds --max-sharer-input-tokens: "
                    f"{culprit['example_id']} length={longest}"
                )
            encoded = tokenizer.pad(
                {"input_ids": prompt_ids},
                padding=True,
                return_attention_mask=True,
                return_tensors="pt",
            )
            encoded = {key: value.to(model.device) for key, value in encoded.items()}
            width = int(encoded["input_ids"].shape[1])
            generated = model.sharer.generate(
                **encoded,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                max_new_tokens=int(max_new_tokens),
                use_cache=True,
                pad_token_id=int(tokenizer.pad_token_id),
                eos_token_id=tokenizer.eos_token_id,
            )
            for position, example in enumerate(chunk):
                draft_ids, terminated = _trim_generated(
                    generated[position, width:].tolist(),
                    eos_ids=eos_ids,
                    pad_token_id=int(tokenizer.pad_token_id),
                )
                response = tokenizer.decode(draft_ids, skip_special_tokens=True).strip()
                if not draft_ids or not response:
                    raise RuntimeError(
                        f"Sharer generated an empty response: {example['example_id']}"
                    )
                cache[str(example["example_id"])] = {
                    "example_id": str(example["example_id"]),
                    "prompt_token_ids": prompt_ids[position],
                    "draft_token_ids": draft_ids,
                    "terminated_eos": bool(terminated),
                    "response": response,
                }
            completed = len(cache)
            print(f"Sharer cache {completed}/{len(examples)}", flush=True)
            if (batch_index + 1) % int(save_every_batches) == 0:
                _atomic_jsonl(cache_path, _ordered_cache(cache, example_ids))
    finally:
        tokenizer.padding_side = previous_padding
    _atomic_jsonl(cache_path, _ordered_cache(cache, example_ids))
    if set(cache) != set(example_ids):
        raise RuntimeError("Sharer generation ended with an incomplete cache")
    return cache


def _collate_receiver(
    prompts: Sequence[Sequence[int]], *, pad_token_id: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(len(row) for row in prompts)
    ids = torch.tensor(
        [[pad_token_id] * (width - len(row)) + list(row) for row in prompts],
        dtype=torch.long,
        device=device,
    )
    mask = torch.tensor(
        [[0] * (width - len(row)) + [1] * len(row) for row in prompts],
        dtype=torch.long,
        device=device,
    )
    return ids, mask


def _measure_latency_ms(
    run_fn: Callable[[], Any], device: torch.device
) -> tuple[Any, float]:
    """Measure one inference call using the same timing semantics as the option-logit evaluator."""
    use_cuda_events = (
        isinstance(device, torch.device)
        and device.type == "cuda"
        and torch.cuda.is_available()
    )
    if use_cuda_events:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start_event.record()
        result = run_fn()
        end_event.record()
        torch.cuda.synchronize()
        return result, float(start_event.elapsed_time(end_event))

    started = time.perf_counter()
    result = run_fn()
    return result, float((time.perf_counter() - started) * 1000.0)


def _latency_statistics(values: Sequence[float]) -> Dict[str, float]:
    """Return aggregate latency statistics in milliseconds."""
    if not values:
        raise ValueError("cannot summarize an empty latency sequence")
    ordered = np.asarray([float(value) for value in values], dtype=np.float64)
    return {
        "count": int(ordered.size),
        "mean_ms": float(np.mean(ordered)),
        "median_ms": float(np.percentile(ordered, 50)),
        "p95_ms": float(np.percentile(ordered, 95)),
        "p99_ms": float(np.percentile(ordered, 99)),
        "min_ms": float(np.min(ordered)),
        "max_ms": float(np.max(ordered)),
        "total_model_ms": float(np.sum(ordered)),
    }


def _packet(
    model: Any,
    rows: Sequence[Mapping[str, Any]],
    *,
    pad_token_id: int,
    width: Optional[int] = None,
) -> Any:
    sequences: list[list[int]] = []
    draft_masks: list[list[int]] = []
    for row in rows:
        prompt = [int(value) for value in row["prompt_token_ids"]]
        draft = [int(value) for value in row["draft_token_ids"]]
        if not prompt or not draft:
            raise RuntimeError("cached packet has an empty prompt or response")
        sequences.append(prompt + draft)
        draft_masks.append([0] * len(prompt) + [1] * len(draft))
    observed_width = max(len(row) for row in sequences)
    width = observed_width if width is None else int(width)
    if width < observed_width:
        raise ValueError("requested packet width is shorter than a real sequence")
    ids = torch.tensor(
        [row + [pad_token_id] * (width - len(row)) for row in sequences],
        dtype=torch.long,
        device=model.device,
    )
    attention = torch.tensor(
        [[1] * len(row) + [0] * (width - len(row)) for row in sequences],
        dtype=torch.long,
        device=model.device,
    )
    draft_mask = torch.tensor(
        [row + [0] * (width - len(row)) for row in draft_masks],
        dtype=torch.long,
        device=model.device,
    )
    return model.make_packet(ids, attention, draft_mask, detach=True)


def _option_token_ids(tokenizer: Any) -> list[int]:
    # Identical convention to the standard draft_kv.utils.evaluate.get_option_token_ids.
    result = []
    for index in range(10):
        encoded = tokenizer.encode(
            " " + chr(ord("A") + index), add_special_tokens=False
        )
        if not encoded:
            raise RuntimeError("Receiver tokenizer cannot encode an option letter")
        result.append(int(encoded[0]))
    if len(result) != len(set(result)):
        raise RuntimeError(
            "space-prefixed option letters do not have unique first tokens for this Receiver"
        )
    return result


def _predictions(
    logits: torch.Tensor,
    option_token_ids: Sequence[int],
    option_counts: Sequence[int],
) -> tuple[list[str], list[float]]:
    candidate = logits[:, -1, list(option_token_ids)].float()
    labels: list[str] = []
    goldless_confidence: list[float] = []
    for index, count in enumerate(option_counts):
        values = candidate[index, : int(count)]
        probabilities = torch.softmax(values, dim=-1)
        prediction = int(values.argmax().item())
        labels.append(chr(ord("A") + prediction))
        goldless_confidence.append(float(probabilities[prediction].item()))
    return labels, goldless_confidence


def _row_order(
    rows: Mapping[tuple[str, str], Mapping[str, Any]],
    example_ids: Sequence[str],
    conditions: Sequence[str] = CONDITIONS,
) -> list[Mapping[str, Any]]:
    return [
        rows[(condition, example_id)]
        for example_id in example_ids
        for condition in conditions
        if (condition, example_id) in rows
    ]


def _existing_eval(path: Path) -> Dict[tuple[str, str], Dict[str, Any]]:
    rows = _read_jsonl(path) if path.exists() else []
    result: Dict[tuple[str, str], Dict[str, Any]] = {}
    for row in rows:
        key = (str(row["condition"]), str(row["example_id"]))
        if key in result:
            raise RuntimeError(f"duplicate evaluation row {key}")
        result[key] = row
    return result


@torch.no_grad()
def _evaluate(
    model: Any,
    receiver_tokenizer: Any,
    sharer_tokenizer: Any,
    examples: Sequence[Mapping[str, Any]],
    cache: Mapping[str, Mapping[str, Any]],
    donors: Mapping[str, str],
    output_path: Path,
    *,
    batch_size: int,
    max_receiver_input_tokens: int,
    save_every_batches: int,
    prompt_style: str,
    conditions: Sequence[str],
) -> tuple[Dict[tuple[str, str], Dict[str, Any]], Dict[str, float]]:
    rows = _existing_eval(output_path)
    valid_ids = {str(row["example_id"]) for row in examples}
    if any(
        condition not in conditions or example_id not in valid_ids
        for condition, example_id in rows
    ):
        raise RuntimeError("existing evaluation rows are outside this manifest")
    example_ids = [str(row["example_id"]) for row in examples]
    by_id = {str(row["example_id"]): row for row in examples}
    for (condition, example_id), row in rows.items():
        expected_donor = (
            None
            if condition == "Zero"
            else example_id if condition == "Matched" else str(donors[example_id])
        )
        if (
            row.get("donor_example_id") != expected_donor
            or str(row.get("gold")) != str(by_id[example_id]["answer_label"])
            or str(row.get("dataset")) != str(by_id[example_id]["dataset"])
            or str(row.get("subject")) != str(by_id[example_id]["subject"])
            or str(row.get("prediction"))
            not in {
                chr(ord("A") + index)
                for index in range(len(by_id[example_id]["choices"]))
            }
            or bool(row.get("correct"))
            != (str(row.get("prediction")) == str(row.get("gold")))
        ):
            raise RuntimeError(
                f"existing evaluation row fails replay validation: {(condition, example_id)}"
            )
    completed_ids = [
        example_id
        for example_id in example_ids
        if all((condition, example_id) in rows for condition in conditions)
    ]
    partial_ids = [
        example_id
        for example_id in example_ids
        if any((condition, example_id) in rows for condition in conditions)
        and not all((condition, example_id) in rows for condition in conditions)
    ]
    if partial_ids or completed_ids != example_ids[: len(completed_ids)]:
        raise RuntimeError(
            "existing evaluation output is not a complete ordered prefix"
        )
    if len(completed_ids) != len(example_ids) and len(completed_ids) % int(batch_size):
        raise RuntimeError(
            "incomplete evaluation output ends inside a fixed target batch"
        )
    option_ids = _option_token_ids(receiver_tokenizer)
    parity: Dict[str, float] = {}
    parity_checked = "Zero" not in conditions
    for batch_index, start in enumerate(range(0, len(examples), int(batch_size))):
        chunk = examples[start : start + int(batch_size)]
        chunk_ids = [str(row["example_id"]) for row in chunk]
        # Recompute the first batch after a resume so the result always carries
        # an observed native/split parity value.  Later complete batches skip.
        if parity_checked and all(
            (condition, example_id) in rows
            for example_id in chunk_ids
            for condition in conditions
        ):
            continue
        prompts = [
            _chat_ids(
                receiver_tokenizer,
                _evaluation_prompt(
                    row,
                    prompt_style=prompt_style,
                    use_cot=False,
                ),
                assistant_prefix=PLAIN_ANSWER_PREFIX,
            )
            for row in chunk
        ]
        longest = max(len(values) for values in prompts)
        if longest > int(max_receiver_input_tokens):
            raise RuntimeError(
                f"Receiver prompt exceeds --max-receiver-input-tokens: {longest}"
            )
        ids, attention = _collate_receiver(
            prompts,
            pad_token_id=int(receiver_tokenizer.pad_token_id),
            device=model.device,
        )
        zero_logits = None
        if "Zero" in conditions:
            if not parity:
                native = model.receiver(
                    input_ids=ids, attention_mask=attention, use_cache=False
                ).logits
            zero_logits, _, _ = model.forward_receiver(
                ids,
                attention,
                packet=None,
                disable_communication=True,
                use_cache=False,
            )
            if not parity:
                parity = {
                    "native_vs_zero_max_abs": float(
                        (native - zero_logits).abs().max().item()
                    )
                }
                if parity["native_vs_zero_max_abs"] > 1e-5:
                    raise RuntimeError(f"native/split parity failed: {parity}")
                parity_checked = True
        matched_rows = [cache[example_id] for example_id in chunk_ids]
        deranged_ids = [str(donors[example_id]) for example_id in chunk_ids]
        deranged_rows = [cache[example_id] for example_id in deranged_ids]
        packet_rows = []
        if "Matched" in conditions:
            packet_rows.extend(matched_rows)
        if "Deranged" in conditions:
            packet_rows.extend(deranged_rows)
        packet_width = (
            max(
                len(row["prompt_token_ids"]) + len(row["draft_token_ids"])
                for row in packet_rows
            )
            if packet_rows
            else 0
        )
        matched_logits = None
        if "Matched" in conditions:
            matched_packet = _packet(
                model,
                matched_rows,
                pad_token_id=int(sharer_tokenizer.pad_token_id),
                width=packet_width,
            )
            matched_logits, _, _ = model.forward_receiver(
                ids,
                attention,
                packet=matched_packet,
                disable_communication=False,
                use_cache=False,
            )
        deranged_logits = None
        if "Deranged" in conditions:
            deranged_packet = _packet(
                model,
                deranged_rows,
                pad_token_id=int(sharer_tokenizer.pad_token_id),
                width=packet_width,
            )
            deranged_logits, _, _ = model.forward_receiver(
                ids,
                attention,
                packet=deranged_packet,
                disable_communication=False,
                use_cache=False,
            )
        counts = [len(row["choices"]) for row in chunk]
        logits_by_condition = {
            "Zero": (zero_logits, [None] * len(chunk)),
            "Matched": (matched_logits, chunk_ids),
            "Deranged": (deranged_logits, deranged_ids),
        }
        for condition in conditions:
            logits, donor_ids = logits_by_condition[condition]
            predictions, confidence = _predictions(logits, option_ids, counts)
            for index, example in enumerate(chunk):
                example_id = chunk_ids[index]
                prediction = predictions[index]
                rows[(condition, example_id)] = {
                    "example_id": example_id,
                    "dataset": str(example["dataset"]),
                    "subject": str(example["subject"]),
                    "condition": condition,
                    "donor_example_id": donor_ids[index],
                    "prediction": prediction,
                    "gold": str(example["answer_label"]),
                    "correct": prediction == str(example["answer_label"]),
                    "predicted_option_probability": confidence[index],
                    "packet_common_width": int(packet_width),
                    "packet_length": (
                        None
                        if donor_ids[index] is None
                        else len(cache[str(donor_ids[index])]["prompt_token_ids"])
                        + len(cache[str(donor_ids[index])]["draft_token_ids"])
                    ),
                }
        completed = sum(
            all((condition, example_id) in rows for condition in conditions)
            for example_id in example_ids
        )
        print(f"Receiver evaluation {completed}/{len(examples)}", flush=True)
        if (batch_index + 1) % int(save_every_batches) == 0:
            _atomic_jsonl(output_path, _row_order(rows, example_ids, conditions))
    _atomic_jsonl(output_path, _row_order(rows, example_ids, conditions))
    if any(
        (condition, example_id) not in rows
        for example_id in example_ids
        for condition in conditions
    ):
        raise RuntimeError("Receiver evaluation ended with incomplete rows")
    return rows, parity


def _existing_latency(path: Path) -> Dict[str, Dict[str, Any]]:
    """Load resumable Matched-only latency rows."""
    rows = _read_jsonl(path) if path.exists() else []
    result: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        example_id = str(row.get("example_id", ""))
        if not example_id:
            raise RuntimeError("latency output contains a row without example_id")
        if example_id in result:
            raise RuntimeError(f"duplicate latency row {example_id}")
        if row.get("condition") != "Matched":
            raise RuntimeError("latency output may contain Matched rows only")
        value = row.get("latency_ms")
        if value is None or not np.isfinite(float(value)) or float(value) < 0:
            raise RuntimeError(f"invalid latency value for {example_id}")
        result[example_id] = row
    return result


def _ordered_latency(
    rows: Mapping[str, Mapping[str, Any]], example_ids: Sequence[str]
) -> list[Mapping[str, Any]]:
    return [rows[example_id] for example_id in example_ids if example_id in rows]


@torch.no_grad()
def _evaluate_matched_latency(
    model: Any,
    receiver_tokenizer: Any,
    sharer_tokenizer: Any,
    examples: Sequence[Mapping[str, Any]],
    cache: Mapping[str, Mapping[str, Any]],
    output_path: Path,
    *,
    max_receiver_input_tokens: int,
    save_every_examples: int,
    prompt_style: str,
) -> tuple[Dict[str, Dict[str, Any]], Dict[str, float]]:
    """Measure one Matched answer per question, without Zero/Deranged passes.

    The timed closure contains packet construction from the already-frozen
    Sharer cache and the Receiver forward.  This is the Draft-KV analogue of
    the standard ``logits_with_context`` timer; it intentionally reports a cached
    Sharer-draft latency and does not include the one-time cache generation.
    Each question is evaluated separately so the reported mean is a true
    per-question latency rather than a batch time divided by batch size.
    """
    rows = _existing_latency(output_path)
    example_ids = [str(row["example_id"]) for row in examples]
    by_id = {str(row["example_id"]): row for row in examples}
    if any(example_id not in by_id for example_id in rows):
        raise RuntimeError("existing latency rows are outside this manifest")
    for example_id, row in rows.items():
        example = by_id[example_id]
        if (
            str(row.get("dataset")) != str(example["dataset"])
            or str(row.get("subject")) != str(example["subject"])
            or str(row.get("gold")) != str(example["answer_label"])
        ):
            raise RuntimeError(
                f"existing latency row fails replay validation: {example_id}"
            )
    completed = [example_id for example_id in example_ids if example_id in rows]
    if completed != example_ids[: len(completed)]:
        raise RuntimeError("existing latency output is not an ordered prefix")

    option_ids = _option_token_ids(receiver_tokenizer)
    latencies = [float(rows[example_id]["latency_ms"]) for example_id in completed]
    for index, example in enumerate(examples):
        example_id = str(example["example_id"])
        if example_id in rows:
            continue
        prompt_ids = _chat_ids(
            receiver_tokenizer,
            _evaluation_prompt(
                example,
                prompt_style=prompt_style,
                use_cot=False,
            ),
            assistant_prefix=PLAIN_ANSWER_PREFIX,
        )
        if len(prompt_ids) > int(max_receiver_input_tokens):
            raise RuntimeError(
                f"Receiver prompt exceeds --max-receiver-input-tokens: "
                f"{example_id} length={len(prompt_ids)}"
            )
        cache_row = cache[example_id]
        packet_width = len(cache_row["prompt_token_ids"]) + len(
            cache_row["draft_token_ids"]
        )

        def _matched_call() -> Any:
            # Keep all GPU work in the timed closure.  CUDA event timing then
            # matches the standard synchronization/event implementation exactly;
            # CPU fallback also includes tensor/packet preparation.
            ids, attention = _collate_receiver(
                [prompt_ids],
                pad_token_id=int(receiver_tokenizer.pad_token_id),
                device=model.device,
            )
            packet = _packet(
                model,
                [cache_row],
                pad_token_id=int(sharer_tokenizer.pad_token_id),
                width=packet_width,
            )
            return model.forward_receiver(
                ids,
                attention,
                packet=packet,
                disable_communication=False,
                use_cache=False,
            )

        (logits, _, _), latency_ms = _measure_latency_ms(_matched_call, model.device)
        predictions, confidence = _predictions(
            logits, option_ids, [len(example["choices"])]
        )
        prediction = predictions[0]
        rows[example_id] = {
            "example_id": example_id,
            "dataset": str(example["dataset"]),
            "subject": str(example["subject"]),
            "condition": "Matched",
            "prediction": prediction,
            "gold": str(example["answer_label"]),
            "correct": prediction == str(example["answer_label"]),
            "predicted_option_probability": confidence[0],
            "latency_ms": float(latency_ms),
            "latency_scope": "packet_build_plus_receiver_forward_cached_sharer_draft",
            "packet_length": int(packet_width),
        }
        latencies.append(float(latency_ms))
        print(f"Matched latency {index + 1}/{len(examples)}", flush=True)
        if (index + 1) % int(save_every_examples) == 0:
            _atomic_jsonl(output_path, _ordered_latency(rows, example_ids))

    _atomic_jsonl(output_path, _ordered_latency(rows, example_ids))
    if len(rows) != len(example_ids):
        raise RuntimeError("Matched latency evaluation ended with incomplete rows")
    return rows, _latency_statistics(latencies)


def _paired(
    rows: Mapping[tuple[str, str], Mapping[str, Any]],
    example_ids: Sequence[str],
    left: str,
    right: str,
    *,
    samples: int,
    seed: int,
) -> Dict[str, float]:
    return bootstrap_ci(
        [
            float(rows[(left, example_id)]["correct"])
            - float(rows[(right, example_id)]["correct"])
            for example_id in example_ids
        ],
        samples=int(samples),
        seed=int(seed),
    )


def _summary(
    examples: Sequence[Mapping[str, Any]],
    cache: Mapping[str, Mapping[str, Any]],
    rows: Mapping[tuple[str, str], Mapping[str, Any]],
    parity: Mapping[str, float],
    *,
    bootstrap_samples: int,
    seed: int,
    conditions: Sequence[str] = CONDITIONS,
) -> Dict[str, Any]:
    example_ids = [str(row["example_id"]) for row in examples]
    by_subject: Dict[str, list[str]] = defaultdict(list)
    for row in examples:
        by_subject[str(row["subject"])].append(str(row["example_id"]))

    def accuracy(ids: Sequence[str], condition: str) -> float:
        return float(
            np.mean([bool(rows[(condition, value)]["correct"]) for value in ids])
        )

    correct_count = {
        condition: int(
            sum(bool(rows[(condition, value)]["correct"]) for value in example_ids)
        )
        for condition in conditions
    }
    sharer_correct = 0
    sharer_parseable = 0
    for example in examples:
        example_id = str(example["example_id"])
        choice = extract_choice(
            str(cache[example_id]["response"]),
            option_count=len(example["choices"]),
            choices=example["choices"],
        )
        sharer_parseable += choice is not None
        sharer_correct += choice == str(example["answer_label"])
    result = {
        "protocol": PROTOCOL,
        "count": len(example_ids),
        "accuracy": {
            condition: correct_count[condition] / len(example_ids)
            for condition in conditions
        },
        "correct_count": correct_count,
        "paired_bootstrap": {},
        "accuracy_by_subject": {
            subject: {
                "count": len(ids),
                **{condition: accuracy(ids, condition) for condition in conditions},
            }
            for subject, ids in sorted(by_subject.items())
        },
        "matched_new_correct": None,
        "matched_damaged_zero_correct": None,
        "donor_sensitive_count": None,
        "sharer_response": {
            "parseable_count": int(sharer_parseable),
            "correct_count": int(sharer_correct),
            "accuracy": float(sharer_correct / len(example_ids)),
            "truncated_count": int(
                sum(
                    not bool(cache[value].get("terminated_eos"))
                    for value in example_ids
                )
            ),
        },
        "parity": dict(parity),
    }
    if "Matched" in conditions and "Zero" in conditions:
        result["paired_bootstrap"]["Matched_minus_Zero"] = _paired(
            rows, example_ids, "Matched", "Zero",
            samples=bootstrap_samples, seed=seed,
        )
        result["matched_new_correct"] = int(sum(
            bool(rows[("Matched", value)]["correct"])
            and not bool(rows[("Zero", value)]["correct"])
            for value in example_ids
        ))
        result["matched_damaged_zero_correct"] = int(sum(
            bool(rows[("Zero", value)]["correct"])
            and not bool(rows[("Matched", value)]["correct"])
            for value in example_ids
        ))
    if "Matched" in conditions and "Deranged" in conditions:
        result["paired_bootstrap"]["Matched_minus_Deranged"] = _paired(
            rows, example_ids, "Matched", "Deranged",
            samples=bootstrap_samples, seed=seed + 1,
        )
        result["donor_sensitive_count"] = int(sum(
            rows[("Matched", value)]["prediction"]
            != rows[("Deranged", value)]["prediction"]
            for value in example_ids
        ))
    return result


def _ensure_manifest(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        observed = json.loads(path.read_text(encoding="utf-8"))
        if observed != dict(value):
            raise RuntimeError(
                f"existing manifest differs from this command; choose a new output: {path}"
            )
    else:
        _atomic_json(path, value)


def _ensure_cache_manifest(path: Path, value: Mapping[str, Any]) -> None:
    """Accept exact manifests and older cache-only manifest schemas.

    Cache rows are independently replay-validated by ``_generate_sharer_cache``
    (including prompt token IDs and decoded responses).  This allows a cache
    produced before non-semantic evaluator fields such as split/prompt-role
    metadata or the runner hash were added to be reused safely, while still
    rejecting changes to the dataset, example selection, sharer artifacts, or
    generation configuration.

    Sharer caches are receiver-agnostic: ``_generate_sharer_cache`` only uses
    the sharer model/tokenizer plus dataset and generation settings, so
    receiver-side identity fields (``receiver``, ``receiver_artifact_sha256``,
    ``model_pair``, ``layer_mapping``) are deliberately excluded from the
    immutable set.  A cache generated for one receiver (e.g. the 8B->0.5B
    pair) can therefore be reused for any other receiver paired with the same
    sharer (e.g. 8B->4B) via ``--draft-cache-dir``; the receiving run's own
    ``eval_manifest.json`` still records its true receiver identity.
    """
    if not path.exists():
        _atomic_json(path, value)
        return
    observed = json.loads(path.read_text(encoding="utf-8"))
    if observed == dict(value):
        return

    # Receiver-side identity (receiver, receiver_artifact_sha256, model_pair,
    # layer_mapping) is intentionally absent: sharer cache rows depend only on
    # the sharer side, so a completed cache may be shared across receivers
    # paired with the same sharer.
    immutable_keys = (
        "protocol",
        "dataset",
        "data_path",
        "data_sha256",
        "data_registration",
        "example_count",
        "example_ids",
        "example_ids_file",
        "selection_sha256",
        "subjects",
        "selected_subjects",
        "sharer",
        "sharer_artifact_sha256",
        "sharer_use_cot",
        "max_sharer_input_tokens",
        "draft_max_new_tokens",
        "draft_batch_size",
    )
    mismatches = [
        key for key in immutable_keys if observed.get(key) != value.get(key)
    ]
    observed_configuration = observed.get("configuration", {})
    expected_configuration = value.get("configuration", {})
    for key in (
        "conditions",
        "derangement_seed",
        "draft_batch_size",
        "draft_max_new_tokens",
        "max_receiver_input_tokens",
        "max_sharer_input_tokens",
        "prompt_style",
    ):
        if observed_configuration.get(key) != expected_configuration.get(key):
            mismatches.append(f"configuration.{key}")
    if mismatches:
        raise RuntimeError(
            "existing cache manifest is incompatible with this command: "
            f"{path} (mismatched {', '.join(mismatches)})"
        )
    print(
        f"Reusing cache manifest with compatible generation settings: {path}",
        flush=True,
    )


def _resolve_pair(args: argparse.Namespace) -> tuple[str, str, str, str]:
    if args.model_pair == "current":
        if args.receiver or args.sharer or args.layer_mapping:
            raise ValueError(
                "--model-pair current uses the registered defaults; use custom to override models"
            )
        return (
            DEFAULT_RECEIVER,
            DEFAULT_SHARER,
            DEFAULT_MAPPING,
            DEFAULT_PAIR,
        )
    if args.model_pair != "custom":
        raise ValueError("--model-pair must be current or custom")
    if not args.receiver or not args.sharer or not args.layer_mapping:
        raise ValueError(
            "custom model pair requires --receiver, --sharer, and --layer-mapping"
        )
    fingerprint = hashlib.sha256(
        f"{Path(args.receiver).resolve()}\0{Path(args.sharer).resolve()}\0{args.layer_mapping}".encode(
            "utf-8"
        )
    ).hexdigest()[:10]
    pair_name = (
        f"{_safe_tag(Path(args.sharer).name)}_to_"
        f"{_safe_tag(Path(args.receiver).name)}_{fingerprint}"
    )
    return args.receiver, args.sharer, args.layer_mapping, pair_name


def run(args: argparse.Namespace) -> Dict[str, Any]:
    dataset = canonical_dataset_name(args.dataset)
    split = str(args.split or _default_split(dataset)).strip().lower()
    if dataset == "ceval" and split != "val":
        raise ValueError("C-EVAL local accuracy evaluation requires the labeled val split")
    if dataset != "ceval" and split != "test":
        raise ValueError(f"{dataset} evaluation currently requires its test split")
    receiver, sharer, mapping_string, pair_name = _resolve_pair(args)
    mapping = parse_layer_mapping(mapping_string)
    sharer_prompt_style = str(args.sharer_prompt_style or args.prompt_style)
    receiver_prompt_style = str(args.receiver_prompt_style or args.prompt_style)
    checkpoint_path = Path(args.checkpoint).resolve()
    data_root = Path(args.data_root).resolve()
    data_path = _data_path(data_root, dataset, split)
    for label, path in (("dataset", data_path), ("checkpoint", checkpoint_path)):
        if not path.is_file():
            raise FileNotFoundError(f"missing {label}: {path}")
    for label, path in (("Receiver", Path(receiver)), ("Sharer", Path(sharer))):
        if not path.is_dir():
            raise FileNotFoundError(f"missing {label} model directory: {path}")
    all_examples = _read_jsonl(data_path)
    for row in all_examples:
        validate_normalized_row(row)
        if row["dataset"] != dataset:
            raise RuntimeError("adapted data contains the wrong dataset")
    all_example_ids = [str(row["example_id"]) for row in all_examples]
    if len(all_example_ids) != len(set(all_example_ids)):
        raise RuntimeError("adapted data contains duplicate IDs")
    data_registration = _dataset_manifest_entry(data_root, dataset, data_path)
    if int(data_registration["adapted_file"]["count"]) != len(all_examples):
        raise RuntimeError(
            "adapted dataset row count differs from its download manifest"
        )
    data_sha = sha256_file(data_path)
    example_ids_binding = None
    example_ids_file = getattr(args, "example_ids_file", None)
    if example_ids_file:
        if args.subjects or args.max_examples is not None:
            raise ValueError(
                "--example-ids-file cannot be combined with --subjects or "
                "--max-examples"
            )
        ids_path = Path(example_ids_file).resolve()
        requested_ids, ids_metadata = _load_example_ids_file(ids_path)
        examples = _select_examples_by_id(all_examples, requested_ids)
        example_ids_binding = _example_ids_file_binding(
            ids_path,
            ids_metadata,
            requested_ids,
            dataset=dataset,
            data_path=data_path,
            data_sha256=data_sha,
            data_registration=data_registration,
        )
    else:
        examples = list(all_examples)
    subjects = {
        value.strip() for value in str(args.subjects or "").split(",") if value.strip()
    }
    if subjects:
        examples = [row for row in examples if str(row["subject"]) in subjects]
        missing = subjects - {str(row["subject"]) for row in examples}
        if missing:
            raise ValueError(f"unknown/empty subjects: {sorted(missing)}")
    if args.max_examples is not None:
        examples = examples[: int(args.max_examples)]
    if len(examples) < 2:
        raise RuntimeError("evaluation requires at least two examples")
    example_ids = [str(row["example_id"]) for row in examples]
    if len(example_ids) != len(set(example_ids)):
        raise RuntimeError("adapted data contains duplicate IDs")
    checkpoint_sha = sha256_file(checkpoint_path)
    selection_sha = _selection_sha256(example_ids)
    model_tag = pair_name.replace("/", "_")
    latency_only = bool(getattr(args, "latency_only", False))
    eval_conditions = _normalize_conditions(args.conditions, latency_only=latency_only)
    configuration = {
        "split": split,
        "max_sharer_input_tokens": int(args.max_sharer_input_tokens),
        "draft_max_new_tokens": int(args.draft_max_new_tokens),
        "draft_batch_size": int(args.draft_batch_size),
        "max_receiver_input_tokens": int(args.max_receiver_input_tokens),
        "eval_batch_size": int(args.eval_batch_size),
        "derangement_seed": int(args.derangement_seed),
        "conditions": list(eval_conditions),
        "choice_parser": "markdown_and_unique_option_text",
        "prompt_style": str(args.prompt_style),
        "sharer_prompt_style": sharer_prompt_style,
        "receiver_prompt_style": receiver_prompt_style,
        "plain_prompt_option_logits": (
            "plain no-CoT prompt body and response-text suffix"
            if args.prompt_style == "plain"
            else None
        ),
    }
    if latency_only:
        configuration["latency_only"] = True
    configuration_sha = _configuration_sha256(configuration)
    run_tag = f"{selection_sha[:12]}-{configuration_sha[:12]}"
    cache_configuration = dict(configuration)
    cache_configuration["conditions"] = list(CONDITIONS)
    cache_configuration.pop("latency_only", None)
    # Receiver evaluation batching does not affect Sharer generation.  Keep it
    # out of the cache identity so a completed Sharer cache can be reused after
    # lowering --eval-batch-size to avoid receiver-side OOM.
    cache_configuration.pop("eval_batch_size", None)
    cache_configuration_sha = _configuration_sha256(cache_configuration)
    cache_run_tag = f"{selection_sha[:12]}-{cache_configuration_sha[:12]}"
    output = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else (
            Path(args.output_root).resolve()
            / "evaluations"
            / dataset
            / model_tag
            / checkpoint_sha[:12]
            / run_tag
        )
    )
    cache_dir = (
        Path(args.draft_cache_dir).resolve()
        if args.draft_cache_dir
        else (
            Path(args.output_root).resolve()
            / "sharer_cache"
            / dataset
            / model_tag
            / cache_run_tag
        )
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    checkpoint_mapping = checkpoint.get("layer_mapping")
    if (
        checkpoint_mapping is not None
        and {int(key): int(value) for key, value in checkpoint_mapping.items()}
        != mapping
    ):
        raise RuntimeError("checkpoint layer mapping differs from --layer-mapping")
    for key, requested in (("receiver", receiver), ("sharer", sharer)):
        registered = checkpoint.get(key)
        if (
            registered is not None
            and Path(str(registered)).resolve() != Path(requested).resolve()
        ):
            raise RuntimeError(f"checkpoint {key} differs from the selected model pair")
    communication_state = checkpoint.get("communication_state", checkpoint)
    if not communication_state.get("projection") or not communication_state.get(
        "gate_logits"
    ):
        raise RuntimeError("checkpoint has no complete communication state")
    receiver_artifacts = _model_artifact_hashes(receiver)
    sharer_artifacts = _model_artifact_hashes(sharer)
    donors = (
        {}
        if latency_only
        else make_no_fixed_point_id_mapping(
            example_ids, seed=int(args.derangement_seed)
        )
    )
    sharer_preflight_tokenizer = AutoTokenizer.from_pretrained(
        sharer,
        local_files_only=True,
        trust_remote_code=True,
    )
    receiver_preflight_tokenizer = AutoTokenizer.from_pretrained(
        receiver,
        local_files_only=True,
        trust_remote_code=True,
    )
    prompt_length_audit = {
        "sharer": _prompt_length_audit(
            sharer_preflight_tokenizer,
            examples,
            use_cot=sharer_prompt_style == "reasoning",
            assistant_prefix=_sharer_assistant_prefix(sharer_prompt_style),
            limit=args.max_sharer_input_tokens,
            role="Sharer",
            prompt_style=sharer_prompt_style,
        ),
        "receiver": _prompt_length_audit(
            receiver_preflight_tokenizer,
            examples,
            use_cot=False,
            assistant_prefix="The correct answer is",
            limit=args.max_receiver_input_tokens,
            role="Receiver",
            prompt_style=receiver_prompt_style,
        ),
    }
    del sharer_preflight_tokenizer
    del receiver_preflight_tokenizer
    common = {
        "protocol": PROTOCOL,
        "dataset": dataset,
        "data_path": str(data_path),
        "data_sha256": data_sha,
        "data_registration": data_registration,
        "example_count": len(examples),
        "example_ids": example_ids,
        "example_ids_file": example_ids_binding,
        "selection_sha256": selection_sha,
        "configuration": configuration,
        "configuration_sha256": configuration_sha,
        "prompt_length_audit": prompt_length_audit,
        "subjects": sorted(subjects),
        "selected_subjects": sorted({str(row["subject"]) for row in examples}),
        "receiver": str(Path(receiver).resolve()),
        "sharer": str(Path(sharer).resolve()),
        "model_pair": pair_name,
        "receiver_artifact_sha256": receiver_artifacts,
        "sharer_artifact_sha256": sharer_artifacts,
        "layer_mapping": {str(key): int(value) for key, value in mapping.items()},
        "sharer_use_cot": sharer_prompt_style == "reasoning",
        "sharer_prompt_style": sharer_prompt_style,
        "receiver_prompt_style": receiver_prompt_style,
        "max_sharer_input_tokens": int(args.max_sharer_input_tokens),
        "draft_max_new_tokens": int(args.draft_max_new_tokens),
        "runtime_versions": {
            "python": sys.version.split()[0],
            "torch": str(torch.__version__),
        },
        "code_sha256": {
            "runner": sha256_file(Path(__file__)),
            "adapter": sha256_file(
                Path(__file__).resolve().parents[2]
                / "draft_kv/train/downstream_mc_data.py"
            ),
        },
    }
    cache_common = {
        **common,
        "configuration": cache_configuration,
        "configuration_sha256": cache_configuration_sha,
    }
    cache_manifest = {
        **cache_common,
        "draft_batch_size": int(args.draft_batch_size),
    }
    eval_manifest = {
        **common,
        "protocol": LATENCY_PROTOCOL if latency_only else PROTOCOL,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha,
        "conditions": list(eval_conditions),
        "eval_batch_size": int(args.eval_batch_size),
        "max_receiver_input_tokens": int(args.max_receiver_input_tokens),
        "bootstrap_samples": int(args.bootstrap_samples),
        "seed": int(args.seed),
        "option_scoring": "first valid space-prefixed A-J token",
    }
    if not latency_only:
        eval_manifest.update(
            {
                "derangement_seed": int(args.derangement_seed),
                "donor_mapping": donors,
                "matched_deranged_share_packet_tensor_width_per_batch": True,
            }
        )
    output.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    _ensure_manifest(output / "eval_manifest.json", eval_manifest)
    _ensure_cache_manifest(cache_dir / "cache_manifest.json", cache_manifest)
    seed_all(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    model, receiver_tokenizer, sharer_tokenizer = build_model(
        receiver_path=receiver,
        sharer_path=sharer,
        layer_mapping=mapping,
        device_name=args.device,
    )
    load_trainable_state(model, checkpoint)
    model.set_stage("eval")
    if model.trainable_parameter_names():
        raise RuntimeError("evaluation model still has trainable parameters")
    cache = _generate_sharer_cache(
        model,
        sharer_tokenizer,
        examples,
        cache_dir / "sharer_drafts.jsonl",
        batch_size=args.draft_batch_size,
        max_input_tokens=args.max_sharer_input_tokens,
        max_new_tokens=args.draft_max_new_tokens,
        save_every_batches=args.save_every_batches,
        prompt_style=sharer_prompt_style,
    )
    if latency_only:
        latency_rows, latency_summary = _evaluate_matched_latency(
            model,
            receiver_tokenizer,
            sharer_tokenizer,
            examples,
            cache,
            output / "per_example_latency.jsonl",
            max_receiver_input_tokens=args.max_receiver_input_tokens,
            save_every_examples=args.save_every_batches,
            prompt_style=receiver_prompt_style,
        )
        result = {
            "protocol": LATENCY_PROTOCOL,
            "count": len(example_ids),
            "condition": "Matched",
            "timing_backend": (
                "cuda_events"
                if device.type == "cuda" and torch.cuda.is_available()
                else "perf_counter"
            ),
            "latency_scope": "packet_build_plus_receiver_forward_cached_sharer_draft",
            "latency_statistics": latency_summary,
            "accuracy": float(
                np.mean(
                    [
                        bool(latency_rows[example_id]["correct"])
                        for example_id in example_ids
                    ]
                )
            ),
            "correct_count": int(
                sum(
                    bool(latency_rows[example_id]["correct"])
                    for example_id in example_ids
                )
            ),
        }
    else:
        rows, parity = _evaluate(
            model,
            receiver_tokenizer,
            sharer_tokenizer,
            examples,
            cache,
            donors,
            output / "per_example.jsonl",
            batch_size=args.eval_batch_size,
            max_receiver_input_tokens=args.max_receiver_input_tokens,
            save_every_batches=args.save_every_batches,
            prompt_style=receiver_prompt_style,
            conditions=eval_conditions,
        )
        result = _summary(
            examples,
            cache,
            rows,
            parity,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
            conditions=eval_conditions,
        )
    checkpoint_note = None
    if checkpoint_path == Path(DEFAULT_CHECKPOINT).resolve():
        checkpoint_note = (
            "Update 2500 had the lowest recorded matched validation NLL "
            "(1.0061806518), but its weights were not persisted. The default "
            "is the only retained formal Stage-2 checkpoint, update 4000."
        )
    result.update(
        {
            "conditions": list(eval_conditions),
            "latency_only": latency_only,
            "dataset": dataset,
            "split": split,
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_sha,
            "checkpoint_protocol": checkpoint.get("protocol"),
            "checkpoint_optimizer_update": checkpoint.get("optimizer_update"),
            "receiver": str(Path(receiver).resolve()),
            "sharer": str(Path(sharer).resolve()),
            "layer_mapping": {str(key): value for key, value in mapping.items()},
            "data_path": str(data_path),
            "data_sha256": data_sha,
            "example_ids_file": example_ids_binding,
            "configuration_sha256": configuration_sha,
            "prompt_length_audit": prompt_length_audit,
            "sharer_cache": str((cache_dir / "sharer_drafts.jsonl").resolve()),
            "output_dir": str(output),
            "eval_manifest_sha256": sha256_file(output / "eval_manifest.json"),
            "cache_manifest_sha256": sha256_file(cache_dir / "cache_manifest.json"),
            "checkpoint_note": checkpoint_note,
        }
    )
    result["output_artifact_sha256"] = {
        "eval_manifest.json": sha256_file(output / "eval_manifest.json"),
        "sharer_drafts.jsonl": sha256_file(cache_dir / "sharer_drafts.jsonl"),
    }
    if latency_only:
        result["output_artifact_sha256"]["per_example_latency.jsonl"] = sha256_file(
            output / "per_example_latency.jsonl"
        )
    else:
        result["output_artifact_sha256"]["per_example.jsonl"] = sha256_file(
            output / "per_example.jsonl"
        )
    _atomic_json(output / "eval_result.json", result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="mmlu-redux",
        help=(
            "mmlu-redux, arc-e/arc-easy, arc-c/arc-challenge, openbookqa, "
            "ceval, gsm-mc, or math-mc"
        ),
    )
    parser.add_argument(
        "--split",
        help="evaluation split (defaults to test; C-EVAL defaults to public labeled val)",
    )
    parser.add_argument(
        "--prompt-style",
        choices=PROMPT_STYLES,
        default="reasoning",
        help="reasoning enables CoT; plain uses the option-logit prompt",
    )
    parser.add_argument(
        "--sharer-prompt-style",
        choices=PROMPT_STYLES,
        help="override Sharer prompt style",
    )
    parser.add_argument(
        "--receiver-prompt-style",
        choices=PROMPT_STYLES,
        help="override Receiver prompt style; useful for the plain option-logit prompt",
    )
    parser.add_argument(
        "--model-pair", choices=("current", "custom"), default="current"
    )
    parser.add_argument("--receiver")
    parser.add_argument("--sharer")
    parser.add_argument("--layer-mapping")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-dir")
    parser.add_argument("--draft-cache-dir")
    parser.add_argument(
        "--latency-only",
        action="store_true",
        help=(
            "run only the Matched path with per-question timing; "
            "do not evaluate Zero or Deranged"
        ),
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=CONDITION_CHOICES,
        default=CONDITION_CHOICES,
        help=(
            "receiver conditions to evaluate; choose any of zero, matched, "
            "deranged (default: all). --latency-only overrides this to matched."
        ),
    )
    parser.add_argument(
        "--example-ids-file",
        help=(
            "ordered JSON/text ID selection; incompatible with --subjects and "
            "--max-examples and bound into both manifests"
        ),
    )
    parser.add_argument(
        "--subjects", help="comma-separated MMLU or C-EVAL subject filter"
    )
    parser.add_argument("--max-examples", type=int)
    parser.add_argument(
        "--draft-batch-size", type=int, default=DEFAULT_DRAFT_BATCH_SIZE
    )
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument(
        "--max-sharer-input-tokens",
        type=int,
        default=DEFAULT_MAX_SHARER_INPUT_TOKENS,
    )
    parser.add_argument("--draft-max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--max-receiver-input-tokens",
        type=int,
        default=DEFAULT_MAX_RECEIVER_INPUT_TOKENS,
    )
    parser.add_argument("--save-every-batches", type=int, default=10)
    parser.add_argument("--derangement-seed", type=int, default=71031)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=91827)
    parser.add_argument("--device", default="cuda:0")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    positive = (
        args.draft_batch_size,
        args.eval_batch_size,
        args.max_sharer_input_tokens,
        args.draft_max_new_tokens,
        args.max_receiver_input_tokens,
        args.save_every_batches,
        args.bootstrap_samples,
    )
    if any(int(value) <= 0 for value in positive):
        raise ValueError(
            "batch sizes, lengths, save interval, and bootstrap count must be positive"
        )
    if args.max_examples is not None and int(args.max_examples) < 2:
        raise ValueError("--max-examples must be at least 2")
    result = run(args)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
