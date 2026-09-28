"""OpenHermes data flow for Draft-KV.

Receiver and sharer tokenizers are called separately, and the answer is never
passed to the sharer path.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import torch
from torch.utils.data import Dataset


def _content(message: Any) -> str:
    if isinstance(message, Mapping):
        value = message.get("content", message.get("value", ""))
    else:
        value = getattr(message, "content", getattr(message, "value", ""))
    if value is None:
        return ""
    if isinstance(value, list):
        return "\n".join(str(item) for item in value)
    return str(value)


def _role(message: Any) -> str:
    if isinstance(message, Mapping):
        role = message.get("role")
        if role is None:
            role = message.get("from", "user")
        role = str(role).lower()
    else:
        role = str(getattr(message, "role", getattr(message, "from", "user"))).lower()
    aliases = {
        "human": "user",
        "user": "user",
        "gpt": "assistant",
        "assistant": "assistant",
        "model": "assistant",
        "system": "system",
        "tool": "tool",
    }
    return aliases.get(role, role)


def normalize_messages(record: Mapping[str, Any]) -> List[Dict[str, str]]:
    """Normalize common OpenHermes conversation field names."""

    messages = (
        record.get("conversations")
        or record.get("messages")
        or record.get("conversation")
    )
    if messages is None:
        prompt = record.get(
            "prompt",
            record.get(
                "receiver_prompt",
                record.get("question", record.get("instruction", "")),
            ),
        )
        answer = record.get(
            "answer",
            record.get("response", record.get("output", record.get("completion", ""))),
        )
        messages = [
            {"role": "user", "content": str(prompt)},
            {"role": "assistant", "content": str(answer)},
        ]
    normalized = [{"role": _role(item), "content": _content(item)} for item in messages]
    normalized = [
        item
        for item in normalized
        if item["role"] in {"system", "user", "assistant", "tool"}
    ]
    if not normalized:
        raise ValueError("An OpenHermes record has no messages")
    if normalized[-1]["role"] != "assistant":
        # The contract requires an explicit gold assistant answer.
        answer = record.get("answer", record.get("response", record.get("output")))
        if answer is None:
            raise ValueError("The final OpenHermes message must be assistant")
        normalized.append({"role": "assistant", "content": str(answer)})
    return normalized


def normalize_prompt_messages(value: Any) -> List[Dict[str, str]]:
    """Normalize a prompt-only message list without requiring an answer."""

    if isinstance(value, str):
        return [{"role": "user", "content": value}]
    if isinstance(value, Mapping):
        value = value.get("messages", value.get("conversations", value))
    if not isinstance(value, Sequence) or isinstance(value, (bytes, str)):
        return [{"role": "user", "content": str(value)}]
    normalized = [{"role": _role(item), "content": _content(item)} for item in value]
    if not normalized:
        raise ValueError("A prompt message list cannot be empty")
    return normalized


def _template(
    tokenizer: Any,
    messages: Sequence[Mapping[str, str]],
    *,
    add_generation_prompt: bool,
) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        kwargs = dict(
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        # Qwen tokenizers support this flag; small test tokenizers may not.
        try:
            return tokenizer.apply_chat_template(
                list(messages), enable_thinking=False, **kwargs
            )
        except TypeError:
            return tokenizer.apply_chat_template(list(messages), **kwargs)
    return "\n".join(f"{m['role']}: {m['content']}" for m in messages)


def _tokenize(tokenizer: Any, text: str) -> List[int]:
    result = tokenizer(text, add_special_tokens=False)
    ids = result["input_ids"] if isinstance(result, Mapping) else result.input_ids
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(x) for x in ids]


def _pad_id(tokenizer: Any) -> int:
    value = getattr(tokenizer, "pad_token_id", None)
    if value is None:
        value = getattr(tokenizer, "eos_token_id", 0)
    return int(0 if value is None else value)


@dataclass
class OpenHermesRecord:
    messages: List[Dict[str, str]]
    receiver_messages: Optional[List[Dict[str, str]]] = None
    sharer_messages: Optional[List[Dict[str, str]]] = None
    dataset_index: int = -1
    source: str = ""
    category: str = ""
    domain: str = ""

    @classmethod
    def from_mapping(cls, record: Mapping[str, Any], dataset_index: int = -1) -> "OpenHermesRecord":
        messages = normalize_messages(record)
        receiver_messages = record.get("receiver_messages")
        sharer_messages = record.get("sharer_messages")
        if receiver_messages is None and record.get("receiver_prompt") is not None:
            receiver_messages = record["receiver_prompt"]
        if sharer_messages is None and record.get("sharer_prompt") is not None:
            sharer_messages = record["sharer_prompt"]
        return cls(
            messages=messages,
            receiver_messages=(
                normalize_prompt_messages(receiver_messages)
                if receiver_messages is not None
                else None
            ),
            sharer_messages=(
                normalize_prompt_messages(sharer_messages)
                if sharer_messages is not None
                else None
            ),
            dataset_index=int(record.get("dataset_index", dataset_index)),
            source=str(record.get("source", record.get("dataset", ""))),
            category=str(record.get("category", "")),
            domain=str(
                record.get(
                    "domain",
                    record.get(
                        "source",
                        record.get("dataset", record.get("topic", "")),
                    ),
                )
            ),
        )


class OpenHermesDataset(Dataset):
    """Tokenized Draft-KV examples with explicit index selection."""

    def __init__(
        self,
        records: Sequence[Mapping[str, Any]],
        receiver_tokenizer: Any,
        sharer_tokenizer: Any,
        *,
        indices: Optional[Sequence[int]] = None,
        max_receiver_length: int = 4096,
        max_sharer_length: int = 4096,
    ) -> None:
        self.records = [
            item if isinstance(item, OpenHermesRecord) else OpenHermesRecord.from_mapping(item, i)
            for i, item in enumerate(records)
        ]
        self.receiver_tokenizer = receiver_tokenizer
        self.sharer_tokenizer = sharer_tokenizer
        self.max_receiver_length = int(max_receiver_length)
        self.max_sharer_length = int(max_sharer_length)
        # Do not silently select range(num_samples): an explicit list is the
        # only way this dataset changes its sample membership.
        self.indices = (
            list(range(len(self.records)))
            if indices is None
            else [int(index) for index in indices]
        )
        if any(index < 0 or index >= len(self.records) for index in self.indices):
            raise IndexError("Draft-KV dataset index is outside the source record list")

    def __len__(self) -> int:
        return len(self.indices)

    def _encode_side(
        self,
        tokenizer: Any,
        messages: Sequence[Mapping[str, str]],
        max_length: int,
    ) -> Tuple[List[int], int]:
        prompt_messages = list(messages[:-1])
        prompt_text = _template(tokenizer, prompt_messages, add_generation_prompt=True)
        full_text = _template(tokenizer, messages, add_generation_prompt=False)
        prompt_ids = _tokenize(tokenizer, prompt_text)
        full_ids = _tokenize(tokenizer, full_text)
        # In case a custom template does not preserve the prompt as a prefix,
        # use the first occurrence/longest safe prefix rather than leaking any
        # answer tokens into the prompt mask.
        prompt_len = min(len(prompt_ids), len(full_ids))
        if full_ids[:prompt_len] != prompt_ids[:prompt_len]:
            prompt_len = min(prompt_len, max(0, len(full_ids) - 1))
        if len(full_ids) > max_length:
            # Preserve the assistant answer and truncate the oldest context.
            answer_ids = full_ids[prompt_len:]
            if len(answer_ids) >= max_length:
                full_ids = answer_ids[-max_length:]
                prompt_len = 0
            else:
                keep_prompt = max_length - len(answer_ids)
                full_ids = full_ids[max(0, prompt_len - keep_prompt) : prompt_len] + answer_ids
                prompt_len = min(keep_prompt, len(full_ids))
        return full_ids, prompt_len

    def __getitem__(self, item: int) -> Dict[str, Any]:
        source_index = self.indices[item]
        record = self.records[source_index]
        receiver_messages = list(record.receiver_messages or record.messages)
        sharer_messages = list(record.sharer_messages or record.messages)
        gold_answer = record.messages[-1]
        if _role(receiver_messages[-1]) != "assistant":
            receiver_messages.append(gold_answer)
        receiver_ids, receiver_prompt_len = self._encode_side(
            self.receiver_tokenizer, receiver_messages, self.max_receiver_length
        )
        # Sharer receives prompt messages only, never the answer.
        if _role(sharer_messages[-1]) == "assistant":
            sharer_prompt_messages = sharer_messages[:-1]
        else:
            sharer_prompt_messages = sharer_messages
        sharer_text = _template(
            self.sharer_tokenizer,
            sharer_prompt_messages,
            add_generation_prompt=True,
        )
        sharer_ids = _tokenize(self.sharer_tokenizer, sharer_text)
        sharer_ids = sharer_ids[-self.max_sharer_length :]
        receiver_prompt_len = min(receiver_prompt_len, len(receiver_ids))
        receiver_prompt_mask = [1] * receiver_prompt_len + [0] * (
            len(receiver_ids) - receiver_prompt_len
        )
        labels = [-100] * receiver_prompt_len + receiver_ids[receiver_prompt_len:]
        return {
            "receiver_input_ids": receiver_ids,
            "receiver_prompt_mask": receiver_prompt_mask,
            "receiver_attention_mask": [1] * len(receiver_ids),
            "sharer_input_ids": sharer_ids,
            "sharer_attention_mask": [1] * len(sharer_ids),
            "labels": labels,
            "dataset_index": record.dataset_index if record.dataset_index >= 0 else source_index,
            "source": record.source,
            "category": record.category,
            "domain": record.domain,
        }


def make_explicit_splits(
    num_records: int,
    *,
    train_indices: Optional[Sequence[int]] = None,
    heldout_indices: Optional[Sequence[int]] = None,
    train_examples: int = 2048,
    heldout_examples: int = 256,
    seed: int = 42,
) -> Tuple[List[int], List[int]]:
    """Build non-overlapping splits without assuming file order is semantic."""

    if train_indices is not None or heldout_indices is not None:
        if train_indices is None:
            heldout = list(heldout_indices or [])
            heldout_set = set(heldout)
            train = [index for index in range(num_records) if index not in heldout_set]
        elif heldout_indices is None:
            train = list(train_indices)
            train_set = set(train)
            heldout = [index for index in range(num_records) if index not in train_set]
        else:
            train = list(train_indices)
            heldout = list(heldout_indices)
    else:
        all_indices = list(range(num_records))
        random.Random(seed).shuffle(all_indices)
        train = all_indices[: min(train_examples, num_records)]
        remaining = [i for i in all_indices if i not in set(train)]
        heldout = remaining[: min(heldout_examples, len(remaining))]
    if set(train) & set(heldout):
        raise ValueError("train_indices and heldout_indices must be disjoint")
    for index in train + heldout:
        if not 0 <= int(index) < num_records:
            raise IndexError("an explicit split index is out of range")
    return [int(i) for i in train], [int(i) for i in heldout]


class OpenHermesDataCollator:
    """Pad independent receiver/sharer streams and construct chunk masks."""

    def __init__(
        self,
        receiver_tokenizer: Any,
        sharer_tokenizer: Any,
        *,
        chunk_size: int = 32,
        num_memory_cells: int = 16,
        pad_to_multiple_of: Optional[int] = None,
    ) -> None:
        self.receiver_pad_id = _pad_id(receiver_tokenizer)
        self.sharer_pad_id = _pad_id(sharer_tokenizer)
        self.chunk_size = int(chunk_size)
        self.num_memory_cells = int(num_memory_cells)
        self.pad_to_multiple_of = pad_to_multiple_of

    @staticmethod
    def _target_length(length: int, multiple: Optional[int]) -> int:
        if multiple is None:
            return length
        return ((length + multiple - 1) // multiple) * multiple

    def __call__(self, features: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        if not features:
            raise ValueError("OpenHermesDataCollator received an empty batch")
        receiver_len = self._target_length(
            max(len(x["receiver_input_ids"]) for x in features),
            self.pad_to_multiple_of,
        )
        sharer_len = self._target_length(
            max(len(x["sharer_input_ids"]) for x in features),
            self.pad_to_multiple_of,
        )
        receiver_ids, receiver_mask, prompt_mask, labels = [], [], [], []
        sharer_ids, sharer_mask = [], []
        metadata: Dict[str, List[Any]] = {
            "dataset_index": [],
            "source": [],
            "category": [],
            "domain": [],
        }
        for feature in features:
            rids = list(feature["receiver_input_ids"])
            rmask = list(feature["receiver_attention_mask"])
            rpm = list(feature["receiver_prompt_mask"])
            lbl = list(feature["labels"])
            sids = list(feature["sharer_input_ids"])
            smask = list(feature["sharer_attention_mask"])
            receiver_ids.append(rids + [self.receiver_pad_id] * (receiver_len - len(rids)))
            receiver_mask.append(rmask + [0] * (receiver_len - len(rmask)))
            prompt_mask.append(rpm + [0] * (receiver_len - len(rpm)))
            labels.append(lbl + [-100] * (receiver_len - len(lbl)))
            sharer_ids.append(sids + [self.sharer_pad_id] * (sharer_len - len(sids)))
            sharer_mask.append(smask + [0] * (sharer_len - len(smask)))
            for key in metadata:
                metadata[key].append(feature.get(key, ""))
        batch_sharer_mask = torch.tensor(sharer_mask, dtype=torch.long)
        chunk_masks = torch.zeros(
            len(features),
            self.num_memory_cells,
            sharer_len,
            dtype=torch.bool,
        )
        for cell in range(self.num_memory_cells):
            start = cell * self.chunk_size
            end = min(start + self.chunk_size, sharer_len)
            if start < sharer_len:
                chunk_masks[:, cell, start:end] = batch_sharer_mask[:, start:end].bool()
        return {
            "receiver_input_ids": torch.tensor(receiver_ids, dtype=torch.long),
            "receiver_attention_mask": torch.tensor(receiver_mask, dtype=torch.long),
            "receiver_prompt_mask": torch.tensor(prompt_mask, dtype=torch.long),
            "sharer_input_ids": torch.tensor(sharer_ids, dtype=torch.long),
            "sharer_attention_mask": batch_sharer_mask,
            "sharer_chunk_masks": chunk_masks,
            "labels": torch.tensor(labels, dtype=torch.long),
            **metadata,
        }


def load_openhermes_records(
    source: Union[str, Path, Sequence[Mapping[str, Any]]],
    *,
    split: str = "train",
) -> List[Dict[str, Any]]:
    """Load JSON/JSONL/HuggingFace records without forcing datasets at import."""

    if not isinstance(source, (str, Path)):
        return [dict(item) for item in source]
    path = Path(source)
    if path.exists():
        if path.suffix.lower() == ".jsonl":
            return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        payload = json.loads(path.read_text())
        if isinstance(payload, Mapping):
            payload = payload.get(split, payload.get("data", []))
        return [dict(item) for item in payload]
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise FileNotFoundError(
            f"{source} is neither a local file nor available without datasets"
        ) from exc
    data = load_dataset(str(source), split=split)
    return [dict(data[i]) for i in range(len(data))]


def synthetic_openhermes_records(num_records: int = 16) -> List[Dict[str, Any]]:
    """Small deterministic records used by CPU tests and smoke runs."""

    return [
        {
            "conversations": [
                {"role": "user", "content": f"Solve toy problem {i}: what is {i}+1?"},
                {"role": "assistant", "content": str(i + 1)},
            ],
            "dataset_index": i,
            "source": "synthetic",
            "category": "math",
            "domain": "toy",
        }
        for i in range(num_records)
    ]


__all__ = [
    "OpenHermesRecord",
    "OpenHermesDataset",
    "OpenHermesDataCollator",
    "load_openhermes_records",
    "make_explicit_splits",
    "normalize_messages",
    "synthetic_openhermes_records",
]
