"""OpenHermes Stage-2 data flow for response-Draft-KV communication.

The frozen Sharer receives the conversation context and greedily generates a
complete draft response.  Its exact prompt and generated token IDs are stored
in a cache.  The frozen Receiver receives the same context and is supervised
on the dataset's gold assistant response.  The gold response is never present
on the Sharer path.

This module also builds the text-to-text control.  It adds the decoded Sharer
draft to the *same* Receiver context used by both ``text_only`` and
``text_plus_latent`` arms, so their only difference is the latent packet.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import torch
from torch.utils.data import Dataset

from .draft_kv_reconstruction_data import encode_assistant_message
from .openhermes import normalize_messages


PROTOCOL = "draft_kv_openhermes_stage2"
EOS_POOL_PROTOCOL = "draft_kv_openhermes_stage2_eos_pool"
SUPPORTED_PROTOCOLS = frozenset({PROTOCOL, EOS_POOL_PROTOCOL})
TEXT_DRAFT_HEADER = (
    "A collaborating model proposed the draft answer below. Treat it as "
    "fallible evidence, check it, and then answer the original request."
)


def _chat_token_ids(
    tokenizer: Any,
    messages: Sequence[Mapping[str, str]],
    *,
    add_generation_prompt: bool,
) -> list[int]:
    kwargs = {
        "tokenize": True,
        "add_generation_prompt": bool(add_generation_prompt),
    }
    try:
        encoded = tokenizer.apply_chat_template(
            list(messages), enable_thinking=False, **kwargs
        )
    except TypeError:
        encoded = tokenizer.apply_chat_template(list(messages), **kwargs)
    if isinstance(encoded, Mapping):
        encoded = encoded["input_ids"]
    if isinstance(encoded, torch.Tensor):
        encoded = encoded.tolist()
    if encoded and isinstance(encoded[0], list):
        encoded = encoded[0]
    return [int(value) for value in encoded]


def build_stage2_record(
    record: Mapping[str, Any], *, dataset_index: int
) -> Dict[str, Any]:
    """Persist the context/gold boundary without model-specific token IDs."""

    messages = normalize_messages(record)
    if any(row["role"] not in {"system", "user", "assistant"} for row in messages):
        raise ValueError("Stage 2 excludes tool and unknown roles")
    if len(messages) < 2 or messages[-1]["role"] != "assistant":
        raise ValueError("Stage 2 source must end with a gold assistant message")
    context = [dict(row) for row in messages[:-1]]
    if not context or context[-1]["role"] != "user":
        raise ValueError("Stage 2 context must end with a user message")
    gold = str(messages[-1]["content"]).strip()
    if not gold:
        raise ValueError("Stage 2 gold assistant message is empty")
    canonical = json.dumps(
        {"context": context, "gold": gold},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return {
        "example_id": f"openhermes:{int(dataset_index)}",
        "dataset_index": int(dataset_index),
        "source": str(record.get("source", record.get("dataset", "OpenHermes-2.5"))),
        "context_messages": context,
        "gold_message": gold,
        "record_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    }


def sharer_prompt_token_ids(tokenizer: Any, record: Mapping[str, Any]) -> list[int]:
    ids = _chat_token_ids(
        tokenizer,
        record["context_messages"],
        add_generation_prompt=True,
    )
    if not ids:
        raise ValueError("Sharer chat template produced an empty prompt")
    return ids


def receiver_target_encoding(
    tokenizer: Any, record: Mapping[str, Any]
) -> tuple[list[int], int, list[int]]:
    return encode_assistant_message(
        tokenizer,
        record["context_messages"],
        str(record["gold_message"]),
    )


def text_augmented_context(
    context_messages: Sequence[Mapping[str, str]], draft_text: str
) -> list[Dict[str, str]]:
    """Inject visible draft text while preserving a valid final user turn."""

    context = [dict(row) for row in context_messages]
    if not context or context[-1].get("role") != "user":
        raise ValueError("text control requires a context ending in user")
    draft = str(draft_text).strip()
    if not draft:
        raise ValueError("text control cannot receive an empty draft")
    context[-1]["content"] = (
        str(context[-1].get("content", ""))
        + "\n\n"
        + TEXT_DRAFT_HEADER
        + "\n<collaborator_draft>\n"
        + draft
        + "\n</collaborator_draft>"
    )
    return context


def read_jsonl_by_id(path: str | Path) -> Dict[str, Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    source = Path(path)
    if not source.exists():
        return rows
    for line in source.read_text(encoding="utf-8").split("\n"):
        if not line.strip():
            continue
        row = json.loads(line)
        key = str(row["example_id"])
        if key in rows:
            raise ValueError(f"duplicate JSONL row: {key}")
        rows[key] = row
    return rows


def write_jsonl(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
    temporary.replace(output)


def validate_draft_row(
    row: Mapping[str, Any],
    record: Mapping[str, Any],
    sharer_tokenizer: Any | None = None,
) -> None:
    if str(row.get("example_id")) != str(record["example_id"]):
        raise ValueError("Stage 2 draft/record example ID mismatch")
    if str(row.get("record_sha256")) != str(record["record_sha256"]):
        raise ValueError("Stage 2 draft was generated from a different record")
    prompt = [int(value) for value in row.get("prompt_token_ids", [])]
    draft = [int(value) for value in row.get("draft_token_ids", [])]
    if not prompt or not draft:
        raise ValueError("Stage 2 draft cache contains an empty prompt or response")


def encode_stage2_record(
    record: Mapping[str, Any],
    draft: Mapping[str, Any],
    receiver_tokenizer: Any,
    sharer_tokenizer: Any,
    *,
    receiver_mode: str,
    max_receiver_length: int,
    max_sharer_length: int,
) -> Dict[str, Any]:
    if receiver_mode not in {"context", "text"}:
        raise ValueError("receiver_mode must be 'context' or 'text'")
    validate_draft_row(draft, record, sharer_tokenizer)
    context = [dict(row) for row in record["context_messages"]]
    if receiver_mode == "text":
        context = text_augmented_context(context, str(draft["response"]))
    receiver_full, receiver_prompt_length, receiver_target = encode_assistant_message(
        receiver_tokenizer,
        context,
        str(record["gold_message"]),
    )
    sharer_prompt = [int(value) for value in draft["prompt_token_ids"]]
    sharer_draft = [int(value) for value in draft["draft_token_ids"]]
    sharer_full = sharer_prompt + sharer_draft
    if len(receiver_full) > int(max_receiver_length):
        raise ValueError("Stage 2 Receiver sequence exceeds max_receiver_length")
    if len(sharer_full) > int(max_sharer_length):
        raise ValueError("Stage 2 Sharer sequence exceeds max_sharer_length")
    return {
        "example_id": str(record["example_id"]),
        "dataset_index": int(record["dataset_index"]),
        "split": str(record.get("split", "")),
        "gold_message": str(record["gold_message"]),
        "draft_text": str(draft["response"]),
        "receiver_mode": receiver_mode,
        "receiver_prompt_input_ids": receiver_full[:receiver_prompt_length],
        "receiver_input_ids": receiver_full,
        "receiver_target_token_ids": receiver_target,
        "labels": [-100] * receiver_prompt_length + receiver_target,
        "sharer_input_ids": sharer_full,
        # Only generated response tokens are visible as communication memory.
        "sharer_draft_mask": [0] * len(sharer_prompt) + [1] * len(sharer_draft),
    }


class OpenHermesDraftKVStage2Dataset(Dataset):
    """Gold-answer Receiver examples paired with cached Sharer response KVs."""

    def __init__(
        self,
        records: Sequence[Mapping[str, Any]],
        drafts: Mapping[str, Mapping[str, Any]],
        receiver_tokenizer: Any,
        sharer_tokenizer: Any,
        *,
        split: str,
        receiver_mode: str = "context",
        max_receiver_length: int = 1024,
        max_sharer_length: int = 2048,
    ) -> None:
        selected = [dict(row) for row in records if str(row.get("split")) == split]
        if not selected:
            raise ValueError(f"no Stage 2 records found for split {split!r}")
        self.rows = [
            encode_stage2_record(
                row,
                drafts[str(row["example_id"])],
                receiver_tokenizer,
                sharer_tokenizer,
                receiver_mode=receiver_mode,
                max_receiver_length=max_receiver_length,
                max_sharer_length=max_sharer_length,
            )
            for row in selected
        ]
        ids = [str(row["example_id"]) for row in self.rows]
        if len(ids) != len(set(ids)):
            raise ValueError("Stage 2 dataset contains duplicate example IDs")
        self.by_id = {str(row["example_id"]): row for row in self.rows}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        return dict(self.rows[int(item)])


class DraftKVStage2Collator:
    """Right-pad independent Receiver and Sharer streams."""

    def __init__(self, receiver_tokenizer: Any, sharer_tokenizer: Any) -> None:
        if receiver_tokenizer.pad_token_id is None or sharer_tokenizer.pad_token_id is None:
            raise ValueError("both tokenizers must define pad_token_id")
        self.receiver_pad = int(receiver_tokenizer.pad_token_id)
        self.sharer_pad = int(sharer_tokenizer.pad_token_id)

    @staticmethod
    def _pad(rows: Sequence[Sequence[int]], value: int) -> torch.Tensor:
        width = max(len(row) for row in rows)
        return torch.tensor(
            [list(row) + [int(value)] * (width - len(row)) for row in rows],
            dtype=torch.long,
        )

    def __call__(self, features: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        if not features:
            raise ValueError("cannot collate an empty Stage 2 batch")
        receiver_ids = self._pad(
            [row["receiver_input_ids"] for row in features], self.receiver_pad
        )
        sharer_ids = self._pad(
            [row["sharer_input_ids"] for row in features], self.sharer_pad
        )
        return {
            "example_id": [str(row["example_id"]) for row in features],
            "dataset_index": [int(row["dataset_index"]) for row in features],
            "gold_message": [str(row["gold_message"]) for row in features],
            "draft_text": [str(row["draft_text"]) for row in features],
            "receiver_mode": [str(row["receiver_mode"]) for row in features],
            "receiver_input_ids": receiver_ids,
            "receiver_attention_mask": self._pad(
                [[1] * len(row["receiver_input_ids"]) for row in features], 0
            ),
            "labels": self._pad([row["labels"] for row in features], -100),
            "sharer_input_ids": sharer_ids,
            "sharer_attention_mask": self._pad(
                [[1] * len(row["sharer_input_ids"]) for row in features], 0
            ),
            "sharer_draft_mask": self._pad(
                [row["sharer_draft_mask"] for row in features], 0
            ),
        }


__all__ = [
    "DraftKVStage2Collator",
    "EOS_POOL_PROTOCOL",
    "OpenHermesDraftKVStage2Dataset",
    "PROTOCOL",
    "SUPPORTED_PROTOCOLS",
    "TEXT_DRAFT_HEADER",
    "build_stage2_record",
    "encode_stage2_record",
    "read_jsonl_by_id",
    "receiver_target_encoding",
    "sharer_prompt_token_ids",
    "text_augmented_context",
    "validate_draft_row",
    "write_jsonl",
]
