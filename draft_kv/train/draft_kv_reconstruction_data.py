"""OpenHermes data flow for Draft-KV latent-to-text reconstruction.

This module is intentionally separate from :mod:`openhermes`.  The normal
Draft-KV OpenHermes path hides the final assistant response from the Sharer because
it trains answer prediction.  Reconstruction does the opposite: the frozen
Sharer teacher-forces the complete final assistant message, while the frozen
Receiver sees only one constant decoding instruction.  Consequently the
sample-specific target can reach the Receiver only through the Draft-KV packet.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import torch
from torch.utils.data import Dataset

from .openhermes import normalize_messages


PROTOCOL = "draft_kv_openhermes_reconstruction"
RECONSTRUCTION_SYSTEM = (
    "You are a lossless communication decoder. Recover the hidden message "
    "exactly and do not add, remove, explain, or rewrite anything."
)
RECONSTRUCTION_USER = "Output the hidden message exactly."
DRAFT_KV_KEY_PREFIX = "DRAFT-KV-KEY-"
PRIVATE_KEY_PATTERN = re.compile(r"DRAFT-KV-KEY-[0-9A-F]{12}", re.IGNORECASE)


def _validate_explicit_conversation(record: Mapping[str, Any]) -> None:
    """Reject explicit conversations that the shared normalizer would alter."""

    messages = (
        record.get("conversations")
        or record.get("messages")
        or record.get("conversation")
    )
    if messages is None:
        return
    if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
        raise ValueError("source conversation is not a message sequence")
    aliases = {
        "human": "user",
        "user": "user",
        "gpt": "assistant",
        "assistant": "assistant",
        "model": "assistant",
        "system": "system",
    }
    roles = []
    for item in messages:
        if isinstance(item, Mapping):
            raw_role = item.get("role", item.get("from", "user"))
        else:
            raw_role = getattr(item, "role", getattr(item, "from", "user"))
        role = aliases.get(str(raw_role).lower())
        if role is None:
            raise ValueError("source conversation contains tool or unknown roles")
        roles.append(role)
    if not roles or roles[-1] != "assistant":
        raise ValueError("explicit source conversation must end with assistant")


def normalize_message_text(value: Any) -> str:
    """Normalize transport-level newlines while preserving message wording."""

    return str(value).replace("\r\n", "\n").replace("\r", "\n").strip()


def canonical_message_hash(value: Any) -> str:
    """Return a duplicate-resistant hash without changing the training target."""

    text = normalize_message_text(value)
    canonical = " ".join(unicodedata.normalize("NFKC", text).casefold().split())
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def private_key_for(dataset_index: int, seed: int) -> str:
    """Create a deterministic, sample-private held-out reconstruction canary."""

    payload = f"{PROTOCOL}:{int(seed)}:{int(dataset_index)}".encode("utf-8")
    suffix = hashlib.sha256(payload).hexdigest()[:12].upper()
    return f"{DRAFT_KV_KEY_PREFIX}{suffix}"


def append_private_key(message: str, private_key: str) -> str:
    message = normalize_message_text(message)
    if not message:
        raise ValueError("reconstruction message cannot be empty")
    if PRIVATE_KEY_PATTERN.search(message):
        raise ValueError("source message already contains a Draft-KV private key")
    return f"{message}\n\nTransmission key: {private_key}"


def reconstruction_prompt_messages() -> list[Dict[str, str]]:
    """Return a fresh copy of the sample-independent Receiver prompt."""

    return [
        {"role": "system", "content": RECONSTRUCTION_SYSTEM},
        {"role": "user", "content": RECONSTRUCTION_USER},
    ]


def reconstruction_prompt_input_ids(tokenizer: Any) -> list[int]:
    """Encode the one Receiver prompt shared by every reconstruction sample."""

    return _token_ids(
        tokenizer,
        _chat_text(
            tokenizer,
            reconstruction_prompt_messages(),
            add_generation_prompt=True,
        ),
    )


def _chat_text(
    tokenizer: Any,
    messages: Sequence[Mapping[str, str]],
    *,
    add_generation_prompt: bool,
) -> str:
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": bool(add_generation_prompt),
    }
    try:
        return tokenizer.apply_chat_template(
            list(messages), enable_thinking=False, **kwargs
        )
    except TypeError:
        return tokenizer.apply_chat_template(list(messages), **kwargs)


def _token_ids(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    ids = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded.input_ids
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(value) for value in ids]


def encode_assistant_message(
    tokenizer: Any,
    prompt_messages: Sequence[Mapping[str, str]],
    message: str,
) -> tuple[list[int], int, list[int]]:
    """Encode a chat and return ``(full_ids, prompt_length, message_suffix)``.

    Prefix equality is deliberately strict.  A fuzzy boundary could expose
    prompt tokens as message memory or mask actual message tokens, invalidating
    the causal reconstruction test.
    """

    prompt = [dict(item) for item in prompt_messages]
    full = prompt + [{"role": "assistant", "content": str(message)}]
    prompt_ids = _token_ids(
        tokenizer,
        _chat_text(tokenizer, prompt, add_generation_prompt=True),
    )
    full_ids = _token_ids(
        tokenizer,
        _chat_text(tokenizer, full, add_generation_prompt=False),
    )
    if not prompt_ids:
        raise ValueError("chat template produced an empty prompt")
    if len(full_ids) <= len(prompt_ids):
        raise ValueError("chat template produced no assistant-message tokens")
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("assistant message is not an exact token suffix of the prompt")
    suffix = full_ids[len(prompt_ids) :]
    return full_ids, len(prompt_ids), suffix


def build_reconstruction_record(
    record: Mapping[str, Any],
    *,
    dataset_index: int,
    seed: int,
    include_private_key: bool = True,
) -> Dict[str, Any]:
    """Convert one OpenHermes record into a persisted reconstruction record."""

    _validate_explicit_conversation(record)
    messages = normalize_messages(record)
    if any(item["role"] not in {"system", "user", "assistant"} for item in messages):
        raise ValueError("tool and unknown roles are excluded from reconstruction")
    if not any(item["role"] == "user" for item in messages[:-1]):
        raise ValueError("source conversation has no user message")
    natural_message = normalize_message_text(messages[-1]["content"])
    if not natural_message:
        raise ValueError("final assistant message is empty")
    private_key = private_key_for(dataset_index, seed) if include_private_key else ""
    message = (
        append_private_key(natural_message, private_key)
        if include_private_key
        else natural_message
    )
    source_messages = [dict(item) for item in messages[:-1]] + [
        {"role": "assistant", "content": message}
    ]
    return {
        "example_id": f"openhermes:{int(dataset_index)}",
        "dataset_index": int(dataset_index),
        "source": str(record.get("source", record.get("dataset", "OpenHermes-2.5"))),
        "natural_message": natural_message,
        "message": message,
        "message_sha256": hashlib.sha256(message.encode("utf-8")).hexdigest(),
        "canonical_message_sha256": canonical_message_hash(natural_message),
        "private_key": private_key,
        "source_messages": source_messages,
    }


def encode_reconstruction_record(
    record: Mapping[str, Any],
    receiver_tokenizer: Any,
    sharer_tokenizer: Any,
    *,
    min_message_tokens: int,
    max_message_tokens: int,
    max_receiver_length: int,
    max_sharer_length: int,
) -> Dict[str, Any]:
    """Tokenize one persisted record with strict, independent model tokenizers."""

    message = normalize_message_text(record["message"])
    source_messages = [dict(item) for item in record["source_messages"]]
    if not source_messages or source_messages[-1].get("role") != "assistant":
        raise ValueError(
            "source_messages must end with the transmitted assistant message"
        )
    if normalize_message_text(source_messages[-1].get("content", "")) != message:
        raise ValueError("persisted message differs from source_messages")
    source_text = "\n".join(str(item.get("content", "")) for item in source_messages)
    for name, tokenizer in (
        ("Receiver", receiver_tokenizer),
        ("Sharer", sharer_tokenizer),
    ):
        for special in getattr(tokenizer, "all_special_tokens", ()):
            if special and str(special) in source_text:
                raise ValueError(
                    f"source text contains a reserved {name} special token"
                )

    sharer_full, sharer_prompt_length, sharer_suffix = encode_assistant_message(
        sharer_tokenizer,
        source_messages[:-1],
        message,
    )
    receiver_prompt = reconstruction_prompt_messages()
    receiver_full, receiver_prompt_length, receiver_suffix = encode_assistant_message(
        receiver_tokenizer,
        receiver_prompt,
        message,
    )
    _, _, natural_receiver_suffix = encode_assistant_message(
        receiver_tokenizer,
        receiver_prompt,
        normalize_message_text(record["natural_message"]),
    )
    message_tokens = len(receiver_suffix)
    natural_message_tokens = len(natural_receiver_suffix)
    if natural_message_tokens < int(min_message_tokens):
        raise ValueError("natural Receiver message is shorter than min_message_tokens")
    if message_tokens > int(max_message_tokens):
        raise ValueError("Receiver message is longer than max_message_tokens")
    if len(receiver_full) > int(max_receiver_length):
        raise ValueError("Receiver reconstruction sequence exceeds max_receiver_length")
    if len(sharer_full) > int(max_sharer_length):
        raise ValueError("Sharer source sequence exceeds max_sharer_length")

    observed_lengths = {
        "receiver_message_tokens": message_tokens,
        "receiver_natural_message_tokens": natural_message_tokens,
        "sharer_message_tokens": len(sharer_suffix),
        "receiver_sequence_tokens": len(receiver_full),
        "sharer_sequence_tokens": len(sharer_full),
    }
    for key, observed in observed_lengths.items():
        if key in record and int(record[key]) != int(observed):
            raise RuntimeError(
                f"persisted {key} differs from current tokenizer encoding"
            )

    labels = [-100] * receiver_prompt_length + receiver_suffix
    return {
        "example_id": str(record["example_id"]),
        "dataset_index": int(record["dataset_index"]),
        "split": str(record.get("split", "")),
        "message": message,
        "natural_message": normalize_message_text(record["natural_message"]),
        "message_sha256": str(record["message_sha256"]),
        "private_key": str(record.get("private_key", "")),
        "receiver_prompt_input_ids": receiver_full[:receiver_prompt_length],
        "receiver_input_ids": receiver_full,
        "receiver_target_token_ids": receiver_suffix,
        "labels": labels,
        "sharer_input_ids": sharer_full,
        # The Draft-KV model API calls this a draft mask. It marks exactly
        # the teacher-forced final assistant message, including its terminator.
        "sharer_draft_mask": [0] * sharer_prompt_length
        + [1] * len(sharer_suffix),
        "receiver_message_tokens": message_tokens,
        "receiver_natural_message_tokens": natural_message_tokens,
        "sharer_message_tokens": len(sharer_suffix),
    }


class OpenHermesDraftKVReconstructionDataset(Dataset):
    """Frozen-Sharer message packets paired with a constant Receiver prompt."""

    def __init__(
        self,
        records: Sequence[Mapping[str, Any]],
        receiver_tokenizer: Any,
        sharer_tokenizer: Any,
        *,
        split: str,
        min_message_tokens: int = 16,
        max_message_tokens: int = 128,
        max_receiver_length: int = 256,
        max_sharer_length: int = 1024,
    ) -> None:
        selected = [dict(row) for row in records if str(row.get("split")) == str(split)]
        if not selected:
            raise ValueError(f"no reconstruction records found for split {split!r}")
        self.rows = [
            encode_reconstruction_record(
                row,
                receiver_tokenizer,
                sharer_tokenizer,
                min_message_tokens=min_message_tokens,
                max_message_tokens=max_message_tokens,
                max_receiver_length=max_receiver_length,
                max_sharer_length=max_sharer_length,
            )
            for row in selected
        ]
        ids = [row["example_id"] for row in self.rows]
        if len(ids) != len(set(ids)):
            raise ValueError("reconstruction split contains duplicate example IDs")
        self.by_id = {str(row["example_id"]): row for row in self.rows}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        return dict(self.rows[int(item)])


class DraftKVReconstructionCollator:
    """Right-pad reconstruction sequences without masking real EOS tokens."""

    def __init__(self, receiver_tokenizer: Any, sharer_tokenizer: Any) -> None:
        receiver_pad = getattr(receiver_tokenizer, "pad_token_id", None)
        sharer_pad = getattr(sharer_tokenizer, "pad_token_id", None)
        if receiver_pad is None or sharer_pad is None:
            raise ValueError("both tokenizers must define pad_token_id")
        self.receiver_pad = int(receiver_pad)
        self.sharer_pad = int(sharer_pad)

    @staticmethod
    def _pad(rows: Sequence[Sequence[int]], value: int) -> torch.Tensor:
        width = max(len(row) for row in rows)
        return torch.tensor(
            [list(row) + [int(value)] * (width - len(row)) for row in rows],
            dtype=torch.long,
        )

    def __call__(self, features: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        if not features:
            raise ValueError("cannot collate an empty reconstruction batch")
        receiver_ids = self._pad(
            [row["receiver_input_ids"] for row in features], self.receiver_pad
        )
        sharer_ids = self._pad(
            [row["sharer_input_ids"] for row in features], self.sharer_pad
        )
        labels = self._pad([row["labels"] for row in features], -100)
        draft_mask = self._pad([row["sharer_draft_mask"] for row in features], 0)
        receiver_attention = self._pad(
            [[1] * len(row["receiver_input_ids"]) for row in features], 0
        )
        sharer_attention = self._pad(
            [[1] * len(row["sharer_input_ids"]) for row in features], 0
        )
        return {
            "example_id": [str(row["example_id"]) for row in features],
            "dataset_index": [int(row["dataset_index"]) for row in features],
            "split": [str(row.get("split", "")) for row in features],
            "message": [str(row["message"]) for row in features],
            "natural_message": [str(row["natural_message"]) for row in features],
            "private_key": [str(row.get("private_key", "")) for row in features],
            "receiver_input_ids": receiver_ids,
            "receiver_attention_mask": receiver_attention,
            "labels": labels,
            "sharer_input_ids": sharer_ids,
            "sharer_attention_mask": sharer_attention,
            "sharer_draft_mask": draft_mask,
        }


def make_no_fixed_point_id_mapping(
    example_ids: Sequence[str], *, seed: int
) -> Dict[str, str]:
    """Create a deterministic full-packet derangement over string IDs."""

    targets = [str(value) for value in example_ids]
    if len(targets) < 2 or len(targets) != len(set(targets)):
        raise ValueError("derangement requires at least two unique example IDs")
    donors = list(targets)
    rng = random.Random(int(seed))
    for _ in range(1000):
        rng.shuffle(donors)
        if all(target != donor for target, donor in zip(targets, donors)):
            return dict(zip(targets, donors))
    donors = targets[1:] + targets[:1]
    return dict(zip(targets, donors))


def extract_private_key(text: str) -> str | None:
    match = PRIVATE_KEY_PATTERN.search(str(text))
    return match.group(0).upper() if match else None


def load_reconstruction_records(path: str | Path) -> list[Dict[str, Any]]:
    source = Path(path)
    rows = []
    for line in source.read_text(encoding="utf-8").split("\n"):
        if line.strip():
            rows.append(json.loads(line))
    ids = [str(row["example_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("reconstruction record file contains duplicate IDs")
    return rows


def write_reconstruction_records(
    path: str | Path, records: Sequence[Mapping[str, Any]]
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
    temporary.replace(output)


__all__ = [
    "DraftKVReconstructionCollator",
    "OpenHermesDraftKVReconstructionDataset",
    "PRIVATE_KEY_PATTERN",
    "DRAFT_KV_KEY_PREFIX",
    "PROTOCOL",
    "RECONSTRUCTION_SYSTEM",
    "RECONSTRUCTION_USER",
    "append_private_key",
    "build_reconstruction_record",
    "canonical_message_hash",
    "encode_assistant_message",
    "encode_reconstruction_record",
    "extract_private_key",
    "load_reconstruction_records",
    "make_no_fixed_point_id_mapping",
    "normalize_message_text",
    "private_key_for",
    "reconstruction_prompt_input_ids",
    "reconstruction_prompt_messages",
    "write_reconstruction_records",
]
