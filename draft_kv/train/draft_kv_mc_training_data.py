"""ARC option-logit training data for Draft-KV Stage 3.

Only ARC-Easy and ARC-Challenge ``train``/``validation`` source files are
accepted by the preparation path.  Their rows are deterministically repartitioned
into an MC training split and a calibration split.  ARC test rows are handled
only by the final evaluation script after a checkpoint has been selected.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import Dataset

from .downstream_mc_data import (
    build_plain_multiple_choice_prompt,
    build_multiple_choice_prompt,
    normalize_arc,
    validate_normalized_row,
)


PROTOCOL = "draft_kv_mc_option_training"
EVAL_PROTOCOL = "draft_kv_mc_option_evaluation"
SOURCE_SPLITS = ("train", "validation")
TASK_SPLITS = ("train", "calibration")
ARC_CONFIGS = {
    "arc-e": "ARC-Easy",
    "arc-c": "ARC-Challenge",
}
RECEIVER_ASSISTANT_PREFIX = "The correct answer is"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: str | Path) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(dict(json.loads(line)))
            except json.JSONDecodeError as error:
                raise RuntimeError(
                    f"invalid JSONL at {Path(path)}:{line_number}"
                ) from error
    return rows


def read_jsonl_by_id(path: str | Path) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    source = Path(path)
    if not source.exists():
        return result
    for row in read_jsonl(source):
        example_id = str(row["example_id"])
        if example_id in result:
            raise RuntimeError(f"duplicate example ID in {source}: {example_id}")
        result[example_id] = row
    return result


def write_jsonl_atomic(
    path: str | Path, rows: Iterable[Mapping[str, Any]]
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
    temporary.replace(output)


def write_json_atomic(path: str | Path, value: Mapping[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            dict(value),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            default=float,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)


def record_sha256(row: Mapping[str, Any]) -> str:
    payload = {
        "example_id": str(row["example_id"]),
        "dataset": str(row["dataset"]),
        "source_split": str(row["source_split"]),
        "question": str(row["question"]),
        "choices": [str(value) for value in row["choices"]],
        "answer_index": int(row["answer_index"]),
    }
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def normalize_arc_file(
    path: str | Path,
    *,
    dataset: str,
    source_split: str,
) -> list[Dict[str, Any]]:
    """Normalize one allowed ARC source file and bind every row to its hash."""

    if dataset not in ARC_CONFIGS:
        raise ValueError("Stage 3 accepts only arc-e and arc-c")
    if source_split not in SOURCE_SPLITS and source_split != "test":
        raise ValueError("ARC source split must be train, validation, or test")
    rows: list[Dict[str, Any]] = []
    for source_index, raw in enumerate(read_jsonl(path)):
        row = normalize_arc(
            raw,
            config=ARC_CONFIGS[dataset],
            split=source_split,
            source_index=source_index,
        )
        validate_normalized_row(row)
        row["source_split"] = str(source_split)
        row["source_path"] = str(Path(path).resolve())
        row["record_sha256"] = record_sha256(row)
        rows.append(row)
    ids = [str(row["example_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError(f"duplicate normalized IDs in {path}")
    return rows


def deterministic_train_calibration_split(
    rows: Sequence[Mapping[str, Any]],
    *,
    calibration_fraction: float,
    seed: int,
) -> list[Dict[str, Any]]:
    """Stratify by ARC dataset and original source split, then hash-shuffle."""

    if not 0.0 < float(calibration_fraction) < 0.5:
        raise ValueError("calibration_fraction must be in (0, 0.5)")
    groups: Dict[tuple[str, str], list[Dict[str, Any]]] = {}
    for raw in rows:
        row = dict(raw)
        dataset = str(row["dataset"])
        source_split = str(row["source_split"])
        if dataset not in ARC_CONFIGS or source_split not in SOURCE_SPLITS:
            raise ValueError(
                "training partition may contain only ARC train/validation rows"
            )
        groups.setdefault((dataset, source_split), []).append(row)

    train: list[Dict[str, Any]] = []
    calibration: list[Dict[str, Any]] = []
    for key in sorted(groups):
        values = groups[key]
        values.sort(
            key=lambda row: hashlib.sha256(
                (
                    f"{int(seed)}\0{key[0]}\0{key[1]}\0"
                    f"{row['example_id']}\0{row['record_sha256']}"
                ).encode("utf-8")
            ).hexdigest()
        )
        if len(values) < 2:
            raise ValueError(f"partition stratum is too small: {key}")
        calibration_count = max(
            1, min(len(values) - 1, int(round(len(values) * calibration_fraction)))
        )
        for index, row in enumerate(values):
            row["task_split"] = (
                "calibration" if index < calibration_count else "train"
            )
            (calibration if index < calibration_count else train).append(row)

    # Persist a stable, split-major order independent of source file iteration.
    train.sort(key=lambda row: str(row["example_id"]))
    calibration.sort(key=lambda row: str(row["example_id"]))
    result = train + calibration
    ids = [str(row["example_id"]) for row in result]
    if len(ids) != len(set(ids)):
        raise RuntimeError("ARC train/calibration rows are not globally unique")
    return result


def make_no_fixed_point_mapping(
    example_ids: Sequence[str], *, seed: int
) -> Dict[str, str]:
    targets = [str(value) for value in example_ids]
    if len(targets) < 2 or len(targets) != len(set(targets)):
        raise ValueError("derangement requires at least two unique IDs")
    donors = list(targets)
    rng = random.Random(int(seed))
    for _ in range(1000):
        rng.shuffle(donors)
        if all(left != right for left, right in zip(targets, donors)):
            return dict(zip(targets, donors))
    donors = targets[1:] + targets[:1]
    return dict(zip(targets, donors))


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
    encoded = tokenizer.encode(text, add_special_tokens=False)
    if isinstance(encoded, Tensor):
        encoded = encoded.tolist()
    if encoded and isinstance(encoded[0], list):
        encoded = encoded[0]
    values = [int(value) for value in encoded]
    if not values:
        raise ValueError("tokenizer produced an empty sequence")
    return values


def sharer_prompt_token_ids(
    tokenizer: Any, record: Mapping[str, Any]
) -> list[int]:
    prompt = build_multiple_choice_prompt(record, use_cot=True)
    text = _chat_text(
        tokenizer,
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
    )
    return _token_ids(tokenizer, text)


def receiver_prompt_token_ids(
    tokenizer: Any, record: Mapping[str, Any]
) -> list[int]:
    """Build the next-option-logit evaluation prompt."""

    prompt = build_plain_multiple_choice_prompt(record)
    text = _chat_text(
        tokenizer,
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
    )
    return _token_ids(tokenizer, text + RECEIVER_ASSISTANT_PREFIX)


def option_token_ids(tokenizer: Any, *, maximum: int = 10) -> list[int]:
    result = []
    for index in range(int(maximum)):
        encoded = _token_ids(
            tokenizer, " " + chr(ord("A") + index)
        )
        result.append(int(encoded[0]))
    if len(result) != len(set(result)):
        raise RuntimeError("space-prefixed option letters lack unique first tokens")
    return result


def validate_draft_row(
    row: Mapping[str, Any],
    record: Mapping[str, Any],
    tokenizer: Any | None = None,
) -> None:
    if str(row.get("example_id")) != str(record["example_id"]):
        raise RuntimeError("draft cache example ID differs from its ARC record")
    if str(row.get("record_sha256")) != str(record["record_sha256"]):
        raise RuntimeError("draft cache record SHA differs from its ARC record")
    prompt = [int(value) for value in row.get("prompt_token_ids", [])]
    draft = [int(value) for value in row.get("draft_token_ids", [])]
    if not prompt or not draft or not str(row.get("response", "")).strip():
        raise RuntimeError("draft cache contains an empty prompt or response")


def encode_mc_record(
    record: Mapping[str, Any],
    draft: Mapping[str, Any],
    receiver_tokenizer: Any,
    sharer_tokenizer: Any,
    *,
    max_receiver_length: int,
    max_sharer_length: int,
) -> Dict[str, Any]:
    validate_draft_row(draft, record, sharer_tokenizer)
    receiver_ids = receiver_prompt_token_ids(receiver_tokenizer, record)
    sharer_prompt = [int(value) for value in draft["prompt_token_ids"]]
    sharer_draft = [int(value) for value in draft["draft_token_ids"]]
    sharer_ids = sharer_prompt + sharer_draft
    if len(receiver_ids) > int(max_receiver_length):
        raise ValueError("Receiver MC prompt exceeds max_receiver_length")
    if len(sharer_ids) > int(max_sharer_length):
        raise ValueError("Sharer MC sequence exceeds max_sharer_length")
    return {
        "example_id": str(record["example_id"]),
        "dataset": str(record["dataset"]),
        "subject": str(record["subject"]),
        "source_split": str(record["source_split"]),
        "task_split": str(record["task_split"]),
        "question": str(record["question"]),
        "choices": [str(value) for value in record["choices"]],
        "answer_index": int(record["answer_index"]),
        "answer_label": str(record["answer_label"]),
        "record_sha256": str(record["record_sha256"]),
        "receiver_input_ids": receiver_ids,
        "sharer_input_ids": sharer_ids,
        "sharer_draft_mask": [0] * len(sharer_prompt) + [1] * len(sharer_draft),
        "draft_text": str(draft["response"]),
    }


class DraftKVMCOptionDataset(Dataset):
    def __init__(
        self,
        records: Sequence[Mapping[str, Any]],
        drafts: Mapping[str, Mapping[str, Any]],
        receiver_tokenizer: Any,
        sharer_tokenizer: Any,
        *,
        split: str,
        max_receiver_length: int,
        max_sharer_length: int,
    ) -> None:
        selected = [
            dict(row) for row in records if str(row.get("task_split")) == str(split)
        ]
        if not selected:
            raise ValueError(f"no MC records found for split {split!r}")
        self.rows = [
            encode_mc_record(
                row,
                drafts[str(row["example_id"])],
                receiver_tokenizer,
                sharer_tokenizer,
                max_receiver_length=max_receiver_length,
                max_sharer_length=max_sharer_length,
            )
            for row in selected
        ]
        ids = [str(row["example_id"]) for row in self.rows]
        if len(ids) != len(set(ids)):
            raise RuntimeError("MC dataset split contains duplicate IDs")
        self.by_id = {str(row["example_id"]): row for row in self.rows}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        return dict(self.rows[int(item)])


class DraftKVMCOptionCollator:
    """Left-pad Receiver prompts and right-pad independent Sharer packets."""

    def __init__(self, receiver_tokenizer: Any, sharer_tokenizer: Any) -> None:
        if (
            getattr(receiver_tokenizer, "pad_token_id", None) is None
            or getattr(sharer_tokenizer, "pad_token_id", None) is None
        ):
            raise ValueError("both tokenizers must define pad_token_id")
        self.receiver_pad = int(receiver_tokenizer.pad_token_id)
        self.sharer_pad = int(sharer_tokenizer.pad_token_id)

    @staticmethod
    def _left_pad(rows: Sequence[Sequence[int]], value: int) -> Tensor:
        width = max(len(row) for row in rows)
        return torch.tensor(
            [[int(value)] * (width - len(row)) + list(row) for row in rows],
            dtype=torch.long,
        )

    @staticmethod
    def _right_pad(
        rows: Sequence[Sequence[int]], value: int, *, width: int | None = None
    ) -> Tensor:
        observed = max(len(row) for row in rows)
        width = observed if width is None else int(width)
        if width < observed:
            raise ValueError("requested packet width is shorter than a sequence")
        return torch.tensor(
            [list(row) + [int(value)] * (width - len(row)) for row in rows],
            dtype=torch.long,
        )

    def packet_batch(
        self,
        features: Sequence[Mapping[str, Any]],
        *,
        width: int | None = None,
    ) -> Dict[str, Tensor]:
        ids = [row["sharer_input_ids"] for row in features]
        masks = [row["sharer_draft_mask"] for row in features]
        return {
            "sharer_input_ids": self._right_pad(
                ids, self.sharer_pad, width=width
            ),
            "sharer_attention_mask": self._right_pad(
                [[1] * len(row) for row in ids], 0, width=width
            ),
            "sharer_draft_mask": self._right_pad(masks, 0, width=width),
        }

    def __call__(self, features: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        if not features:
            raise ValueError("cannot collate an empty MC batch")
        receiver = [row["receiver_input_ids"] for row in features]
        packet = self.packet_batch(features)
        return {
            "example_id": [str(row["example_id"]) for row in features],
            "dataset": [str(row["dataset"]) for row in features],
            "answer_index": torch.tensor(
                [int(row["answer_index"]) for row in features], dtype=torch.long
            ),
            "answer_label": [str(row["answer_label"]) for row in features],
            "option_count": torch.tensor(
                [len(row["choices"]) for row in features], dtype=torch.long
            ),
            "receiver_input_ids": self._left_pad(receiver, self.receiver_pad),
            "receiver_attention_mask": self._left_pad(
                [[1] * len(row) for row in receiver], 0
            ),
            **packet,
        }


def option_classification_stats(
    next_token_logits: Tensor,
    *,
    option_ids: Sequence[int],
    option_counts: Tensor,
    gold_indices: Tensor,
) -> Dict[str, Tensor]:
    """Cross-entropy over each example's valid option-token logits."""

    if next_token_logits.ndim != 2:
        raise ValueError("next_token_logits must have shape [batch, vocabulary]")
    option_counts = option_counts.to(next_token_logits.device, dtype=torch.long)
    gold_indices = gold_indices.to(next_token_logits.device, dtype=torch.long)
    if option_counts.shape != gold_indices.shape or option_counts.ndim != 1:
        raise ValueError("option counts and gold indices must be rank-one peers")
    if len(option_ids) < int(option_counts.max().item()):
        raise ValueError("option_ids does not cover the largest option count")
    candidates = next_token_logits[:, list(option_ids)].float()
    positions = torch.arange(candidates.shape[1], device=candidates.device)
    valid = positions.unsqueeze(0) < option_counts.unsqueeze(1)
    candidates = candidates.masked_fill(~valid, -torch.inf)
    if bool((gold_indices < 0).any()) or bool((gold_indices >= option_counts).any()):
        raise ValueError("gold option index is outside its example's options")
    nll = F.cross_entropy(candidates, gold_indices, reduction="none")
    log_probabilities = F.log_softmax(candidates, dim=-1)
    gold_log_probability = log_probabilities.gather(
        1, gold_indices.unsqueeze(1)
    ).squeeze(1)
    predictions = candidates.argmax(dim=-1)
    return {
        "nll": nll,
        "gold_log_probability": gold_log_probability,
        "prediction": predictions,
        "correct": predictions.eq(gold_indices),
        "candidate_logits": candidates,
    }


def option_distribution_kl(
    reference_candidate_logits: Tensor,
    student_candidate_logits: Tensor,
    *,
    option_counts: Tensor,
) -> Tensor:
    """Return per-example ``KL(reference || student)`` over valid options.

    The reference distribution is detached inside this function.  This makes
    it suitable for distilling the frozen Receiver-only judgment into the
    Deranged communication condition without allowing the reference branch to
    move.  Invalid padded option positions are excluded from both
    distributions.
    """

    if reference_candidate_logits.shape != student_candidate_logits.shape:
        raise ValueError("reference and student candidate logits must match")
    if reference_candidate_logits.ndim != 2:
        raise ValueError("candidate logits must have shape [batch, options]")
    if option_counts.ndim != 1 or option_counts.shape[0] != (
        reference_candidate_logits.shape[0]
    ):
        raise ValueError("option_counts must be rank one with one value per row")

    option_counts = option_counts.to(
        reference_candidate_logits.device, dtype=torch.long
    )
    maximum = int(reference_candidate_logits.shape[1])
    if bool((option_counts <= 0).any()) or bool((option_counts > maximum).any()):
        raise ValueError("option_counts is outside the candidate-logit width")

    positions = torch.arange(maximum, device=reference_candidate_logits.device)
    valid = positions.unsqueeze(0) < option_counts.unsqueeze(1)
    reference = reference_candidate_logits.float().detach().masked_fill(
        ~valid, -torch.inf
    )
    student = student_candidate_logits.float().masked_fill(~valid, -torch.inf)
    reference_log_probability = F.log_softmax(reference, dim=-1)
    student_log_probability = F.log_softmax(student, dim=-1)
    reference_probability = reference_log_probability.exp()

    # Replace invalid -inf values before subtraction so padded positions cannot
    # introduce NaNs through expressions such as 0 * (-inf - -inf).
    reference_log_probability = reference_log_probability.masked_fill(~valid, 0.0)
    student_log_probability = student_log_probability.masked_fill(~valid, 0.0)
    terms = reference_probability * (
        reference_log_probability - student_log_probability
    )
    return terms.masked_fill(~valid, 0.0).sum(dim=-1)


def deranged_no_harm_loss(
    deranged_nll: Tensor,
    zero_nll: Tensor,
    *,
    tolerance: float,
) -> Tensor:
    """Per-example excess answer NLL; Zero is a detached reference.

    Inputs may also be length-normalized response NLLs for generation.
    Apply the hinge before averaging examples so improvements cannot cancel harm.
    """
    import math

    if deranged_nll.shape != zero_nll.shape:
        raise ValueError("Deranged and Zero NLL shapes must match")
    if not math.isfinite(float(tolerance)) or tolerance < 0:
        raise ValueError("tolerance must be finite and nonnegative")
    return F.relu(deranged_nll - zero_nll.detach() - float(tolerance))


def load_mc_bundle(
    data_dir: str | Path,
) -> tuple[
    Dict[str, Any],
    list[Dict[str, Any]],
    Dict[str, Dict[str, Any]],
    Path,
    Path,
    Path,
]:
    root = Path(data_dir)
    manifest_path = root / "data_manifest.json"
    result_path = root / "prepare_result.json"
    if not manifest_path.is_file() or not result_path.is_file():
        raise FileNotFoundError("MC data directory is not completely prepared")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not str(manifest.get("protocol", "")).startswith(PROTOCOL):
        raise RuntimeError("unexpected MC data protocol")
    records_path = root / str(manifest["records_file"])
    drafts_path = root / str(manifest["draft_cache_file"])
    if sha256_file(records_path) != str(manifest["records_sha256"]):
        raise RuntimeError("MC records SHA differs from manifest")
    records = read_jsonl(records_path)
    drafts = read_jsonl_by_id(drafts_path)
    ids = [str(row["example_id"]) for row in records]
    expected = [
        str(example_id)
        for split in TASK_SPLITS
        for example_id in manifest["split_ids"][split]
    ]
    if ids != expected or len(ids) != len(set(ids)):
        raise RuntimeError("MC records order/IDs differ from manifest")
    if set(ids) != set(drafts):
        raise RuntimeError("MC draft cache IDs differ from records")
    expected_split = {
        str(example_id): split
        for split, values in manifest["split_ids"].items()
        for example_id in values
    }
    for row in records:
        validate_normalized_row(row)
        if str(row.get("source_split")) not in SOURCE_SPLITS:
            raise RuntimeError("MC training bundle contains a non-training ARC split")
        if str(row.get("task_split")) != expected_split[str(row["example_id"])]:
            raise RuntimeError("MC record split differs from manifest")
        if record_sha256(row) != str(row.get("record_sha256")):
            raise RuntimeError("MC record content hash mismatch")
        validate_draft_row(drafts[str(row["example_id"])], row)
    for split, key in (
        ("train", "train_derangement"),
        ("calibration", "calibration_derangement"),
    ):
        split_ids = set(str(value) for value in manifest["split_ids"][split])
        mapping = {
            str(target): str(donor)
            for target, donor in manifest[key].items()
        }
        if (
            set(mapping) != split_ids
            or set(mapping.values()) != split_ids
            or any(target == donor for target, donor in mapping.items())
        ):
            raise RuntimeError(f"{key} is not a full derangement")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if (
        not str(result.get("protocol", "")).startswith(PROTOCOL)
        or result.get("manifest_sha256") != sha256_file(manifest_path)
        or result.get("records_sha256") != sha256_file(records_path)
        or result.get("draft_cache_sha256") != sha256_file(drafts_path)
    ):
        raise RuntimeError("prepare_result does not authenticate the MC bundle")
    return manifest, records, drafts, manifest_path, records_path, drafts_path


__all__ = [
    "ARC_CONFIGS",
    "DraftKVMCOptionCollator",
    "DraftKVMCOptionDataset",
    "EVAL_PROTOCOL",
    "PROTOCOL",
    "RECEIVER_ASSISTANT_PREFIX",
    "SOURCE_SPLITS",
    "TASK_SPLITS",
    "deterministic_train_calibration_split",
    "encode_mc_record",
    "load_mc_bundle",
    "make_no_fixed_point_mapping",
    "normalize_arc_file",
    "option_classification_stats",
    "option_distribution_kl",
    "option_token_ids",
    "read_jsonl",
    "read_jsonl_by_id",
    "receiver_prompt_token_ids",
    "record_sha256",
    "sha256_file",
    "sharer_prompt_token_ids",
    "validate_draft_row",
    "write_json_atomic",
    "write_jsonl_atomic",
]
