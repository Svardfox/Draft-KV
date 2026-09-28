"""Normalized multiple-choice data for Draft-KV downstream evaluation.

The adapter deliberately keeps dataset parsing separate from model inference.
It follows the prompt style used by ``script/evaluation/unified_evaluator.py``
while fixing two easy-to-miss details: MMLU-Redux corrected labels and ARC's
occasionally numeric/non-contiguous source labels.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Dict, Mapping, Optional, Sequence


DATASET_ALIASES = {
    "mmlu-redux": "mmlu-redux",
    "mmlu_redux": "mmlu-redux",
    "openbookqa": "openbookqa",
    "openbook": "openbookqa",
    "open-book": "openbookqa",
    "ceval": "ceval",
    "c-eval": "ceval",
    "c_eval": "ceval",
    "arc-e": "arc-e",
    "arc-easy": "arc-e",
    "arc_e": "arc-e",
    "arc-c": "arc-c",
    "arc-challenge": "arc-c",
    "arc_c": "arc-c",
    "gsm-mc": "gsm-mc",
    "gsm_mc": "gsm-mc",
    "gsm-mc-4": "gsm-mc",
    "gsm8k-mc": "gsm-mc",
    "math-mc": "math-mc",
    "math_mc": "math-mc",
    "math-mc-4": "math-mc",
}


def canonical_dataset_name(value: str) -> str:
    key = str(value).strip().lower()
    try:
        return DATASET_ALIASES[key]
    except KeyError as error:
        raise ValueError(
            f"unknown dataset {value!r}; choose mmlu-redux, arc-e, arc-c, "
            "openbookqa, ceval, gsm-mc, or math-mc"
        ) from error


def _answer_index(value: Any, *, option_count: int) -> int:
    if isinstance(value, bool):
        raise ValueError("boolean is not a valid answer")
    if isinstance(value, int):
        result = int(value)
    else:
        text = str(value).strip().upper()
        if re.fullmatch(r"[0-9]+", text):
            result = int(text)
        elif len(text) == 1 and "A" <= text <= "J":
            result = ord(text) - ord("A")
        else:
            raise ValueError(f"invalid answer value {value!r}")
    if not 0 <= result < int(option_count):
        raise ValueError(f"answer index {result} is outside {option_count} choices")
    return result


def _stable_id(dataset: str, subject: str, split: str, source_id: str) -> str:
    canonical = json.dumps(
        [dataset, subject, split, source_id],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:20]
    return f"{dataset}:{subject}:{split}:{digest}"


def normalize_mmlu_redux(
    row: Mapping[str, Any],
    *,
    subject: str,
    split: str,
    source_index: int,
) -> Optional[Dict[str, Any]]:
    """Normalize one MMLU-Redux row, returning ``None`` if it is unscorable."""

    error_type = str(row.get("error_type", "") or "").strip().lower()
    # This matches the option-logit evaluator: these rows do not have a uniquely
    # scorable target and must not silently fall back to a label.
    if error_type in {"no_correct_answer", "expert"}:
        return None
    question = str(row.get("question", "")).strip()
    choices = [str(value).strip() for value in row.get("choices", [])]
    if (
        not question
        or len(choices) < 2
        or len(choices) > 10
        or any(not x for x in choices)
    ):
        raise ValueError("invalid MMLU-Redux question or choices")
    answer_value = row.get("answer")
    if error_type == "wrong_groundtruth" and row.get("correct_answer") is not None:
        answer_value = row["correct_answer"]
    answer = _answer_index(answer_value, option_count=len(choices))
    source_id = str(row.get("id", row.get("question_id", source_index)))
    dataset = "mmlu-redux"
    return {
        "example_id": _stable_id(dataset, subject, split, source_id),
        "dataset": dataset,
        "subject": str(subject),
        "split": str(split),
        "source_id": source_id,
        "source_index": int(source_index),
        "question": question,
        "choices": choices,
        "answer_index": answer,
        "answer_label": chr(ord("A") + answer),
        "source_error_type": error_type,
    }


def _arc_choices(value: Any) -> tuple[list[str], list[str]]:
    if isinstance(value, Mapping):
        texts = list(value.get("text", []))
        labels = list(value.get("label", []))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        texts, labels = [], []
        for index, item in enumerate(value):
            if isinstance(item, Mapping):
                texts.append(item.get("text", ""))
                labels.append(item.get("label", chr(ord("A") + index)))
            else:
                texts.append(item)
                labels.append(chr(ord("A") + index))
    else:
        raise ValueError("ARC choices have an unknown representation")
    texts = [str(value).strip() for value in texts]
    labels = [str(value).strip().upper() for value in labels]
    if (
        not 2 <= len(texts) <= 10
        or len(texts) != len(labels)
        or any(not value for value in texts + labels)
        or len(set(labels)) != len(labels)
    ):
        raise ValueError("invalid ARC choices")
    return texts, labels


def normalize_arc(
    row: Mapping[str, Any],
    *,
    config: str,
    split: str,
    source_index: int,
) -> Dict[str, Any]:
    """Normalize ARC without assuming that source labels are A/B/C/D."""

    canonical_config = str(config).strip().lower()
    if canonical_config in {"arc-e", "arc-easy", "easy"}:
        dataset, subject = "arc-e", "ARC-Easy"
    elif canonical_config in {"arc-c", "arc-challenge", "challenge"}:
        dataset, subject = "arc-c", "ARC-Challenge"
    else:
        raise ValueError(f"unknown ARC config {config!r}")
    question = str(row.get("question", row.get("question_stem", ""))).strip()
    choices, source_labels = _arc_choices(row.get("choices"))
    answer_key = str(row.get("answerKey", row.get("answer_key", ""))).strip().upper()
    if not question or answer_key not in source_labels:
        raise ValueError(
            "ARC question is empty or answerKey is absent from choice labels"
        )
    answer = source_labels.index(answer_key)
    source_id = str(row.get("id", source_index))
    return {
        "example_id": _stable_id(dataset, subject, split, source_id),
        "dataset": dataset,
        "subject": subject,
        "split": str(split),
        "source_id": source_id,
        "source_index": int(source_index),
        "question": question,
        "choices": choices,
        "answer_index": int(answer),
        "answer_label": chr(ord("A") + answer),
        "source_choice_labels": source_labels,
        "source_answer_key": answer_key,
    }


def _choice_texts(value: Any) -> tuple[list[str], list[str]]:
    """Normalize HF choices represented as a dict or a list of dicts."""
    if isinstance(value, Mapping):
        texts = list(value.get("text", []))
        labels = list(value.get("label", []))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        texts, labels = [], []
        for index, item in enumerate(value):
            if isinstance(item, Mapping):
                texts.append(item.get("text", ""))
                labels.append(item.get("label", chr(ord("A") + index)))
            else:
                texts.append(item)
                labels.append(chr(ord("A") + index))
    else:
        raise ValueError("multiple-choice choices have an unknown representation")
    texts = [str(item).strip() for item in texts]
    labels = [str(item).strip().upper() for item in labels]
    if (
        not 2 <= len(texts) <= 10
        or len(texts) != len(labels)
        or any(not item for item in texts + labels)
        or len(set(labels)) != len(labels)
    ):
        raise ValueError("invalid multiple-choice choices")
    return texts, labels


def normalize_openbookqa(
    row: Mapping[str, Any], *, split: str, source_index: int
) -> Dict[str, Any]:
    """Normalize an OpenBookQA row from the ``main`` HF configuration."""
    question = str(row.get("question_stem", row.get("question", ""))).strip()
    choices, source_labels = _choice_texts(row.get("choices"))
    answer_key = str(row.get("answerKey", row.get("answer_key", ""))).strip().upper()
    if not question or answer_key not in source_labels:
        raise ValueError("OpenBookQA question or answerKey is invalid")
    answer = source_labels.index(answer_key)
    source_id = str(row.get("id", source_index))
    dataset, subject = "openbookqa", "main"
    return {
        "example_id": _stable_id(dataset, subject, split, source_id),
        "dataset": dataset,
        "subject": subject,
        "split": str(split),
        "source_id": source_id,
        "source_index": int(source_index),
        "question": question,
        "choices": choices,
        "answer_index": int(answer),
        "answer_label": chr(ord("A") + answer),
        "source_choice_labels": source_labels,
        "source_answer_key": answer_key,
    }


def normalize_ceval(
    row: Mapping[str, Any], *, subject: str, split: str, source_index: int
) -> Optional[Dict[str, Any]]:
    """Normalize one C-EVAL row (the public labeled ``val`` split)."""
    question = str(row.get("question", "")).strip()
    choices = [str(row.get(letter, "")).strip() for letter in "ABCD"]
    answer_value = row.get("answer")
    if answer_value is None or str(answer_value).strip() == "":
        return None
    if not question or any(not value for value in choices):
        raise ValueError("C-EVAL question or choices are invalid")
    answer = _answer_index(answer_value, option_count=4)
    dataset = "ceval"
    source_id = str(row.get("id", row.get("question_id", source_index)))
    return {
        "example_id": _stable_id(dataset, subject, split, source_id),
        "dataset": dataset,
        "subject": str(subject),
        "split": str(split),
        "source_id": source_id,
        "source_index": int(source_index),
        "question": question,
        "choices": choices,
        "answer_index": int(answer),
        "answer_label": chr(ord("A") + answer),
    }


def normalize_gsm_mc(
    row: Mapping[str, Any],
    *,
    split: str,
    source_index: int,
) -> Optional[Dict[str, Any]]:
    """Normalize one 4-way MC row from the MC-Evaluation release (GSM-MC or MATH-MC).

    Rows whose choices were dropped by the upstream LaTeX rendering pipeline
    (empty option strings) are unscorable and return ``None``.
    """

    dataset = "gsm-mc"
    subject = "GSM-MC"
    question = str(row.get("Question", row.get("question", ""))).strip()
    choices = [str(row.get(letter, "")).strip() for letter in "ABCD"]
    answer_value = row.get("Answer", row.get("answer"))
    if answer_value is None or str(answer_value).strip() == "":
        return None
    if not question or any(not value for value in choices):
        return None
    answer = _answer_index(answer_value, option_count=4)
    source_id = str(row.get("id", row.get("question_id", source_index)))
    normalized: Dict[str, Any] = {
        "example_id": _stable_id(dataset, subject, split, source_id),
        "dataset": dataset,
        "subject": subject,
        "split": str(split),
        "source_id": source_id,
        "source_index": int(source_index),
        "question": question,
        "choices": choices,
        "answer_index": answer,
        "answer_label": chr(ord("A") + answer),
    }
    for extra in ("Level", "Type"):
        if row.get(extra) is not None:
            normalized[f"source_{extra.lower()}"] = str(row[extra]).strip()
    return normalized


def normalize_math_mc(
    row: Mapping[str, Any],
    *,
    split: str,
    source_index: int,
) -> Optional[Dict[str, Any]]:
    """Normalize one MATH-MC row via the shared MC-Evaluation adapter."""

    normalized = normalize_gsm_mc(row, split=split, source_index=source_index)
    if normalized is None:
        return None
    normalized["dataset"] = "math-mc"
    normalized["subject"] = "MATH-MC"
    normalized["example_id"] = _stable_id(
        "math-mc",
        normalized["subject"],
        split,
        str(normalized["source_id"]),
    )
    return normalized


def build_multiple_choice_prompt(row: Mapping[str, Any], *, use_cot: bool) -> str:
    choices = [str(value) for value in row["choices"]]
    labels = [chr(ord("A") + index) for index in range(len(choices))]
    rendered = "\n".join(f"{label}. {choice}" for label, choice in zip(labels, choices))
    allowed = "/".join(labels)
    if use_cot:
        instructions = (
            "- Carefully read the question and all options.\n"
            "- Think step by step and explain your reasoning briefly.\n"
            '- End with the exact form "The correct answer is LETTER".'
        )
    else:
        instructions = (
            "- Carefully read the question and all options.\n"
            "- Select the single most correct answer.\n"
            f'- Respond ONLY in the format "The correct answer is {allowed}".\n'
            "- Do not include explanations or additional text."
        )
    return (
        "Accurately answer the following question:\n\n"
        f"{str(row['question']).strip()}\n\n"
        f"Choices:\n{rendered}\n\n"
        f"Instructions:\n{instructions}"
    )


def build_plain_multiple_choice_prompt(row: Mapping[str, Any]) -> str:
    """Build the standard exact no-CoT MMLU prompt body."""

    choices = [str(value) for value in row["choices"]]
    rendered = "\n".join(
        f"{chr(ord('A') + index)}. {choice}" for index, choice in enumerate(choices)
    )
    allowed = "/".join(chr(ord("A") + index) for index in range(len(choices)))
    return (
        "Accurately answer the following question:\n\n"
        f"{str(row['question']).strip()}\n\n"
        f"Choices:\n{rendered}\n\n"
        "Instructions:\n"
        "- Carefully read the question and all options.\n"
        "- Select the single most correct answer.\n"
        f'- Respond ONLY in the following format: "The correct answer is {allowed}".\n'
        "- Do not include any explanations, additional text, or punctuation besides "
        "the answer.\n\n"
        "The correct answer is"
    )


def _normalized_answer_text(value: str) -> str:
    return re.sub(r"[^0-9a-z]+", " ", str(value).casefold()).strip()


def extract_choice(
    text: str,
    *,
    option_count: int,
    choices: Optional[Sequence[str]] = None,
) -> Optional[str]:
    """Extract a final A-J choice while preferring explicit answer phrases.

    Qwen commonly wraps the option letter in Markdown, for example
    ``The correct answer is **C**.``.  Some responses instead repeat the
    option text without its letter.  The latter is accepted only when it
    uniquely matches one of the supplied choices after an explicit answer
    cue, or uniquely matches the end of the response.
    """

    if not 2 <= int(option_count) <= 10:
        raise ValueError("option_count must be between 2 and 10")
    if choices is not None and len(choices) != int(option_count):
        raise ValueError("choices length must equal option_count")
    upper = chr(ord("A") + int(option_count) - 1)
    source = str(text).strip()
    wrappers = r"(?:\*\*|__|`|\$|\\boxed\s*\{|[\[({])*"
    letter_group = "([A-" + upper + "])"
    patterns = (
        r"(?:the\s+correct\s+answer\s+is|answer|choice|option)"
        + r"\s*[:=-]?\s*"
        + wrappers
        + r"\s*"
        + letter_group
        + r"(?=[^A-Za-z0-9]|$)",
        r"(?:\*\*|__|`|\$|[\[({])*\s*"
        + letter_group
        + r"\s*(?:[.)\]}]|\*\*|__|`|\$)*\s*[.!?]*\s*$",
    )
    for pattern in patterns:
        matches = re.findall(pattern, source, flags=re.IGNORECASE)
        if matches:
            return str(matches[-1]).upper()
    if choices is not None:
        normalized_choices = [_normalized_answer_text(value) for value in choices]
        if len(set(normalized_choices)) != len(normalized_choices):
            return None
        cue = re.compile(
            r"(?:the\s+correct\s+answer\s+is|answer|choice|option)" r"\s*[:=-]?\s*",
            flags=re.IGNORECASE,
        )
        candidate_segments = [
            source[match.end() : match.end() + 500] for match in cue.finditer(source)
        ]
        candidate_segments.append(source[-500:])
        for segment in reversed(candidate_segments):
            normalized = _normalized_answer_text(segment)
            matches = [
                index
                for index, option in enumerate(normalized_choices)
                if option
                and (
                    normalized == option
                    or normalized.startswith(option + " ")
                    or normalized.endswith(" " + option)
                    or f" {option} " in f" {normalized} "
                )
            ]
            if len(matches) == 1:
                return chr(ord("A") + matches[0])
    return None


def validate_normalized_row(row: Mapping[str, Any]) -> None:
    dataset = canonical_dataset_name(str(row["dataset"]))
    choices = list(row["choices"])
    answer = int(row["answer_index"])
    if not str(row.get("example_id", "")) or not str(row.get("question", "")).strip():
        raise ValueError("normalized row lacks ID or question")
    if not 2 <= len(choices) <= 10 or not 0 <= answer < len(choices):
        raise ValueError("normalized row has invalid choices or answer")
    if str(row.get("answer_label")) != chr(ord("A") + answer):
        raise ValueError("normalized row answer label/index mismatch")
    if str(row["dataset"]) != dataset:
        raise ValueError("normalized row uses a non-canonical dataset name")


__all__ = [
    "build_plain_multiple_choice_prompt",
    "build_multiple_choice_prompt",
    "canonical_dataset_name",
    "extract_choice",
    "normalize_arc",
    "normalize_ceval",
    "normalize_gsm_mc",
    "normalize_math_mc",
    "normalize_mmlu_redux",
    "normalize_openbookqa",
    "validate_normalized_row",
]
