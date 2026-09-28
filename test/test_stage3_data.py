import json
from pathlib import Path

import pytest
import torch

from draft_kv.train.draft_kv_mc_training_data import (
    DraftKVMCOptionCollator,
    deterministic_train_calibration_split,
    encode_mc_record,
    make_no_fixed_point_mapping,
    normalize_arc_file,
    option_classification_stats,
    option_distribution_kl,
    option_token_ids,
    receiver_prompt_token_ids,
    sharer_prompt_token_ids,
)
from script.draft_kv.eval_stage3 import _selection_gate


class _Tokenizer:
    pad_token_id = 0
    eos_token_id = 1
    padding_side = "right"

    @staticmethod
    def _ids(text):
        if text.startswith(" ") and len(text) == 2 and text[1].isalpha():
            return [100 + ord(text[1].upper()) - ord("A")]
        return [2 + (ord(value) % 89) for value in text]

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    ):
        del enable_thinking
        text = "".join(
            f"<{row['role']}>{row['content']}</{row['role']}>"
            for row in messages
        )
        if add_generation_prompt:
            text += "<assistant>"
        return self._ids(text) if tokenize else text

    def encode(self, text, *, add_special_tokens=False):
        del add_special_tokens
        return self._ids(text)


def _row(dataset, source_split, index):
    answer = index % 4
    row = {
        "example_id": f"{dataset}:{source_split}:{index}",
        "dataset": dataset,
        "subject": "ARC-Easy" if dataset == "arc-e" else "ARC-Challenge",
        "source_split": source_split,
        "question": f"question {index}",
        "choices": ["one", "two", "three", "four"],
        "answer_index": answer,
        "answer_label": chr(ord("A") + answer),
        "record_sha256": f"{index:064x}",
    }
    return row


def test_split_is_deterministic_stratified_and_rejects_test_rows():
    rows = [
        _row(dataset, source_split, index)
        for dataset in ("arc-e", "arc-c")
        for source_split in ("train", "validation")
        for index in range(10)
    ]
    first = deterministic_train_calibration_split(
        rows, calibration_fraction=0.2, seed=7
    )
    second = deterministic_train_calibration_split(
        list(reversed(rows)), calibration_fraction=0.2, seed=7
    )
    assert [
        (row["example_id"], row["task_split"]) for row in first
    ] == [
        (row["example_id"], row["task_split"]) for row in second
    ]
    for dataset in ("arc-e", "arc-c"):
        for source_split in ("train", "validation"):
            group = [
                row
                for row in first
                if row["dataset"] == dataset
                and row["source_split"] == source_split
            ]
            assert sum(row["task_split"] == "calibration" for row in group) == 2
            assert sum(row["task_split"] == "train" for row in group) == 8
    invalid = dict(rows[0], source_split="test")
    with pytest.raises(ValueError, match="train/validation"):
        deterministic_train_calibration_split(
            rows + [invalid], calibration_fraction=0.2, seed=7
        )


def test_normalizer_never_accepts_mmlu_as_an_arc_training_source(tmp_path):
    path = tmp_path / "row.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "x",
                "question": "q",
                "choices": {"text": ["x", "y"], "label": ["A", "B"]},
                "answerKey": "A",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="only arc-e and arc-c"):
        normalize_arc_file(
            path, dataset="mmlu-redux", source_split="train"
        )


def test_mc_encoding_has_no_gold_target_on_sharer_path_and_left_pads_receiver():
    tokenizer = _Tokenizer()
    record = _row("arc-e", "train", 3)
    record["task_split"] = "train"
    prompt = sharer_prompt_token_ids(tokenizer, record)
    draft = {
        "example_id": record["example_id"],
        "record_sha256": record["record_sha256"],
        "prompt_token_ids": prompt,
        "draft_token_ids": [71, 72, 1],
        "response": "reasoning and answer",
    }
    encoded = encode_mc_record(
        record,
        draft,
        tokenizer,
        tokenizer,
        max_receiver_length=4096,
        max_sharer_length=4096,
    )
    assert encoded["sharer_input_ids"] == prompt + [71, 72, 1]
    assert encoded["sharer_draft_mask"] == [0] * len(prompt) + [1, 1, 1]
    assert encoded["receiver_input_ids"] == receiver_prompt_token_ids(
        tokenizer, record
    )

    shorter = dict(encoded)
    shorter["receiver_input_ids"] = encoded["receiver_input_ids"][-5:]
    shorter["sharer_input_ids"] = encoded["sharer_input_ids"][-4:]
    shorter["sharer_draft_mask"] = [0, 1, 1, 1]
    batch = DraftKVMCOptionCollator(tokenizer, tokenizer)([shorter, encoded])
    assert batch["receiver_attention_mask"][0, :-5].sum().item() == 0
    assert batch["receiver_attention_mask"][0, -5:].tolist() == [1] * 5
    assert batch["sharer_attention_mask"][0, :4].tolist() == [1] * 4
    assert batch["sharer_attention_mask"][0, 4:].sum().item() == 0


def test_option_ce_masks_invalid_choices():
    tokenizer = _Tokenizer()
    ids = option_token_ids(tokenizer)
    logits = torch.zeros(2, 256)
    # Example 0 has only A/B.  An enormous C logit must be masked.
    logits[0, ids[0]] = 1.0
    logits[0, ids[1]] = 2.0
    logits[0, ids[2]] = 100.0
    # Example 1 has A/B/C/D and gold D.
    logits[1, ids[3]] = 4.0
    stats = option_classification_stats(
        logits,
        option_ids=ids,
        option_counts=torch.tensor([2, 4]),
        gold_indices=torch.tensor([1, 3]),
    )
    assert stats["prediction"].tolist() == [1, 3]
    assert stats["correct"].tolist() == [True, True]


def test_option_distribution_kl_masks_padding_and_detaches_reference():
    reference = torch.tensor(
        [[1.0, 3.0, 100.0], [2.0, 0.0, -1.0]], requires_grad=True
    )
    student = torch.tensor(
        [[1.0, 3.0, -100.0], [0.0, 2.0, -1.0]], requires_grad=True
    )
    losses = option_distribution_kl(
        reference,
        student,
        option_counts=torch.tensor([2, 3]),
    )
    assert losses.shape == (2,)
    assert losses[0].item() == pytest.approx(0.0, abs=1e-7)
    assert losses[1].item() > 0.0
    losses.sum().backward()
    assert reference.grad is None
    assert student.grad is not None
    assert student.grad[0, 2].item() == 0.0


def test_derangement_is_deterministic_and_has_no_fixed_points():
    ids = [f"x:{index}" for index in range(12)]
    first = make_no_fixed_point_mapping(ids, seed=11)
    second = make_no_fixed_point_mapping(ids, seed=11)
    assert first == second
    assert set(first) == set(ids)
    assert set(first.values()) == set(ids)
    assert all(target != donor for target, donor in first.items())


def test_test_gate_requires_selected_calibration_for_same_checkpoint(tmp_path):
    checkpoint_sha = "a" * 64
    data_sha = "b" * 64
    eval_manifest = tmp_path / "eval_manifest.json"
    eval_manifest.write_text(
        json.dumps(
            {
                "protocol": "draft_kv_mc_option_evaluation",
                "split": "calibration",
                "checkpoint_sha256": checkpoint_sha,
                "mc_data_manifest_sha256": data_sha,
            }
        ),
        encoding="utf-8",
    )
    result = tmp_path / "result.json"
    result.write_text(
        json.dumps(
            {
                "protocol": "draft_kv_mc_option_evaluation",
                "split": "calibration",
                "decision": "SELECTED",
                "checkpoint_sha256": checkpoint_sha,
                "mc_data_manifest_sha256": data_sha,
                "eval_manifest_sha256": __import__(
                    "hashlib"
                ).sha256(eval_manifest.read_bytes()).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    verified = _selection_gate(
        result,
        checkpoint_sha256=checkpoint_sha,
        mc_manifest_sha256=data_sha,
    )
    assert verified["selection_result_sha256"]
    with pytest.raises(RuntimeError, match="differs from selected checkpoint"):
        _selection_gate(
            result,
            checkpoint_sha256="c" * 64,
            mc_manifest_sha256=data_sha,
        )
