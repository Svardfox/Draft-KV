import copy

import torch

from draft_kv.train.draft_kv_openhermes_stage2_data import (
    DraftKVStage2Collator,
    OpenHermesDraftKVStage2Dataset,
    build_stage2_record,
    encode_stage2_record,
    sharer_prompt_token_ids,
    text_augmented_context,
)
from script.draft_kv.eval_draft_kv_openhermes_stage2 import summarize


class _Tokenizer:
    pad_token_id = 1
    eos_token_id = 1

    @staticmethod
    def _ids(text):
        return [2 + (ord(value) % 509) for value in text]

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
            f"<{row['role']}>{row['content']}</{row['role']}>\n"
            for row in messages
        )
        if add_generation_prompt:
            text += "<assistant>"
        return self._ids(text) if tokenize else text

    def __call__(self, text, *, add_special_tokens=False):
        del add_special_tokens
        return {"input_ids": self._ids(text)}


def _source(gold="The gold response."):
    return {
        "conversations": [
            {"from": "system", "value": "Be concise."},
            {"from": "human", "value": "What is the answer?"},
            {"from": "gpt", "value": gold},
        ],
        "source": "unit-test",
    }


def _draft(record, tokenizer):
    return {
        "example_id": record["example_id"],
        "record_sha256": record["record_sha256"],
        "prompt_token_ids": sharer_prompt_token_ids(tokenizer, record),
        "draft_token_ids": [91, 92, 1],
        "terminated_eos": True,
        "response": "A fallible collaborator draft.",
    }


def test_stage2_sharer_path_never_contains_gold_answer_tokens():
    tokenizer = _Tokenizer()
    first = build_stage2_record(_source("Gold one."), dataset_index=1)
    second = build_stage2_record(_source("A different gold."), dataset_index=2)
    # The context is identical, so the frozen Sharer sees the same prompt even
    # though Receiver supervision differs.
    assert sharer_prompt_token_ids(tokenizer, first) == sharer_prompt_token_ids(
        tokenizer, second
    )
    first["split"] = "train"
    draft = _draft(first, tokenizer)
    encoded = encode_stage2_record(
        first,
        draft,
        tokenizer,
        tokenizer,
        receiver_mode="context",
        max_receiver_length=2048,
        max_sharer_length=2048,
    )
    prompt_length = len(draft["prompt_token_ids"])
    assert encoded["sharer_input_ids"] == (
        draft["prompt_token_ids"] + draft["draft_token_ids"]
    )
    assert encoded["sharer_draft_mask"][:prompt_length] == [0] * prompt_length
    assert encoded["sharer_draft_mask"][prompt_length:] == [1, 1, 1]
    assert encoded["labels"][-len(encoded["receiver_target_token_ids"]):] == encoded[
        "receiver_target_token_ids"
    ]


def test_text_only_and_text_plus_latent_use_identical_visible_prompt_and_packet():
    tokenizer = _Tokenizer()
    record = build_stage2_record(_source(), dataset_index=3)
    record["split"] = "gate_val"
    draft = _draft(record, tokenizer)
    context = encode_stage2_record(
        record,
        draft,
        tokenizer,
        tokenizer,
        receiver_mode="context",
        max_receiver_length=2048,
        max_sharer_length=2048,
    )
    text = encode_stage2_record(
        record,
        draft,
        tokenizer,
        tokenizer,
        receiver_mode="text",
        max_receiver_length=4096,
        max_sharer_length=2048,
    )
    assert context["receiver_prompt_input_ids"] != text["receiver_prompt_input_ids"]
    assert context["receiver_target_token_ids"] == text["receiver_target_token_ids"]
    assert context["sharer_input_ids"] == text["sharer_input_ids"]
    assert context["sharer_draft_mask"] == text["sharer_draft_mask"]
    augmented = text_augmented_context(record["context_messages"], draft["response"])
    assert draft["response"] in augmented[-1]["content"]


def test_stage2_dataset_collator_preserves_real_eos_when_pad_equals_eos():
    tokenizer = _Tokenizer()
    records = []
    drafts = {}
    for index, gold in enumerate(("Short.", "A somewhat longer gold response.")):
        row = build_stage2_record(_source(gold), dataset_index=index)
        row["split"] = "train"
        records.append(row)
        drafts[row["example_id"]] = _draft(row, tokenizer)
    dataset = OpenHermesDraftKVStage2Dataset(
        records,
        drafts,
        tokenizer,
        tokenizer,
        split="train",
        max_receiver_length=4096,
        max_sharer_length=4096,
    )
    batch = DraftKVStage2Collator(tokenizer, tokenizer)([dataset[0], dataset[1]])
    assert batch["receiver_attention_mask"].sum(dim=1).tolist() == [
        len(dataset[index]["receiver_input_ids"]) for index in range(2)
    ]
    assert torch.equal(
        batch["sharer_draft_mask"].bool()
        & ~batch["sharer_attention_mask"].bool(),
        torch.zeros_like(batch["sharer_draft_mask"], dtype=torch.bool),
    )


def _nll_rows(count=24):
    values = {
        "zero": 2.0,
        "matched": 1.0,
        "deranged": 2.2,
        "static": 2.1,
        "text_only": 0.9,
        "text_matched": 0.5,
        "text_deranged": 1.0,
        "text_static": 1.1,
    }
    return [
        {
            "condition": condition,
            "example_id": f"example:{index}",
            "nll": value,
            "token_correct": 1,
            "token_count": 2,
        }
        for index in range(count)
        for condition, value in values.items()
    ]


def test_stage2_summary_separates_core_go_from_beyond_text_evidence():
    checkpoint = {"reconstruction_preservation": {"eligible": True}}
    passed = summarize(
        _nll_rows(), checkpoint=checkpoint, bootstrap_samples=500, seed=9
    )
    assert passed["decision"] == "GO"
    assert passed["beyond_text_evidence"] == "YES"
    failed_rows = copy.deepcopy(_nll_rows())
    for row in failed_rows:
        if row["condition"] == "text_deranged":
            row["nll"] = 0.5
    failed = summarize(
        failed_rows, checkpoint=checkpoint, bootstrap_samples=500, seed=9
    )
    assert failed["decision"] == "GO"
    assert failed["beyond_text_evidence"] == "NO"
