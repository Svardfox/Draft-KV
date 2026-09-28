import copy
import hashlib
import json

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM, Qwen3Config, Qwen3ForCausalLM

from draft_kv.model.draft_kv import DraftKVModel
from draft_kv.train.draft_kv_reconstruction_data import (
    DraftKVReconstructionCollator,
    OpenHermesDraftKVReconstructionDataset,
    build_reconstruction_record,
    encode_reconstruction_record,
    extract_private_key,
    make_no_fixed_point_id_mapping,
    private_key_for,
    reconstruction_prompt_input_ids,
)
from script.draft_kv.draft_kv_reconstruction_common import (
    natural_payload,
    summarize_reconstruction,
    token_reconstruction_stats,
    validate_go_prerequisite,
    verify_receiver_prompt,
)


class _CharTokenizer:
    pad_token_id = 1
    eos_token_id = 1

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    ):
        del tokenize, enable_thinking
        text = "".join(
            f"<{row['role']}>{row['content']}</{row['role']}>\n" for row in messages
        )
        if add_generation_prompt:
            text += "<assistant>"
        return text

    def __call__(self, text, *, add_special_tokens=False):
        del add_special_tokens
        return {"input_ids": [2 + (ord(value) % 509) for value in text]}

    def decode(self, ids, *, skip_special_tokens=True):
        del ids, skip_special_tokens
        return ""


def _raw_record(index: int, answer: str):
    return {
        "conversations": [
            {"from": "human", "value": f"Question {index}"},
            {"from": "gpt", "value": answer},
        ],
        "source": "unit-test",
    }


def test_reconstruction_data_hides_message_from_constant_receiver_prompt():
    tokenizer = _CharTokenizer()
    first = build_reconstruction_record(
        _raw_record(1, "The first private answer."),
        dataset_index=1,
        seed=91827,
    )
    second = build_reconstruction_record(
        _raw_record(2, "A completely different response."),
        dataset_index=2,
        seed=91827,
    )
    first["split"] = "train"
    second["split"] = "train"
    encoded_first = encode_reconstruction_record(
        first,
        tokenizer,
        tokenizer,
        min_message_tokens=1,
        max_message_tokens=512,
        max_receiver_length=1024,
        max_sharer_length=1024,
    )
    encoded_second = encode_reconstruction_record(
        second,
        tokenizer,
        tokenizer,
        min_message_tokens=1,
        max_message_tokens=512,
        max_receiver_length=1024,
        max_sharer_length=1024,
    )
    assert encoded_first["receiver_prompt_input_ids"] == encoded_second[
        "receiver_prompt_input_ids"
    ]
    prompt_length = len(encoded_first["receiver_prompt_input_ids"])
    assert encoded_first["labels"][:prompt_length] == [-100] * prompt_length
    assert encoded_first["labels"][prompt_length:] == encoded_first[
        "receiver_target_token_ids"
    ]
    source_message_count = sum(encoded_first["sharer_draft_mask"])
    assert source_message_count == encoded_first["sharer_message_tokens"]
    assert all(
        value == 0
        for value in encoded_first["sharer_draft_mask"][:-source_message_count]
    )
    assert extract_private_key(first["message"]) == first["private_key"]
    assert first["private_key"] == private_key_for(1, 91827)
    assert first["private_key"] != second["private_key"]


def test_reconstruction_rejects_context_roles_the_normalizer_would_drop():
    record = _raw_record(1, "answer")
    record["conversations"].insert(
        1, {"from": "tool", "value": "sample-specific hidden tool result"}
    )
    try:
        build_reconstruction_record(record, dataset_index=1, seed=91827)
    except ValueError as error:
        assert "unknown roles" in str(error)
    else:
        raise AssertionError("tool-bearing source conversations must be rejected")


def test_reconstruction_dataset_collator_preserves_real_pad_equal_eos():
    tokenizer = _CharTokenizer()
    records = []
    for index, answer in enumerate(("short", "a somewhat longer message")):
        row = build_reconstruction_record(
            _raw_record(index, answer),
            dataset_index=index,
            seed=4,
        )
        row["split"] = "train"
        records.append(row)
    dataset = OpenHermesDraftKVReconstructionDataset(
        records,
        tokenizer,
        tokenizer,
        split="train",
        min_message_tokens=1,
        max_message_tokens=512,
        max_receiver_length=1024,
        max_sharer_length=1024,
    )
    batch = DraftKVReconstructionCollator(tokenizer, tokenizer)(
        [dataset[0], dataset[1]]
    )
    receiver_lengths = [len(dataset[index]["receiver_input_ids"]) for index in range(2)]
    sharer_lengths = [len(dataset[index]["sharer_input_ids"]) for index in range(2)]
    assert batch["receiver_attention_mask"].sum(dim=1).tolist() == receiver_lengths
    assert batch["sharer_attention_mask"].sum(dim=1).tolist() == sharer_lengths
    assert torch.equal(
        batch["sharer_draft_mask"].bool()
        & ~batch["sharer_attention_mask"].bool(),
        torch.zeros_like(batch["sharer_draft_mask"], dtype=torch.bool),
    )


def test_reconstruction_derangement_is_deterministic_and_has_no_fixed_point():
    ids = [f"example:{index}" for index in range(12)]
    first = make_no_fixed_point_id_mapping(ids, seed=71)
    second = make_no_fixed_point_id_mapping(ids, seed=71)
    assert first == second
    assert set(first) == set(ids)
    assert set(first.values()) == set(ids)
    assert all(target != donor for target, donor in first.items())


def test_receiver_prompt_manifest_verification_detects_tokenizer_drift():
    tokenizer = _CharTokenizer()
    token_ids = reconstruction_prompt_input_ids(tokenizer)
    digest = hashlib.sha256(
        json.dumps(token_ids, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    manifest = {
        "reconstruction_prompt": {
            "token_count": len(token_ids),
            "receiver_token_ids_sha256": digest,
        }
    }
    assert verify_receiver_prompt(tokenizer, manifest)["token_ids_sha256"] == digest
    manifest["reconstruction_prompt"]["token_count"] += 1
    try:
        verify_receiver_prompt(tokenizer, manifest)
    except RuntimeError as error:
        assert "token count" in str(error)
    else:
        raise AssertionError("tokenizer drift must fail prompt verification")


def _nll_rows(count: int = 20):
    values = {"matched": 0.1, "deranged": 0.9, "static": 0.8, "zero": 1.0}
    return [
        {
            "condition": condition,
            "example_id": f"example:{index}",
            "nll": value,
            "token_correct": 9 if condition == "matched" else 1,
            "token_count": 10,
        }
        for index in range(count)
        for condition, value in values.items()
    ]


def _generation_rows(count: int = 20):
    rows = []
    for index in range(count):
        for condition in ("matched", "deranged"):
            rows.append(
                {
                    "condition": condition,
                    "example_id": f"example:{index}",
                    "private_key_matches_donor": True,
                    "follows_donor": condition == "deranged",
                    "exact_to_donor": True,
                    "sequence_ratio_to_donor": 1.0,
                    "natural_sequence_ratio_to_donor": 1.0,
                }
            )
    return rows


def test_reconstruction_summary_keeps_generation_metrics_diagnostic():
    passed = summarize_reconstruction(
        _nll_rows(),
        _generation_rows(),
        bootstrap_samples=500,
        seed=3,
    )
    assert passed["decision"] == "GO"
    failed_rows = copy.deepcopy(_generation_rows())
    for row in failed_rows:
        if row["condition"] == "deranged":
            row["follows_donor"] = False
    failed = summarize_reconstruction(
        _nll_rows(),
        failed_rows,
        bootstrap_samples=500,
        seed=3,
    )
    assert failed["decision"] == "GO"


def test_natural_payload_removes_only_the_protocol_canary_suffix():
    assert natural_payload(
        "A natural answer.\n\nTransmission key: DRAFT-KV-KEY-012345ABCDEF"
    ) == "A natural answer."
    assert natural_payload("Transmission key is discussed in prose.") == (
        "Transmission key is discussed in prose."
    )


def test_go_prerequisite_is_bound_to_data_records_and_checkpoint(tmp_path):
    data_manifest = tmp_path / "data_manifest.json"
    records = tmp_path / "records.jsonl"
    data_manifest.write_text("{}\n", encoding="utf-8")
    records.write_text("{}\n", encoding="utf-8")

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    gate = tmp_path / "gate"
    gate.mkdir()
    eval_manifest = gate / "eval_manifest.json"
    eval_manifest.write_text(
        json.dumps(
            {
                "protocol": "draft_kv_openhermes_reconstruction",
                "split": "gate_val",
                "data_manifest_sha256": digest(data_manifest),
                "records_sha256": digest(records),
                "checkpoint_sha256": "checkpoint-sha",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    result = gate / "eval_result.json"
    payload = {
        "protocol": "draft_kv_openhermes_reconstruction",
        "split": "gate_val",
        "decision": "GO",
        "checkpoint_sha256": "checkpoint-sha",
        "eval_manifest_sha256": digest(eval_manifest),
    }
    result.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    checked = validate_go_prerequisite(
        result,
        expected_split="gate_val",
        data_manifest_path=data_manifest,
        records_path=records,
        checkpoint_sha256="checkpoint-sha",
    )
    assert checked["decision"] == "GO"

    payload["decision"] = "NO_GO"
    result.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    try:
        validate_go_prerequisite(
            result,
            expected_split="gate_val",
            data_manifest_path=data_manifest,
            records_path=records,
            checkpoint_sha256="checkpoint-sha",
        )
    except RuntimeError as error:
        assert "not GO" in str(error)
    else:
        raise AssertionError("NO_GO must not authorize the next protocol gate")


def _tiny_models():
    common = dict(
        vocab_size=512,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=1024,
        use_cache=True,
        _attn_implementation="eager",
    )
    receiver = Qwen2ForCausalLM(Qwen2Config(**common)).eval()
    sharer_config = dict(common)
    sharer_config["num_key_value_heads"] = 1
    sharer = Qwen3ForCausalLM(Qwen3Config(**sharer_config)).eval()
    return receiver, sharer


def test_reconstruction_batch_backpropagates_only_through_communication_path():
    torch.manual_seed(17)
    tokenizer = _CharTokenizer()
    records = []
    for index, answer in enumerate(("alpha payload", "beta payload")):
        row = build_reconstruction_record(
            _raw_record(index, answer),
            dataset_index=index,
            seed=8,
        )
        row["split"] = "train"
        records.append(row)
    dataset = OpenHermesDraftKVReconstructionDataset(
        records,
        tokenizer,
        tokenizer,
        split="train",
        min_message_tokens=1,
        max_message_tokens=512,
        max_receiver_length=1024,
        max_sharer_length=1024,
    )
    batch = DraftKVReconstructionCollator(tokenizer, tokenizer)(
        [dataset[0], dataset[1]]
    )
    receiver, sharer = _tiny_models()
    model = DraftKVModel(receiver, sharer, {2: 2})
    model.set_stage("communication")
    with torch.no_grad():
        model.consumer.external["2"].gate_logits.fill_(0.1)
    packet = model.make_packet(
        batch["sharer_input_ids"],
        batch["sharer_attention_mask"],
        batch["sharer_draft_mask"],
    )
    logits, _, _ = model.forward_receiver(
        batch["receiver_input_ids"],
        batch["receiver_attention_mask"],
        packet=packet,
        use_cache=False,
    )
    stats = token_reconstruction_stats(logits, batch["labels"])
    stats["mean"].mean().backward()
    assert all(
        parameter.grad is not None and float(parameter.grad.abs().sum()) > 0
        for parameter in model.projection.parameters()
    )
    assert model.consumer.external["2"].gate_logits.grad is not None
    assert not any(parameter.grad is not None for parameter in receiver.parameters())
    assert not any(parameter.grad is not None for parameter in sharer.parameters())
