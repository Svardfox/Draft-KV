import importlib.util
from pathlib import Path


# This adapter is deliberately CPU-only.  Load the file directly so this unit
# test does not import draft_kv.train's unrelated torch training dependencies.
_ADAPTER_PATH = (
    Path(__file__).resolve().parents[1] / "draft_kv/train/downstream_mc_data.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "draft_kv_test_downstream_mc_data", _ADAPTER_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
_ADAPTER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_ADAPTER)
build_multiple_choice_prompt = _ADAPTER.build_multiple_choice_prompt
build_plain_multiple_choice_prompt = _ADAPTER.build_plain_multiple_choice_prompt
canonical_dataset_name = _ADAPTER.canonical_dataset_name
extract_choice = _ADAPTER.extract_choice
normalize_arc = _ADAPTER.normalize_arc
normalize_gsm_mc = _ADAPTER.normalize_gsm_mc
normalize_math_mc = _ADAPTER.normalize_math_mc
normalize_mmlu_redux = _ADAPTER.normalize_mmlu_redux
validate_normalized_row = _ADAPTER.validate_normalized_row


def test_mmlu_redux_uses_corrected_groundtruth_and_skips_unscorable():
    base = {
        "question": "q",
        "choices": ["x", "y", "z", "w"],
        "answer": 0,
        "error_type": "wrong_groundtruth",
        "correct_answer": "C",
    }
    row = normalize_mmlu_redux(base, subject="math", split="test", source_index=3)
    assert row["answer_index"] == 2
    assert row["answer_label"] == "C"
    validate_normalized_row(row)
    base["error_type"] = "no_correct_answer"
    assert (
        normalize_mmlu_redux(base, subject="math", split="test", source_index=3) is None
    )
    base["error_type"] = "expert"
    assert (
        normalize_mmlu_redux(base, subject="math", split="test", source_index=3) is None
    )


def test_mmlu_redux_accepts_numeric_corrected_answer_strings():
    row = normalize_mmlu_redux(
        {
            "question": "q",
            "choices": ["x", "y", "z", "w"],
            "answer": 0,
            "error_type": "wrong_groundtruth",
            "correct_answer": "2",
        },
        subject="math",
        split="test",
        source_index=4,
    )
    assert row["answer_label"] == "C"


def test_arc_maps_numeric_source_labels_to_normalized_letters():
    row = normalize_arc(
        {
            "id": "abc",
            "question": "q",
            "choices": {"text": ["x", "y", "z", "w"], "label": ["1", "2", "3", "4"]},
            "answerKey": "3",
        },
        config="ARC-Easy",
        split="test",
        source_index=0,
    )
    assert row["answer_index"] == 2
    assert row["answer_label"] == "C"
    assert row["source_answer_key"] == "3"
    validate_normalized_row(row)


def test_arc_relabels_noncontiguous_source_labels_by_choice_position():
    row = normalize_arc(
        {
            "id": "noncontiguous",
            "question": "q",
            "choices": {
                "text": ["first", "second", "third"],
                "label": ["A", "C", "D"],
            },
            "answerKey": "C",
        },
        config="ARC-Challenge",
        split="test",
        source_index=0,
    )
    assert row["source_answer_key"] == "C"
    assert row["answer_index"] == 1
    assert row["answer_label"] == "B"


def test_prompt_and_parser_support_variable_option_count():
    row = {
        "question": "q",
        "choices": ["one", "two", "three", "four", "five"],
    }
    prompt = build_multiple_choice_prompt(row, use_cot=False)
    assert "E. five" in prompt
    assert "A/B/C/D/E" in prompt
    assert extract_choice("Reasoning. The correct answer is E.", option_count=5) == "E"
    assert (
        extract_choice("Reasoning. The correct answer is **C**.", option_count=5) == "C"
    )
    assert (
        extract_choice(
            "The correct answer is E7_{16}.",
            option_count=4,
            choices=["17_{16}", "E4_{16}", "E7_{16}", "F4_{16}"],
        )
        == "C"
    )
    assert extract_choice("F", option_count=5) is None


def test_plain_prompt_matches_option_logits_template():
    prompt = build_plain_multiple_choice_prompt(
        {
            "question": "q",
            "choices": ["one", "two", "three", "four"],
        }
    )
    assert "Choices:\nA. one\nB. two\nC. three\nD. four" in prompt
    assert (
        'Respond ONLY in the following format: "The correct answer is A/B/C/D".'
        in prompt
    )
    assert prompt.endswith("The correct answer is")


def test_dataset_aliases_are_stable():
    assert canonical_dataset_name("ARC-Easy") == "arc-e"
    assert canonical_dataset_name("arc_c") == "arc-c"
    assert canonical_dataset_name("gsm8k-mc") == "gsm-mc"
    assert canonical_dataset_name("gsm_mc") == "gsm-mc"
    assert canonical_dataset_name("math-mc") == "math-mc"


def test_gsm_mc_normalizes_upstream_release_row():
    row = normalize_gsm_mc(
        {
            "A": "22",
            "B": "64",
            "C": "18",
            "D": "12",
            "Answer": "C",
            "Question": "Janet's ducks lay 16 eggs per day.",
        },
        split="test",
        source_index=0,
    )
    assert row["dataset"] == "gsm-mc"
    assert row["subject"] == "GSM-MC"
    assert row["choices"] == ["22", "64", "18", "12"]
    assert row["answer_index"] == 2
    assert row["answer_label"] == "C"
    assert row["example_id"].startswith("gsm-mc:GSM-MC:test:")
    validate_normalized_row(row)


def test_math_mc_rewrites_identity_but_keeps_body():
    row = normalize_math_mc(
        {
            "A": "1",
            "B": "2",
            "C": "3",
            "D": "4",
            "Answer": "D",
            "Question": "q",
            "Level": "Level 4",
            "Type": "Counting & Probability",
        },
        split="test",
        source_index=5,
    )
    assert row["dataset"] == "math-mc"
    assert row["subject"] == "MATH-MC"
    assert row["answer_label"] == "D"
    assert row["source_level"] == "Level 4"
    assert row["source_type"] == "Counting & Probability"
    assert row["example_id"].startswith("math-mc:MATH-MC:test:")
    validate_normalized_row(row)
    gsm_row = normalize_gsm_mc(
        {"A": "1", "B": "2", "C": "3", "D": "4", "Answer": "D", "Question": "q"},
        split="test",
        source_index=5,
    )
    assert gsm_row["example_id"] != row["example_id"]


def test_gsm_mc_skips_rows_with_unscorable_options_or_answer():
    base = {"A": "1", "B": "2", "C": "3", "D": "4", "Answer": "A", "Question": "q"}
    assert normalize_gsm_mc(base, split="test", source_index=0) is not None
    empty_option = dict(base, A="")
    assert normalize_gsm_mc(empty_option, split="test", source_index=0) is None
    empty_question = dict(base, Question="  ")
    assert normalize_gsm_mc(empty_question, split="test", source_index=0) is None
    missing_answer = {key: value for key, value in base.items() if key != "Answer"}
    assert normalize_gsm_mc(missing_answer, split="test", source_index=0) is None
