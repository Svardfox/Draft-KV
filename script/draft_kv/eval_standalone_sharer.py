#!/usr/bin/env python3
"""Evaluate one frozen model with the canonical option-logit protocol.

This evaluator is deliberately independent of Draft-KV checkpoints and draft
generation.  It measures the standalone model that occupies the Sharer role,
using the same adapted datasets, plain prompt, chat-template suffix, and first
space-prefixed option-token scoring convention as the downstream Draft-KV
evaluator.  Outputs are resumable and bound to immutable manifests.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from draft_kv.train.downstream_mc_data import (  # noqa: E402
    canonical_dataset_name,
    validate_normalized_row,
)
from script.draft_kv.draft_kv_common import (  # noqa: E402
    load_causal_lm,
    prepare_tokenizer,
    seed_all,
    sha256_file,
)
from script.draft_kv.run_stage3_eval import (  # noqa: E402
    PLAIN_ANSWER_PREFIX,
    _atomic_json,
    _atomic_jsonl,
    _chat_ids,
    _collate_receiver,
    _configuration_sha256,
    _data_path,
    _dataset_manifest_entry,
    _default_split,
    _evaluation_prompt,
    _model_artifact_hashes,
    _option_token_ids,
    _predictions,
    _prompt_length_audit,
    _read_jsonl,
    _safe_tag,
    _selection_sha256,
)


PROTOCOL = "draft_kv_standalone_sharer_mc"
DEFAULT_DATASETS = "mmlu-redux,arc-e,arc-c,openbookqa,ceval,gsm-mc,math-mc"
DEFAULT_DATA_ROOT = "/workspace/datasets"
DEFAULT_OUTPUT_ROOT = (
    "/workspace/draft-kv/"
    "standalone_sharer_only"
)


def _parse_datasets(value: str) -> list[str]:
    datasets: list[str] = []
    seen: set[str] = set()
    for item in str(value).split(","):
        if not item.strip():
            continue
        dataset = canonical_dataset_name(item)
        if dataset in seen:
            raise ValueError(f"duplicate dataset: {dataset}")
        seen.add(dataset)
        datasets.append(dataset)
    if not datasets:
        raise ValueError("--datasets cannot be empty")
    return datasets


def _artifact_fingerprint(artifacts: Mapping[str, str]) -> str:
    payload = json.dumps(
        dict(artifacts), sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _existing_rows(
    path: Path, examples: Sequence[Mapping[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    stored = _read_jsonl(path) if path.exists() else []
    result: Dict[str, Dict[str, Any]] = {}
    by_id = {str(row["example_id"]): row for row in examples}
    for row in stored:
        example_id = str(row.get("example_id", ""))
        if not example_id or example_id in result or example_id not in by_id:
            raise RuntimeError(f"invalid/duplicate standalone row: {example_id!r}")
        source = by_id[example_id]
        valid_predictions = {
            chr(ord("A") + index) for index in range(len(source["choices"]))
        }
        if (
            row.get("condition") != "Standalone"
            or str(row.get("dataset")) != str(source["dataset"])
            or str(row.get("subject")) != str(source["subject"])
            or str(row.get("gold")) != str(source["answer_label"])
            or str(row.get("prediction")) not in valid_predictions
            or bool(row.get("correct"))
            != (str(row.get("prediction")) == str(row.get("gold")))
        ):
            raise RuntimeError(
                f"standalone row fails replay validation: {example_id}"
            )
        result[example_id] = dict(row)
    expected_ids = [str(row["example_id"]) for row in examples]
    if list(result) != expected_ids[: len(result)]:
        raise RuntimeError("standalone output is not an ordered manifest prefix")
    return result


def _summary(
    examples: Sequence[Mapping[str, Any]], rows: Mapping[str, Mapping[str, Any]]
) -> Dict[str, Any]:
    by_subject: Dict[str, list[bool]] = defaultdict(list)
    ordered_correct: list[bool] = []
    for example in examples:
        example_id = str(example["example_id"])
        correct = bool(rows[example_id]["correct"])
        ordered_correct.append(correct)
        by_subject[str(example["subject"])].append(correct)
    count = len(ordered_correct)
    correct_count = sum(ordered_correct)
    return {
        "accuracy": float(correct_count / count),
        "correct_count": int(correct_count),
        "count": int(count),
        "accuracy_by_subject": {
            subject: {
                "accuracy": float(sum(values) / len(values)),
                "correct_count": int(sum(values)),
                "count": int(len(values)),
            }
            for subject, values in sorted(by_subject.items())
        },
    }


def _forward_last_logits(
    model: Any, input_ids: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    """Return only the final-token logits when the architecture supports it."""

    parameters = inspect.signature(model.forward).parameters
    kwargs: Dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "use_cache": False,
    }
    if "logits_to_keep" in parameters:
        kwargs["logits_to_keep"] = 1
    elif "num_logits_to_keep" in parameters:
        kwargs["num_logits_to_keep"] = 1
    output = model(**kwargs)
    logits = output.logits
    if logits.ndim != 3 or logits.shape[1] < 1:
        raise RuntimeError(f"model returned invalid logits shape: {tuple(logits.shape)}")
    return logits[:, -1, :]


def _prepare_examples(
    *, dataset: str, data_root: Path, max_examples: int | None
) -> tuple[str, Path, list[Dict[str, Any]], Dict[str, Any]]:
    split = _default_split(dataset)
    data_path = _data_path(data_root, dataset, split)
    if not data_path.is_file():
        raise FileNotFoundError(f"missing adapted dataset: {data_path}")
    all_examples = _read_jsonl(data_path)
    for row in all_examples:
        validate_normalized_row(row)
        if str(row["dataset"]) != dataset:
            raise RuntimeError("adapted data contains the wrong dataset")
    all_ids = [str(row["example_id"]) for row in all_examples]
    if len(all_ids) != len(set(all_ids)):
        raise RuntimeError("adapted data contains duplicate IDs")
    registration = _dataset_manifest_entry(data_root, dataset, data_path)
    if int(registration["adapted_file"]["count"]) != len(all_examples):
        raise RuntimeError("adapted row count differs from its manifest")
    examples = all_examples[:max_examples] if max_examples is not None else all_examples
    if len(examples) < 1:
        raise RuntimeError(f"no selected examples for {dataset}")
    return split, data_path, examples, registration


@torch.inference_mode()
def _evaluate_dataset(
    *,
    model: Any,
    tokenizer: Any,
    examples: Sequence[Mapping[str, Any]],
    output: Path,
    batch_size: int,
    max_input_tokens: int,
    save_every_batches: int,
) -> Dict[str, Any]:
    output_path = output / "per_example.jsonl"
    rows = _existing_rows(output_path, examples)
    example_ids = [str(row["example_id"]) for row in examples]
    option_ids = _option_token_ids(tokenizer)
    for batch_index, start in enumerate(
        range(len(rows), len(examples), int(batch_size)), start=1
    ):
        chunk = examples[start : start + int(batch_size)]
        prompts = [
            _chat_ids(
                tokenizer,
                _evaluation_prompt(row, prompt_style="plain", use_cot=False),
                assistant_prefix=PLAIN_ANSWER_PREFIX,
            )
            for row in chunk
        ]
        longest = max(len(values) for values in prompts)
        if longest > int(max_input_tokens):
            culprit = chunk[prompts.index(max(prompts, key=len))]
            raise RuntimeError(
                "standalone prompt exceeds --max-input-tokens: "
                f"{culprit['example_id']} length={longest}"
            )
        ids, attention = _collate_receiver(
            prompts,
            pad_token_id=int(tokenizer.pad_token_id),
            device=model.device,
        )
        last_logits = _forward_last_logits(model, ids, attention)
        predictions, confidence = _predictions(
            last_logits[:, None, :],
            option_ids,
            [len(row["choices"]) for row in chunk],
        )
        for index, example in enumerate(chunk):
            example_id = str(example["example_id"])
            prediction = predictions[index]
            rows[example_id] = {
                "example_id": example_id,
                "dataset": str(example["dataset"]),
                "subject": str(example["subject"]),
                "condition": "Standalone",
                "prediction": prediction,
                "gold": str(example["answer_label"]),
                "correct": prediction == str(example["answer_label"]),
                "predicted_option_probability": confidence[index],
                "prompt_tokens": len(prompts[index]),
            }
        print(
            f"Standalone {examples[0]['dataset']} {len(rows)}/{len(examples)}",
            flush=True,
        )
        if batch_index % int(save_every_batches) == 0:
            _atomic_jsonl(output_path, [rows[value] for value in example_ids if value in rows])
    _atomic_jsonl(output_path, [rows[value] for value in example_ids if value in rows])
    if set(rows) != set(example_ids):
        raise RuntimeError("standalone evaluation ended with incomplete rows")
    return _summary(examples, rows)


def run(args: argparse.Namespace) -> Dict[str, Any]:
    model_path = Path(args.model).resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(f"missing model directory: {model_path}")
    datasets = _parse_datasets(args.datasets)
    data_root = Path(args.data_root).resolve()
    output_root = Path(args.output_root).resolve()
    model_artifacts = _model_artifact_hashes(str(model_path))
    model_fingerprint = _artifact_fingerprint(model_artifacts)
    model_tag = _safe_tag(model_path.name)
    model_output = output_root / "models" / model_tag / model_fingerprint[:12]
    tokenizer = prepare_tokenizer(str(model_path), padding_side="left")
    adapter_path = (
        Path(__file__).resolve().parents[2] / "draft_kv/train/downstream_mc_data.py"
    )
    downstream_path = Path(__file__).with_name("run_stage3_eval.py")
    jobs: list[Dict[str, Any]] = []
    results: Dict[str, Any] = {}

    for dataset in datasets:
        split, data_path, examples, registration = _prepare_examples(
            dataset=dataset,
            data_root=data_root,
            max_examples=args.max_examples,
        )
        example_ids = [str(row["example_id"]) for row in examples]
        selection_sha = _selection_sha256(example_ids)
        configuration = {
            "prompt_style": "plain",
            "assistant_prefix": PLAIN_ANSWER_PREFIX,
            "option_scoring": "first token of space-prefixed A-J",
            "eval_batch_size": int(args.eval_batch_size),
            "max_input_tokens": int(args.max_input_tokens),
        }
        configuration_sha = _configuration_sha256(configuration)
        output = model_output / dataset / (
            f"{selection_sha[:12]}-{configuration_sha[:12]}"
        )
        prompt_audit = _prompt_length_audit(
            tokenizer,
            examples,
            use_cot=False,
            assistant_prefix=PLAIN_ANSWER_PREFIX,
            limit=int(args.max_input_tokens),
            role="Standalone",
            prompt_style="plain",
        )
        if int(prompt_audit["over_limit_count"]) != 0:
            raise RuntimeError(
                f"{dataset} has prompts over --max-input-tokens: {prompt_audit}"
            )
        manifest = {
            "protocol": PROTOCOL,
            "dataset": dataset,
            "split": split,
            "data_path": str(data_path.resolve()),
            "data_sha256": sha256_file(data_path),
            "data_registration": registration,
            "example_count": len(examples),
            "example_ids": example_ids,
            "selection_sha256": selection_sha,
            "model": str(model_path),
            "model_artifact_sha256": model_artifacts,
            "model_fingerprint": model_fingerprint,
            "configuration": configuration,
            "configuration_sha256": configuration_sha,
            "prompt_length_audit": prompt_audit,
            "runtime_versions": {
                "python": sys.version.split()[0],
                "torch": str(torch.__version__),
            },
            "code_sha256": {
                "evaluator": sha256_file(Path(__file__)),
                "adapter": sha256_file(adapter_path),
                "downstream_prompt_helpers": sha256_file(downstream_path),
            },
        }
        output.mkdir(parents=True, exist_ok=True)
        manifest_path = output / "eval_manifest.json"
        if manifest_path.exists():
            observed = json.loads(manifest_path.read_text(encoding="utf-8"))
            if observed != manifest:
                raise RuntimeError(
                    f"existing standalone manifest differs from this command: {manifest_path}"
                )
        else:
            _atomic_json(manifest_path, manifest)
        existing = _existing_rows(output / "per_example.jsonl", examples)
        result_path = output / "eval_result.json"
        if result_path.is_file() and len(existing) == len(examples):
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if (
                result.get("protocol") != PROTOCOL
                or int(result.get("count", -1)) != len(examples)
                or result.get("model_fingerprint") != model_fingerprint
                or result.get("configuration_sha256") != configuration_sha
            ):
                raise RuntimeError(f"invalid completed result: {result_path}")
            print(f"Reusing completed {model_tag} {dataset}: {result_path}", flush=True)
            results[dataset] = result
            continue
        jobs.append(
            {
                "dataset": dataset,
                "split": split,
                "data_path": data_path,
                "examples": examples,
                "output": output,
                "manifest_path": manifest_path,
                "configuration_sha": configuration_sha,
                "prompt_audit": prompt_audit,
            }
        )

    model = None
    if jobs:
        seed_all(int(args.seed))
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        model = load_causal_lm(str(model_path), device)
        if any(parameter.requires_grad for parameter in model.parameters()):
            for parameter in model.parameters():
                parameter.requires_grad_(False)

    for job in jobs:
        summary = _evaluate_dataset(
            model=model,
            tokenizer=tokenizer,
            examples=job["examples"],
            output=job["output"],
            batch_size=int(args.eval_batch_size),
            max_input_tokens=int(args.max_input_tokens),
            save_every_batches=int(args.save_every_batches),
        )
        result = {
            "protocol": PROTOCOL,
            **summary,
            "condition": "Standalone",
            "dataset": job["dataset"],
            "split": job["split"],
            "model": str(model_path),
            "model_fingerprint": model_fingerprint,
            "data_path": str(job["data_path"].resolve()),
            "configuration_sha256": job["configuration_sha"],
            "prompt_length_audit": job["prompt_audit"],
            "output_dir": str(job["output"].resolve()),
            "eval_manifest_sha256": sha256_file(job["manifest_path"]),
            "output_artifact_sha256": {
                "eval_manifest.json": sha256_file(job["manifest_path"]),
                "per_example.jsonl": sha256_file(job["output"] / "per_example.jsonl"),
            },
        }
        _atomic_json(job["output"] / "eval_result.json", result)
        results[job["dataset"]] = result
        print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)

    combined = {
        "protocol": PROTOCOL,
        "model": str(model_path),
        "model_fingerprint": model_fingerprint,
        "datasets": {
            dataset: {
                "accuracy": float(results[dataset]["accuracy"]),
                "correct_count": int(results[dataset]["correct_count"]),
                "count": int(results[dataset]["count"]),
                "result": str(Path(results[dataset]["output_dir"]) / "eval_result.json"),
            }
            for dataset in datasets
        },
    }
    _atomic_json(model_output / "summary.json", combined)
    print(f"Standalone summary: {model_output / 'summary.json'}", flush=True)
    return combined


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="local Hugging Face model path")
    parser.add_argument("--datasets", default=DEFAULT_DATASETS)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--eval-batch-size", type=int, default=48)
    parser.add_argument("--max-input-tokens", type=int, default=1280)
    parser.add_argument("--save-every-batches", type=int, default=10)
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--seed", type=int, default=91827)
    parser.add_argument("--device", default="cuda:0")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    for name in ("eval_batch_size", "max_input_tokens", "save_every_batches"):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.max_examples is not None and int(args.max_examples) <= 0:
        raise ValueError("--max-examples must be positive")
    run(args)


if __name__ == "__main__":
    main()
