#!/usr/bin/env python3
"""Create a fixed subject-stratified MMLU calibration/report split."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence


PROTOCOL = "draft_kv_mmlu_calibration_split"
DEFAULT_DATA_ROOT = "/workspace/datasets"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _selection_sha256(example_ids: Sequence[str]) -> str:
    payload = json.dumps(list(example_ids), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_example_ids_file(path: Path) -> tuple[list[str], Dict[str, Any]]:
    """Load an ordered JSON or line-oriented example-ID selection."""

    if not path.is_file():
        raise FileNotFoundError(f"missing example IDs file: {path}")
    text = path.read_text(encoding="utf-8")
    metadata: Dict[str, Any] = {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, Mapping):
        raw_ids = parsed.get("example_ids")
        if not isinstance(raw_ids, list):
            raise ValueError("example IDs JSON object must contain an example_ids list")
        metadata = dict(parsed)
    elif isinstance(parsed, list):
        raw_ids = parsed
    elif parsed is None:
        raw_ids = [
            line.strip()
            for line in text.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    else:
        raise ValueError(
            "example IDs file must be a JSON object/list or one ID per text line"
        )
    example_ids = [str(value).strip() for value in raw_ids]
    if not example_ids or any(not value for value in example_ids):
        raise ValueError("example IDs file contains no IDs or an empty ID")
    if len(example_ids) != len(set(example_ids)):
        raise ValueError("example IDs file contains duplicate IDs")
    registered_count = metadata.get("example_count")
    if registered_count is not None and int(registered_count) != len(example_ids):
        raise RuntimeError("example IDs file count does not match example_ids")
    registered_selection = metadata.get("selection_sha256")
    selection_sha = _selection_sha256(example_ids)
    if registered_selection is not None and str(registered_selection) != selection_sha:
        raise RuntimeError("example IDs file selection SHA256 is invalid")
    return example_ids, metadata


def select_examples_by_id(
    examples: Sequence[Mapping[str, Any]], requested_ids: Sequence[str]
) -> list[Dict[str, Any]]:
    """Select exactly the requested IDs and preserve the ID-file order."""

    by_id: Dict[str, Dict[str, Any]] = {}
    for raw in examples:
        row = dict(raw)
        example_id = str(row["example_id"])
        if example_id in by_id:
            raise RuntimeError("adapted data contains duplicate IDs")
        by_id[example_id] = row
    missing = [example_id for example_id in requested_ids if example_id not in by_id]
    if missing:
        preview = ", ".join(missing[:10])
        raise ValueError(
            f"example IDs file contains {len(missing)} IDs absent from the dataset: "
            f"{preview}"
        )
    return [by_id[example_id] for example_id in requested_ids]


def example_ids_file_binding(
    path: Path,
    metadata: Mapping[str, Any],
    example_ids: Sequence[str],
    *,
    dataset: str,
    data_path: Path,
    data_sha256: str,
    data_registration: Mapping[str, Any],
) -> Dict[str, Any]:
    """Validate optional source bindings and return manifest-ready provenance."""

    if metadata:
        registered_dataset = metadata.get("dataset")
        if registered_dataset is not None and str(registered_dataset) != dataset:
            raise RuntimeError("example IDs file is bound to a different dataset")
        registered_data_sha = metadata.get("source_data_sha256")
        if registered_data_sha is not None and str(registered_data_sha) != data_sha256:
            raise RuntimeError("example IDs file is bound to different adapted data")
        registered_data_path = metadata.get("source_data_path")
        if (
            registered_data_path is not None
            and Path(str(registered_data_path)).resolve() != data_path.resolve()
        ):
            raise RuntimeError("example IDs file is bound to a different data path")
        registered_manifest_sha = metadata.get("source_manifest_sha256")
        observed_manifest_sha = str(data_registration["manifest_sha256"])
        if (
            registered_manifest_sha is not None
            and str(registered_manifest_sha) != observed_manifest_sha
        ):
            raise RuntimeError(
                "example IDs file is bound to a different dataset manifest"
            )
    return {
        "path": str(path.resolve()),
        "sha256": _sha256_file(path),
        "selection_sha256": _selection_sha256(example_ids),
        "example_count": len(example_ids),
        "protocol": metadata.get("protocol"),
        "split": metadata.get("split"),
        "split_manifest_sha256": metadata.get("split_manifest_sha256"),
    }


def _read_jsonl(path: Path) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(f"invalid JSONL at {path}:{line_number}") from error
            if not isinstance(row, Mapping):
                raise RuntimeError(f"non-object JSONL row at {path}:{line_number}")
            rows.append(dict(row))
    return rows


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(value), indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _rank(seed: int, subject: str, example_id: str) -> str:
    return hashlib.sha256(
        f"{int(seed)}\0{subject}\0{example_id}".encode("utf-8")
    ).hexdigest()


def subject_stratified_split(
    rows: Sequence[Mapping[str, Any]],
    *,
    calibration_size: int,
    seed: int,
) -> tuple[list[str], list[str], Dict[str, Dict[str, int]]]:
    """Return ordered calibration/report IDs using capped Hamilton allocation."""

    if calibration_size <= 0 or calibration_size >= len(rows):
        raise ValueError("calibration_size must be between 1 and row_count - 1")
    by_subject: Dict[str, list[str]] = defaultdict(list)
    seen: set[str] = set()
    dataset_order: list[str] = []
    for row in rows:
        example_id = str(row.get("example_id", "")).strip()
        subject = str(row.get("subject", "")).strip()
        if not example_id or not subject:
            raise ValueError("every row must have non-empty example_id and subject")
        if example_id in seen:
            raise ValueError(f"duplicate example ID: {example_id}")
        seen.add(example_id)
        dataset_order.append(example_id)
        by_subject[subject].append(example_id)

    total = len(rows)
    quotas = {
        subject: calibration_size * len(ids) / total
        for subject, ids in by_subject.items()
    }
    # Keep every represented subject in the report split.
    capacities = {subject: max(0, len(ids) - 1) for subject, ids in by_subject.items()}
    if sum(capacities.values()) < calibration_size:
        raise ValueError("calibration_size leaves no report example for some subject")
    allocation = {
        subject: min(int(quotas[subject]), capacities[subject])
        for subject in by_subject
    }
    remaining = calibration_size - sum(allocation.values())
    priority = sorted(
        by_subject,
        key=lambda subject: (
            -(quotas[subject] - int(quotas[subject])),
            _rank(seed, subject, "__allocation__"),
            subject,
        ),
    )
    while remaining:
        progressed = False
        for subject in priority:
            if allocation[subject] >= capacities[subject]:
                continue
            allocation[subject] += 1
            remaining -= 1
            progressed = True
            if not remaining:
                break
        if not progressed:
            raise RuntimeError("unable to allocate the requested calibration size")

    calibration_set: set[str] = set()
    for subject, ids in sorted(by_subject.items()):
        ranked = sorted(ids, key=lambda value: (_rank(seed, subject, value), value))
        calibration_set.update(ranked[: allocation[subject]])
    # Preserve normalized dataset order in both files.  The evaluator then uses
    # the ID-file order exactly, making batching and derangement reproducible.
    calibration_ids = [
        example_id for example_id in dataset_order if example_id in calibration_set
    ]
    report_ids = [
        example_id for example_id in dataset_order if example_id not in calibration_set
    ]
    subject_counts = {
        subject: {
            "total": len(ids),
            "calibration": allocation[subject],
            "report": len(ids) - allocation[subject],
        }
        for subject, ids in sorted(by_subject.items())
    }
    if len(calibration_ids) != calibration_size:
        raise AssertionError("stratified allocation did not reach calibration_size")
    return calibration_ids, report_ids, subject_counts


def _dataset_registration(data_root: Path, data_path: Path) -> Dict[str, Any]:
    dataset_root = data_root / "mmlu-redux-2.0"
    manifest_path = dataset_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing dataset manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    relative = str(data_path.relative_to(dataset_root))
    entries = [
        dict(row)
        for row in manifest.get("files", [])
        if str(row.get("path")) == relative
    ]
    if len(entries) != 1:
        raise RuntimeError(f"dataset manifest does not uniquely register {relative}")
    entry = entries[0]
    if str(entry.get("sha256")) != _sha256_file(data_path):
        raise RuntimeError("adapted MMLU file differs from its dataset manifest")
    return {
        "path": str(manifest_path.resolve()),
        "sha256": _sha256_file(manifest_path),
        "adapted_file": entry,
    }


def create_split(
    *,
    data_root: Path,
    output_dir: Path,
    calibration_size: int,
    seed: int,
) -> Dict[str, Any]:
    data_root = data_root.resolve()
    output_dir = output_dir.resolve()
    data_path = data_root / "mmlu-redux-2.0" / "adapted" / "test.jsonl"
    if not data_path.is_file():
        raise FileNotFoundError(f"missing adapted MMLU data: {data_path}")
    rows = _read_jsonl(data_path)
    if any(str(row.get("dataset")) != "mmlu-redux" for row in rows):
        raise RuntimeError("adapted file contains a non-MMLU row")
    registration = _dataset_registration(data_root, data_path)
    if int(registration["adapted_file"].get("count", -1)) != len(rows):
        raise RuntimeError("adapted row count differs from its dataset manifest")
    calibration_ids, report_ids, subject_counts = subject_stratified_split(
        rows,
        calibration_size=int(calibration_size),
        seed=int(seed),
    )
    paths = {
        "calibration": output_dir / "calibration_ids.json",
        "report": output_dir / "report_ids.json",
        "manifest": output_dir / "split_manifest.json",
    }
    if any(path.exists() for path in paths.values()):
        raise RuntimeError("split output exists; use a new directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    source_common = {
        "protocol": PROTOCOL,
        "dataset": "mmlu-redux",
        "source_data_path": str(data_path.resolve()),
        "source_data_sha256": _sha256_file(data_path),
        "source_manifest_path": registration["path"],
        "source_manifest_sha256": registration["sha256"],
        "seed": int(seed),
        "stratification": "subject_proportional_capped_hamilton_sha256_rank",
    }
    for split, ids in (("calibration", calibration_ids), ("report", report_ids)):
        _write_json(
            paths[split],
            {
                **source_common,
                "split": split,
                "example_count": len(ids),
                "selection_sha256": _selection_sha256(ids),
                "example_ids": ids,
            },
        )
    manifest = {
        **source_common,
        "source_example_count": len(rows),
        "calibration_size": len(calibration_ids),
        "report_size": len(report_ids),
        "subject_counts": subject_counts,
        "files": {
            split: {
                "path": str(paths[split].resolve()),
                "sha256": _sha256_file(paths[split]),
                "example_count": len(ids),
                "selection_sha256": _selection_sha256(ids),
            }
            for split, ids in (("calibration", calibration_ids), ("report", report_ids))
        },
    }
    _write_json(paths["manifest"], manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--calibration-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260827)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = create_split(
        data_root=Path(args.data_root),
        output_dir=Path(args.output_dir),
        calibration_size=args.calibration_size,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
