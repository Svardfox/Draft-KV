#!/usr/bin/env python3
"""Collect Stage 3 multi-Sharer Matched/Deranged results.

Reads each ``stage3_mc_training_*`` run below
``--run-root``, binds every ``eval_result.json`` to that run's ``best.pt``
SHA256 and prints one table row per configuration and dataset.  The output is
the Matched/Deranged slice that feeds the master results table; Sharer-only,
Receiver-only, Draft-only and Text2Text columns are reused from existing
evaluations and are deliberately not recomputed here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

DEFAULT_RUN_ROOT = Path("/workspace/draft-kv-runs")
DEFAULT_DATASETS = ("mmlu-redux", "arc-e", "arc-c", "openbookqa", "ceval")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def find_runs(run_root: Path, patterns: Sequence[str]) -> list[Path]:
    runs: list[Path] = []
    for pattern in patterns:
        runs.extend(sorted(path for path in run_root.glob(pattern) if path.is_dir()))
    unique: Dict[Path, None] = {}
    for run in runs:
        unique[run.resolve()] = None
    return sorted(unique)


def bind_result(
    matched_root: Path,
    dataset: str,
    checkpoint: Path,
    checkpoint_sha: str,
) -> tuple[Dict[str, Any], Path] | None:
    pair_dirs = sorted(
        (matched_root / "evaluations" / dataset).glob("*_to_*")
    )
    candidates: list[tuple[float, Path, Dict[str, Any]]] = []
    for pair_dir in pair_dirs:
        for path in pair_dir.glob(f"{checkpoint_sha[:12]}/*/eval_result.json"):
            try:
                result = read_json(path)
            except Exception:
                continue
            if (
                result.get("checkpoint_sha256") == checkpoint_sha
                and Path(str(result.get("checkpoint", ""))).resolve() == checkpoint
            ):
                candidates.append((path.stat().st_mtime, path, result))
    if not candidates:
        return None
    _, path, result = max(candidates)
    return result, path


def format_percent(value: Any) -> str:
    if value is None:
        return "--"
    return f"{float(value) * 100:.2f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument(
        "--run-pattern",
        action="append",
        default=None,
        help="glob below --run-root (default: stage3_mc_training_*)",
    )
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument(
        "--matched-root",
        type=Path,
        default=None,
        help="evaluation tree (default: <run-root>/downstream_mc)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit JSON instead of a Markdown table",
    )
    args = parser.parse_args()

    run_root = args.run_root.resolve()
    matched_root = (
        args.matched_root.resolve()
        if args.matched_root
        else run_root / "downstream_mc"
    )
    patterns = args.run_pattern or [
        "stage3_mc_training_*"
    ]
    datasets = [value.strip() for value in args.datasets.split(",") if value.strip()]

    payload: list[Dict[str, Any]] = []
    for run in find_runs(run_root, patterns):
        train_result = run / "train" / "train_result.json"
        checkpoint = run / "train" / "best.pt"
        if not train_result.is_file() or not checkpoint.is_file():
            print(f"# skipping incomplete run: {run}")
            continue
        result = read_json(train_result)
        checkpoint_sha = sha256_file(checkpoint)
        sharer = str(result.get("sharer", ""))
        receiver = str(result.get("receiver", ""))
        if not sharer or not receiver:
            import torch

            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            sharer = sharer or str(state.get("sharer", ""))
            receiver = receiver or str(state.get("receiver", ""))
        record: Dict[str, Any] = {
            "run": str(run),
            "sharer": Path(sharer).name,
            "receiver": Path(receiver).name,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_sha,
            "mc_updates": int(result.get("mc_updates", 0)),
            "protection_weight": result.get("protection_weight"),
            "protection_tolerance": result.get("protection_tolerance"),
            "datasets": {},
        }
        for dataset in datasets:
            bound = bind_result(matched_root, dataset, checkpoint, checkpoint_sha)
            if bound is None:
                record["datasets"][dataset] = {"status": "missing"}
                continue
            values, path = bound
            accuracy: Mapping[str, float] = values.get("accuracy", {})
            record["datasets"][dataset] = {
                "status": "ok",
                "matched": accuracy.get("Matched"),
                "deranged": accuracy.get("Deranged"),
                "zero": accuracy.get("Zero"),
                "count": values.get("count"),
                "result": str(path),
            }
        payload.append(record)

    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return

    header = (
        f"{'Sharer':<24}{'dataset':<12}{'Matched':>9}{'Deranged':>10}"
        f"{'M-D':>9}{'n':>7}  {'checkpoint':<12}"
    )
    print(header)
    print("-" * len(header))
    for record in payload:
        for dataset in datasets:
            entry = record["datasets"][dataset]
            if entry.get("status") != "ok":
                continue
            matched = entry.get("matched")
            deranged = entry.get("deranged")
            gap = (
                f"{(float(matched) - float(deranged)) * 100:+9.2f}"
                if matched is not None and deranged is not None
                else f"{'--':>9}"
            )
            print(
                f"{record['sharer']:<24}{dataset:<12}"
                f"{format_percent(matched):>9}{format_percent(deranged):>10}"
                f"{gap}{int(entry.get('count') or 0):7d}"
                f"  {record['checkpoint_sha256'][:12]}"
            )


if __name__ == "__main__":
    main()
