#!/usr/bin/env python3
"""Download and normalize MMLU-Redux, ARC-Easy, and ARC-Challenge.

Large files are written below the shared data root, never into the repository.
Every JSONL file is atomically replaced and authenticated in a manifest.
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:
    from draft_kv.train.downstream_mc_data import (  # noqa: E402
        normalize_arc,
        normalize_gsm_mc,
        normalize_math_mc,
        normalize_mmlu_redux,
        validate_normalized_row,
    )
except ModuleNotFoundError as error:
    # Dataset download/normalization is CPU-only.  Some storage hosts do not
    # have torch even though the runtime environment does; loading this
    # one pure adapter module directly avoids importing draft_kv.train's eager
    # GPU training dependencies on those hosts.
    if error.name != "torch":
        raise
    adapter_path = (
        Path(__file__).resolve().parents[2] / "draft_kv/train/downstream_mc_data.py"
    )
    spec = importlib.util.spec_from_file_location(
        "draft_kv_downstream_mc_data", adapter_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load downstream adapter: {adapter_path}") from error
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    normalize_arc = adapter.normalize_arc
    normalize_gsm_mc = adapter.normalize_gsm_mc
    normalize_math_mc = adapter.normalize_math_mc
    normalize_mmlu_redux = adapter.normalize_mmlu_redux
    validate_normalized_row = adapter.validate_normalized_row


MMLU_SUBJECTS = (
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
    "human_aging",
    "human_sexuality",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "machine_learning",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "moral_disputes",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions",
)

DEFAULT_DATA_ROOT = "/workspace/datasets"
DEFAULT_CACHE = "/workspace/huggingface"

GSM_MC_URL = (
    "https://raw.githubusercontent.com/Geralt-Targaryen/MC-Evaluation/main/data.tar.gz"
)
GSM_MC_SHA256 = "dd64206bf3e07ce4622c74af680f571d640c402a5906f7f0b8fa8822b5fdda57"
GSM_MC_SOURCES = {
    "gsm-mc": ("gsm8k-mc", normalize_gsm_mc),
    "math-mc": ("math-mc", normalize_math_mc),
}


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"cannot JSON-encode {type(value).__name__}")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(dict(row), ensure_ascii=False, default=_json_default) + "\n"
            )
            count += 1
    temporary.replace(path)
    return count


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(value), indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_entry(path: Path, root: Path, count: int) -> Dict[str, Any]:
    return {
        "path": str(path.relative_to(root)),
        "count": int(count),
        "bytes": int(path.stat().st_size),
        "sha256": _sha256(path),
    }


def _load_dataset(*args: Any, cache_dir: str, **kwargs: Any) -> Any:
    # Delayed import keeps adapter unit tests independent from HF/pyarrow.
    from datasets import load_dataset

    return load_dataset(*args, cache_dir=cache_dir, **kwargs)


def download_mmlu_redux(root: Path, cache_dir: str) -> Dict[str, Any]:
    dataset_root = root / "mmlu-redux-2.0"
    files: list[Dict[str, Any]] = []
    adapted: list[Dict[str, Any]] = []
    skipped = 0
    for position, subject in enumerate(MMLU_SUBJECTS, start=1):
        bundle = _load_dataset(
            "edinburgh-dawg/mmlu-redux-2.0", subject, cache_dir=cache_dir
        )
        for split, data in bundle.items():
            raw_path = dataset_root / "raw" / subject / f"{split}.jsonl"
            raw_count = _write_jsonl(raw_path, (dict(row) for row in data))
            files.append(_file_entry(raw_path, dataset_root, raw_count))
            if split == "test":
                for index, row in enumerate(data):
                    normalized = normalize_mmlu_redux(
                        row,
                        subject=subject,
                        split=split,
                        source_index=index,
                    )
                    if normalized is None:
                        skipped += 1
                    else:
                        validate_normalized_row(normalized)
                        adapted.append(normalized)
        print(
            f"mmlu-redux subjects {position}/{len(MMLU_SUBJECTS)} "
            f"adapted={len(adapted)} skipped={skipped}",
            flush=True,
        )
    adapted_path = dataset_root / "adapted" / "test.jsonl"
    adapted_count = _write_jsonl(adapted_path, adapted)
    files.append(_file_entry(adapted_path, dataset_root, adapted_count))
    manifest = {
        "format": "draft_kv_downstream_mc",
        "dataset": "mmlu-redux",
        "source_dataset": "edinburgh-dawg/mmlu-redux-2.0",
        "source_configs": list(MMLU_SUBJECTS),
        "evaluation_split": "test",
        "adapted_count": adapted_count,
        "skipped_unscorable_count": skipped,
        "files": files,
        "downloaded_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    _write_json(dataset_root / "manifest.json", manifest)
    return manifest


def download_arc(root: Path, cache_dir: str) -> Dict[str, Any]:
    dataset_root = root / "ai2_arc"
    manifests: Dict[str, Any] = {}
    for dataset, config in (("arc-e", "ARC-Easy"), ("arc-c", "ARC-Challenge")):
        bundle = _load_dataset("allenai/ai2_arc", config, cache_dir=cache_dir)
        files: list[Dict[str, Any]] = []
        adapted: list[Dict[str, Any]] = []
        for split, data in bundle.items():
            raw_path = dataset_root / "raw" / config / f"{split}.jsonl"
            raw_count = _write_jsonl(raw_path, (dict(row) for row in data))
            files.append(_file_entry(raw_path, dataset_root, raw_count))
            if split == "test":
                for index, row in enumerate(data):
                    normalized = normalize_arc(
                        row, config=config, split=split, source_index=index
                    )
                    validate_normalized_row(normalized)
                    adapted.append(normalized)
        adapted_path = dataset_root / "adapted" / dataset / "test.jsonl"
        adapted_count = _write_jsonl(adapted_path, adapted)
        files.append(_file_entry(adapted_path, dataset_root, adapted_count))
        manifests[dataset] = {
            "format": "draft_kv_downstream_mc",
            "dataset": dataset,
            "source_dataset": "allenai/ai2_arc",
            "source_config": config,
            "evaluation_split": "test",
            "adapted_count": adapted_count,
            "files": files,
            "downloaded_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        }
        print(f"{dataset} adapted={adapted_count}", flush=True)
    _write_json(
        dataset_root / "manifest.json",
        {
            "format": "draft_kv_downstream_mc",
            "source_dataset": "allenai/ai2_arc",
            "datasets": manifests,
        },
    )
    return manifests


def _download_url(url: str, destination: Path) -> None:
    """Fetch ``url`` to ``destination`` (urllib first, curl as proxy fallback)."""

    import shutil
    import subprocess
    import urllib.request

    try:
        urllib.request.urlretrieve(url, destination)
        return
    except Exception as error:  # pragma: no cover - environment dependent
        print(f"urllib download failed ({error}); retrying with curl", flush=True)
    curl = shutil.which("curl")
    if curl is None:
        raise RuntimeError(f"cannot download {url}: urllib failed and curl missing")
    subprocess.run(
        [curl, "-fL", "--retry", "3", "-o", str(destination), url],
        check=True,
    )


def download_gsm_mc(root: Path, cache_dir: str) -> Dict[str, Any]:
    """Materialize GSM-MC/MATH-MC from the pinned MC-Evaluation data release.

    The upstream tarball ships raw 4-way MC rows (``A/B/C/D/Answer/Question``)
    plus distractor pools.  Raw rows are copied verbatim; adapted rows reuse
    the shared normalizer so rows whose options were dropped by upstream LaTeX
    rendering are skipped and recorded instead of silently corrupting counts.
    """

    import tarfile
    import urllib.request

    dataset_root = root / "mc-evaluation"
    staging = dataset_root / ".staging"
    staging.mkdir(parents=True, exist_ok=True)
    archive_path = staging / "data.tar.gz"
    if not archive_path.is_file() or _sha256(archive_path) != GSM_MC_SHA256:
        print(f"downloading {GSM_MC_URL}", flush=True)
        _download_url(GSM_MC_URL, archive_path)
        observed = _sha256(archive_path)
        if observed != GSM_MC_SHA256:
            raise RuntimeError(
                f"MC-Evaluation tarball SHA256 mismatch: {observed} != {GSM_MC_SHA256}"
            )
    raw_root = staging / "data"
    if not raw_root.is_dir():
        with tarfile.open(archive_path, "r:gz") as handle:
            handle.extractall(staging)
    datasets_manifest: Dict[str, Any] = {}
    for dataset, (source_folder, normalizer) in GSM_MC_SOURCES.items():
        dataset_out = dataset_root / dataset
        files: list[Dict[str, Any]] = []
        manifests: Dict[str, Any] = {}
        for split in ("test", "train"):
            source_path = raw_root / source_folder / f"{split}.jsonl"
            raw_rows = [
                json.loads(line)
                for line in source_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            raw_path = dataset_out / "raw" / f"{split}.jsonl"
            raw_count = _write_jsonl(raw_path, raw_rows)
            files.append(_file_entry(raw_path, dataset_root, raw_count))
            adapted: list[Dict[str, Any]] = []
            skipped = 0
            for index, row in enumerate(raw_rows):
                normalized = normalizer(row, split=split, source_index=index)
                if normalized is None:
                    skipped += 1
                else:
                    validate_normalized_row(normalized)
                    adapted.append(normalized)
            if split == "test":
                adapted_path = dataset_out / "adapted" / "test.jsonl"
                adapted_count = _write_jsonl(adapted_path, adapted)
                files.append(_file_entry(adapted_path, dataset_root, adapted_count))
            manifests[split] = {
                "raw_count": raw_count,
                "adapted_count": len(adapted),
                "skipped_unscorable_count": skipped,
            }
            print(
                f"{dataset} {split} raw={raw_count} adapted={len(adapted)} "
                f"skipped={skipped}",
                flush=True,
            )
        manifests["files"] = files
        datasets_manifest[dataset] = manifests
    _write_json(
        dataset_root / "manifest.json",
        {
            "format": "draft_kv_downstream_mc",
            "source": "Geralt-Targaryen/MC-Evaluation",
            "source_archive_sha256": GSM_MC_SHA256,
            "source_url": GSM_MC_URL,
            "datasets": datasets_manifest,
        },
    )
    return datasets_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE)
    args = parser.parse_args()
    root = Path(args.data_root).resolve()
    cache = Path(args.cache_dir).resolve()
    # Any location outside the repository is fine; the point is to keep
    # multi-gigabyte datasets out of version-controlled directories.
    repo_root = Path(__file__).resolve().parents[2]
    if root == repo_root or repo_root in root.parents:
        raise ValueError(f"--data-root must be outside the repository: {repo_root}")
    root.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache))
    os.environ.setdefault("HF_DATASETS_CACHE", str(cache / "datasets"))
    result = {
        "format": "draft_kv_downstream_mc_download",
        "data_root": str(root),
        "huggingface_cache": str(cache),
        "mmlu-redux": download_mmlu_redux(root, str(cache / "datasets")),
        "arc": download_arc(root, str(cache / "datasets")),
        "gsm-mc": download_gsm_mc(root, str(cache / "datasets")),
    }
    _write_json(root / "draft_kv_downstream_mc_manifest.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
