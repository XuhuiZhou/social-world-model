#!/usr/bin/env python3
"""Generate identifier-only manifests for the paper benchmark inputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
from typing import Any, Iterable


def canonical_hash(record: dict[str, Any]) -> str:
    payload = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def csv_records(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def jsonl_records(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def render_csv(fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def write_or_check(path: Path, content: str, check: bool) -> None:
    if check:
        if not path.exists() or path.read_text(encoding="utf-8") != content:
            raise SystemExit(f"manifest is stale or missing: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("reproducibility/splits")
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    sources = {
        "paratomi": args.data_root / "rephrased_tomi_test_600.csv",
        "tomi": args.data_root / "Percept-ToMi.csv",
        "fantom": args.data_root / "fantom_data/fantom_for_tt_processed.jsonl",
        "hitom": args.data_root / "hitom_data/processed_hitom_data100.csv",
        "mmtom": args.data_root / "mmtom-qa/questions.jsonl",
    }
    missing = [str(path) for path in sources.values() if not path.is_file()]
    if missing:
        raise SystemExit("missing benchmark inputs:\n" + "\n".join(missing))

    paratomi = csv_records(sources["paratomi"])
    tomi = csv_records(sources["tomi"])
    fantom = jsonl_records(sources["fantom"])
    hitom = csv_records(sources["hitom"])
    mmtom = jsonl_records(sources["mmtom"])

    expected = {"paratomi": 600, "tomi": 600, "fantom": 64, "hitom": 100, "mmtom": 600}
    actual = {
        "paratomi": len(paratomi),
        "tomi": len(tomi),
        "fantom": len(fantom),
        "hitom": len(hitom),
        "mmtom": len(mmtom),
    }
    if actual != expected:
        raise SystemExit(f"unexpected record counts: expected {expected}, found {actual}")

    simple_fields = ["source_row", "source_index", "record_sha256"]
    for name, records in (("tomi_600", tomi), ("paratomi_600", paratomi)):
        rows = [
            {
                "source_row": number,
                "source_index": record.get("index", ""),
                "record_sha256": canonical_hash(record),
            }
            for number, record in enumerate(records)
        ]
        write_or_check(
            args.output_dir / f"{name}.csv",
            render_csv(simple_fields, rows),
            args.check,
        )

    fantom_rows = [
        {
            "source_row": number,
            "set_id": record.get("set_id", ""),
            "conv_id": record.get("conv_id", ""),
            "part_id": record.get("part_id", ""),
            "record_sha256": canonical_hash(record),
        }
        for number, record in enumerate(fantom)
    ]
    write_or_check(
        args.output_dir / "fantom_64_conversations.csv",
        render_csv(
            ["source_row", "set_id", "conv_id", "part_id", "record_sha256"],
            fantom_rows,
        ),
        args.check,
    )

    hitom_rows = [
        {
            "source_row": number,
            "source_index": record.get("index", ""),
            "set_id": record.get("set_id", ""),
            "question_order": record.get("question_order", ""),
            "record_sha256": canonical_hash(record),
        }
        for number, record in enumerate(hitom)
    ]
    write_or_check(
        args.output_dir / "hitom_100.csv",
        render_csv(
            ["source_row", "source_index", "set_id", "question_order", "record_sha256"],
            hitom_rows,
        ),
        args.check,
    )

    mmtom_rows = [
        {
            "source_row": number,
            "episode": record.get("episode", ""),
            "start_time": record.get("start_time", ""),
            "end_time": record.get("end_time", ""),
            "question_type": record.get("question_type", ""),
            "record_sha256": canonical_hash(record),
        }
        for number, record in enumerate(mmtom)
    ]
    write_or_check(
        args.output_dir / "mmtom_600_candidates.csv",
        render_csv(
            [
                "source_row",
                "episode",
                "start_time",
                "end_time",
                "question_type",
                "record_sha256",
            ],
            mmtom_rows,
        ),
        args.check,
    )

    source_rows = [
        {
            "benchmark": name,
            "relative_path": str(path.relative_to(args.data_root)),
            "records": actual[name],
            "bytes": path.stat().st_size,
            "file_sha256": file_hash(path),
        }
        for name, path in sources.items()
    ]
    write_or_check(
        args.output_dir / "source_file_checksums.csv",
        render_csv(
            ["benchmark", "relative_path", "records", "bytes", "file_sha256"],
            source_rows,
        ),
        args.check,
    )


if __name__ == "__main__":
    main()
