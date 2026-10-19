#!/usr/bin/env python3
"""Rebuild the prospective parent Gold summary from public package members."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

STUDY_ID = "crn-2026-003-finite-domain-dsa"
SUMMARY_MEMBER = Path(
    "prospective_collision_study/results/crn_ca_pr1_20261017/analysis/summary.json"
)
SUMMARY_SHA256 = "6f6166b6797f692375ceac43e675289ff5edc125ef16ffe7f945cda6b4af9f14"
PARENT_MANIFEST = Path(f"studies/{STUDY_ID}/PARENT_EVIDENCE_MANIFEST.json")
ARCHIVAL_SOURCE_SHA256 = "0235cc202ef9bfe16405f58c32e114c919803d2463b52188b7f82746ab5ee5a4"
DEFAULT_OUTPUT = Path(f"studies/{STUDY_ID}/evidence/gold/inference/parent_summary.json")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def verified_bytes(path: Path, expected_sha256: str, label: str) -> bytes:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{label} is absent or symbolic: {path}")
    payload = path.read_bytes()
    actual = sha256_bytes(payload)
    if actual != expected_sha256:
        raise ValueError(f"{label} SHA-256 mismatch: expected {expected_sha256}, got {actual}")
    return payload


def build_record(release_root: Path) -> dict[str, Any]:
    root = release_root.resolve()
    parent_manifest = json.loads((root / PARENT_MANIFEST).read_text(encoding="utf-8"))
    archival_source = parent_manifest.get("archival_build_source", {})
    if archival_source.get("sha256") != ARCHIVAL_SOURCE_SHA256:
        raise ValueError("parent-evidence manifest does not bind the sealed archival source")

    payload = verified_bytes(root / SUMMARY_MEMBER, SUMMARY_SHA256, "prospective summary")
    summary = json.loads(payload)
    return {
        "schema_version": 1,
        "study": "Study B: prospective collision-aware parent campaign",
        "derivation": "hash-verified deterministic field selection from sealed parent evidence",
        "input": {
            "release_origin": "Collision_Aware_DSA_Reproducibility_20261017.zip",
            "archive_sha256": ARCHIVAL_SOURCE_SHA256,
            "archive_member": SUMMARY_MEMBER.as_posix(),
            "member_sha256": sha256_bytes(payload),
        },
        "bootstrap": summary["bootstrap"],
        "confirmatory_gates": summary["confirmatory_gates"],
        "secondary_holm_family": summary["secondary_holm_family"],
    }


def render_record(record: dict[str, Any]) -> bytes:
    return (json.dumps(record, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode("utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else args.release_root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(render_record(build_record(args.release_root)))


if __name__ == "__main__":
    main()
