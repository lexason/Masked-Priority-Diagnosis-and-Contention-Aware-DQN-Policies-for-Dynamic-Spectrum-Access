"""Public-package controls for the generated article-result interface.

This variant deliberately avoids the private manuscript consumer while retaining
Gold regeneration, manifest, uniqueness, claim-registry, and privacy checks.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
from pathlib import Path

import yaml

ANALYSIS_ROOT = Path(__file__).resolve().parent
STUDY_ROOT = ANALYSIS_ROOT.parent
GENERATOR_PATH = ANALYSIS_ROOT / "generate_result_macros.py"
SPEC = importlib.util.spec_from_file_location("crn_result_macros", GENERATOR_PATH)
assert SPEC is not None
assert SPEC.loader is not None
generator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(generator)


MACRO_PATH = STUDY_ROOT / "evidence/gold/manuscript/result_macros.tex"
MANIFEST_PATH = STUDY_ROOT / "evidence/gold/manuscript/result_macros.manifest.json"
CLAIM_REGISTRY = STUDY_ROOT / "manuscript/claims.yaml"


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def test_generated_macro_artifact_is_current_and_hash_bound() -> None:
    expected = generator.render_macros(generator.build_macros())
    actual = MACRO_PATH.read_bytes()
    assert actual == expected

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest["artifact_sha256"] == _sha256(actual)
    assert manifest["generator_sha256"] == generator.sha256_file(generator.Path(generator.__file__))
    for source in manifest["sources"]:
        assert generator.sha256_file(STUDY_ROOT / source["path"]) == source["sha256"]


def test_result_macros_are_unique_and_registered_to_claims() -> None:
    definitions = {
        item.decode("ascii")
        for item in re.findall(rb"\\newcommand\{(\\Result[A-Za-z]+)\}", MACRO_PATH.read_bytes())
    }
    definition_records = re.findall(
        rb"\\newcommand\{(\\Result[A-Za-z]+)\}", MACRO_PATH.read_bytes()
    )
    assert len(definition_records) == len(definitions)

    registry = yaml.safe_load(CLAIM_REGISTRY.read_text(encoding="utf-8"))
    registered = {
        "\\" + macro for claim in registry["claims"] for macro in claim.get("value_macros", [])
    }
    assert len(definitions) == 157
    assert definitions == registered


def test_public_controls_do_not_embed_private_paths() -> None:
    prohibited = (
        "Editorial" + "_Review",
        "SIX" + "_SIGMA_RESEARCH_PROTOCOL",
        "/" + "home/",
        "/" + "tmp/",
        "main" + "_archive/",
    )
    for path in (MACRO_PATH, MANIFEST_PATH, CLAIM_REGISTRY):
        text = path.read_text(encoding="utf-8")
        assert not [token for token in prohibited if token in text]
