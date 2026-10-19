"""Smoke and integrity tests for the publication-figure derivatives."""

from __future__ import annotations

import importlib.util
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest


HERE = Path(__file__).resolve().parent
GENERATOR_PATH = HERE / "generate_publication_figures.py"
SPEC = importlib.util.spec_from_file_location("crn_publication_figures", GENERATOR_PATH)
assert SPEC is not None and SPEC.loader is not None
FIGURES = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FIGURES)

ARCHITECTURE_GENERATOR_PATH = HERE / "generate_architecture_figure.py"
ARCHITECTURE_SPEC = importlib.util.spec_from_file_location(
    "crn_architecture_figure", ARCHITECTURE_GENERATOR_PATH
)
assert ARCHITECTURE_SPEC is not None and ARCHITECTURE_SPEC.loader is not None
ARCHITECTURE = importlib.util.module_from_spec(ARCHITECTURE_SPEC)
ARCHITECTURE_SPEC.loader.exec_module(ARCHITECTURE)


def test_hash_pins_and_redundant_gold_exports_validate() -> None:
    data = FIGURES.load_and_validate()
    assert data["panel_counts"] == {"a": 29, "b": 29, "c": 72, "d": 20}
    assert len(data["reanalysis"]["attempted_only_logical_audit"]["per_checkpoint"]) == 20
    assert data["reanalysis"]["coordinate_positive_control"]["case_count"] == 378


def test_rendering_is_byte_deterministic() -> None:
    data = FIGURES.load_and_validate()
    first = FIGURES.render_all(data)
    second = FIGURES.render_all(data)
    assert first.keys() == second.keys()
    assert first == second
    assert all(payload.startswith(b"%PDF-") for payload in first.values())


def test_checked_artifacts_match_in_memory_render() -> None:
    hashes = FIGURES.generate(output_dir=FIGURES.DEFAULT_OUTPUT_DIR, check=True)
    assert set(hashes) == {
        "fig_archived_evidence.pdf",
        "fig_collision_effects.pdf",
        "fig_collision_mechanisms.pdf",
        "publication_figures.manifest.json",
    }
    manifest = json.loads(
        (FIGURES.DEFAULT_OUTPUT_DIR / "publication_figures.manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["rendering"]["width_inches"] == pytest.approx(7.16)
    assert manifest["rendering"]["minimum_configured_text_size_pt"] >= 7.0
    assert manifest["rendering"]["pdf_fonttype"] == 42
    assert manifest["validation"]["bootstrap_replicates"] == 10_000


@pytest.mark.skipif(
    shutil.which("pdfinfo") is None or shutil.which("pdffonts") is None,
    reason="Poppler command-line validators are unavailable",
)
def test_page_geometry_and_embedded_vector_fonts() -> None:
    for name, expected_height in FIGURES.FIGURE_HEIGHTS_IN.items():
        path = FIGURES.DEFAULT_OUTPUT_DIR / name
        info = subprocess.run(
            ["pdfinfo", str(path)],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        match = re.search(r"Page size:\s+([0-9.]+) x ([0-9.]+) pts", info)
        assert match is not None
        width_pt, height_pt = map(float, match.groups())
        assert width_pt / 72 == pytest.approx(7.16, abs=0.005)
        assert height_pt / 72 == pytest.approx(expected_height, abs=0.005)

        fonts = subprocess.run(
            ["pdffonts", str(path)],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        assert "Type 3" not in fonts
        records = fonts.splitlines()[2:]
        assert records
        for record in records:
            fields = record.split()
            assert "TrueType" in record
            assert fields[-5:-2] == ["yes", "yes", "yes"]


def test_generator_has_no_release_temp_path() -> None:
    source = GENERATOR_PATH.read_text(encoding="utf-8")
    forbidden_temp_root = str(Path("/").joinpath("tmp", "tccn-article-build"))
    assert forbidden_temp_root not in source
    assert "EXPECTED_INPUT_SHA256" in source


def test_architecture_is_byte_deterministic_and_current() -> None:
    first = ARCHITECTURE.render()
    second = ARCHITECTURE.render()
    assert first == second
    assert first.startswith(b"%PDF-")
    assert ARCHITECTURE.DEFAULT_OUTPUT.read_bytes() == first
    source = ARCHITECTURE_GENERATOR_PATH.read_text(encoding="utf-8")
    assert "ACK / requeue" not in source
    assert "20 seeds" in source
    assert "8 cells" in source
    assert "24 blocks" in source
