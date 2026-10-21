#!/usr/bin/env python3
"""Generate the three hash-pinned publication figures for CRN-2026-003.

The generator reads only portable files in the study tree, verifies every
scientific input against a literal SHA-256 pin and the human-readable source
manifest, cross-checks redundant Gold exports, and writes deterministic PDF
bytes plus a machine-readable derivative manifest.  It performs no analysis
that changes an estimand or decision rule; calculations here are plotting-only
summaries (percent conversion, arithmetic means already defined by the Gold
analysis, and deletion estimates expressed relative to the all-data point).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Final

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


ANALYSIS_DIR: Final = Path(__file__).resolve().parent
STUDY_ROOT: Final = ANALYSIS_DIR.parent
WORKSPACE_ROOT: Final = STUDY_ROOT.parent.parent
DEFAULT_OUTPUT_DIR: Final = WORKSPACE_ROOT / "main" / "manuscript" / "figures"
SOURCE_MANIFEST: Final = (
    STUDY_ROOT / "evidence" / "gold" / "figure_inputs" / "source_manifest.json"
)

EXPECTED_INPUT_SHA256: Final[dict[str, str]] = {
    "evidence/gold/figure_inputs/archived_evidence_figure_data.json": (
        "985c718c5016d8e19df3dfc8a97a30d025b40dcbd51224eee56011662f31a0e5"
    ),
    "evidence/gold/figure_inputs/collision_aware_figure_data.csv": (
        "751647f671780d0fa0e38756c22485d0e70cf8d9ccea97e33e39f21dcb15ad6b"
    ),
    "evidence/gold/reanalysis/reanalysis_output.json": (
        "ab6ccafc932ddca28a2aec40d6e8f46091ab17992955aed988a764649affe2e0"
    ),
    "evidence/gold/reanalysis/mechanism_audit_figure_data.csv": (
        "9e69eca00e3c5343d67e342e34f9cb816b311c137f493dc92d2fa2aa12c61e2a"
    ),
    "evidence/gold/reanalysis/attempted_only_logical_audit.csv": (
        "fe686729c5c65d6d1a59b0afb5016e8d9b80c24401d7a4abd7d5cb71c9d8d7f2"
    ),
    "evidence/gold/inference/recomputed.json": (
        "5199f52a44d119010207c2e0358217d2a14ed238f57bb9a743f76f7c189ccca3"
    ),
}

FIGURE_WIDTH_IN: Final = 7.16
FIGURE_HEIGHTS_IN: Final[dict[str, float]] = {
    "fig_archived_evidence.pdf": 5.05,
    "fig_collision_effects.pdf": 6.20,
    "fig_collision_mechanisms.pdf": 5.75,
}
MIN_FONT_PT: Final = 7.0

BLUE: Final = "#0072B2"
ORANGE: Final = "#E69F00"
GREEN: Final = "#009E73"
VERMILLION: Final = "#D55E00"
PURPLE: Final = "#7A5195"
SKY: Final = "#56B4E9"
GREY: Final = "#777777"
LIGHT_GREY: Final = "#D0D0D0"
BLACK: Final = "#222222"

FACTOR_ORDER: Final = (
    "iid_o30_a10",
    "iid_o30_a25",
    "iid_o60_a10",
    "iid_o60_a25",
    "markov_o30_a10",
    "markov_o30_a25",
    "markov_o60_a10",
    "markov_o60_a25",
)
FACTOR_LABELS: Final = {
    "iid_o30_a10": "IID\n$\\rho=.30,\\ \\lambda=.10$",
    "iid_o30_a25": "IID\n$\\rho=.30,\\ \\lambda=.25$",
    "iid_o60_a10": "IID\n$\\rho=.60,\\ \\lambda=.10$",
    "iid_o60_a25": "IID\n$\\rho=.60,\\ \\lambda=.25$",
    "markov_o30_a10": "Markov\n$\\rho=.30,\\ \\lambda=.10$",
    "markov_o30_a25": "Markov\n$\\rho=.30,\\ \\lambda=.25$",
    "markov_o60_a10": "Markov\n$\\rho=.60,\\ \\lambda=.10$",
    "markov_o60_a25": "Markov\n$\\rho=.60,\\ \\lambda=.25$",
}

POLICY_LABELS: Final = {
    "centralized_quality_matching": "Centralized quality\nmatching (reference)",
    "phase_orthogonal": "Phase-orthogonal",
    "collision_aware_dqn": "Full Double-DQN",
    "quality_greedy_persistent": "Quality-greedy",
    "history_free_dqn": "Outcome-history-\nremoved Double-DQN",
    "coordination_blind_dqn": "Coordination-cue-\nremoved Double-DQN",
    "uniform_legal": "Uniform legal",
    "round_robin": "Round robin",
    "static_priority": "Static priority",
}
POLICY_ORDER: Final = (
    "centralized_quality_matching",
    "phase_orthogonal",
    "collision_aware_dqn",
    "quality_greedy_persistent",
    "history_free_dqn",
    "coordination_blind_dqn",
    "uniform_legal",
    "round_robin",
    "static_priority",
)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    return rows


def _assert_close(actual: float, expected: float, label: str, tol: float = 1e-12) -> None:
    if not math.isclose(actual, expected, rel_tol=tol, abs_tol=tol):
        raise ValueError(f"{label}: observed {actual!r}, expected {expected!r}")


def verify_inputs() -> tuple[dict[str, Path], dict[str, str]]:
    """Verify the source manifest and every pinned scientific input."""

    source_manifest = _read_json(SOURCE_MANIFEST)
    records = source_manifest.get("inputs")
    if not isinstance(records, list):
        raise ValueError("source manifest lacks an input list")
    declared = {
        str(record["path_from_study_root"]): str(record["sha256"])
        for record in records
    }
    if declared != EXPECTED_INPUT_SHA256:
        raise ValueError("source manifest does not equal the generator's literal hash pins")

    paths: dict[str, Path] = {}
    observed: dict[str, str] = {}
    for relative, expected in EXPECTED_INPUT_SHA256.items():
        path = STUDY_ROOT / relative
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"missing or symbolic-link figure input: {path}")
        digest = _sha256_file(path)
        if digest != expected:
            raise ValueError(
                f"input hash mismatch for {relative}: observed {digest}, expected {expected}"
            )
        paths[relative] = path
        observed[relative] = digest
    return paths, observed


def load_and_validate() -> dict[str, Any]:
    paths, observed_hashes = verify_inputs()
    archived = _read_json(
        paths["evidence/gold/figure_inputs/archived_evidence_figure_data.json"]
    )
    collision_rows = _read_csv(
        paths["evidence/gold/figure_inputs/collision_aware_figure_data.csv"]
    )
    reanalysis = _read_json(paths["evidence/gold/reanalysis/reanalysis_output.json"])
    mechanism_rows = _read_csv(
        paths["evidence/gold/reanalysis/mechanism_audit_figure_data.csv"]
    )
    attempted_rows = _read_csv(
        paths["evidence/gold/reanalysis/attempted_only_logical_audit.csv"]
    )
    inference = _read_json(paths["evidence/gold/inference/recomputed.json"])

    if archived.get("schema_version") != 1 or len(archived.get("seed_order", [])) != 8:
        raise ValueError("unexpected archived evidence schema")
    assay = archived.get("assay", {})
    if len(assay.get("trained_full_core_agreement", [])) != 8:
        raise ValueError("archived assay must contain eight trained checkpoints")
    if len(assay.get("untrained_constructed_agreement", [])) != 8:
        raise ValueError("archived assay must contain eight untrained controls")
    if len(archived.get("off_archive_crossings", [])) != 8:
        raise ValueError("archived crossing audit must contain eight checkpoints")

    if len(collision_rows) != 150:
        raise ValueError(f"expected 150 sealed collision figure records, got {len(collision_rows)}")
    panel_counts = {
        panel: sum(row["panel"] == panel for row in collision_rows)
        for panel in ("a", "b", "c", "d")
    }
    if panel_counts != {"a": 29, "b": 29, "c": 72, "d": 20}:
        raise ValueError(f"unexpected sealed panel counts: {panel_counts}")

    mechanism_by_series: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in mechanism_rows:
        mechanism_by_series[row["series"]].append(row)
    expected_series = {
        "seven_action_selected",
        "seven_action_attempt_ceiling",
        "attempted_only_calibration_selected",
        "attempted_only_validation_optimal",
    }
    if set(mechanism_by_series) != expected_series:
        raise ValueError("unexpected mechanism series")
    if any(len(rows) != 20 for rows in mechanism_by_series.values()):
        raise ValueError("each mechanism series must contain 20 checkpoints")
    mechanism_seeds = {
        tuple(int(row["training_seed"]) for row in mechanism_by_series[name])
        for name in expected_series
    }
    if len(mechanism_seeds) != 1:
        raise ValueError("mechanism-series checkpoint order differs")

    attempted = reanalysis["attempted_only_logical_audit"]
    checkpoints = attempted["per_checkpoint"]
    if len(checkpoints) != 20 or len(attempted_rows) != 20:
        raise ValueError("logical attempted-only audit must contain 20 checkpoints")
    json_by_seed = {int(row["training_seed"]): row for row in checkpoints}
    for csv_row in attempted_rows:
        seed = int(csv_row["training_seed"])
        json_row = json_by_seed[seed]
        _assert_close(
            float(csv_row["calibration_selected_validation_agreement"]),
            float(json_row["calibration_selected_validation_agreement"]),
            f"attempted-only validation agreement, seed {seed}",
        )
        if csv_row["full_acyclic"] != str(json_row["full_support_certificate"]["acyclic"]):
            raise ValueError(f"attempted-only certificate mismatch, seed {seed}")
        if int(csv_row["full_representing_order_count"]) != int(
            json_row["full_support_certificate"]["representing_order_count"]
        ):
            raise ValueError(f"attempted-only order-count mismatch, seed {seed}")

    seven = reanalysis["original_seven_action"]
    _assert_close(
        np.mean(
            [
                float(row["agreement"])
                for row in mechanism_by_series["seven_action_attempt_ceiling"]
            ]
        ),
        float(seven["validation_equal_checkpoint_factor_attempt_ceiling_same_weights"]),
        "seven-action attempt ceiling macro",
    )
    _assert_close(
        np.mean(
            [
                float(row["agreement"])
                for row in mechanism_by_series["attempted_only_calibration_selected"]
            ]
        ),
        float(attempted["validation_equal_checkpoint_factor_selected_agreement"]),
        "attempted-only agreement macro",
    )

    comparisons = inference["inference"]["comparisons"]
    for panel, comparator in (("a", "phase_orthogonal"), ("b", "history_free_dqn")):
        macro_rows = [
            row
            for row in collision_rows
            if row["panel"] == panel and row["record_type"] == "equal_cell_macro_effect"
        ]
        if len(macro_rows) != 1:
            raise ValueError(f"panel {panel} lacks exactly one macro effect")
        _assert_close(
            float(macro_rows[0]["value"]),
            float(comparisons[comparator]["point_effect"]),
            f"sealed/recomputed point effect for {comparator}",
        )

    positive = reanalysis["coordinate_positive_control"]
    if positive["case_count"] != 378:
        raise ValueError("coordinate positive control must enumerate 6 x 63 cases")
    if not positive["logical"]["acyclic"] or positive["logical"]["representing_order_count"] != 1:
        raise ValueError("logical-coordinate positive control failed")
    if positive["physical"]["acyclic"] or positive["physical"]["representing_order_count"] != 0:
        raise ValueError("coordinate-mixed diagnostic did not expose the known confound")
    if any(row["full_support_certificate"]["acyclic"] for row in checkpoints):
        raise ValueError("unexpected acyclic full-support DQN certificate")

    inference_method = inference["inference"]["method"]
    if inference_method["replicates"] != 10_000:
        raise ValueError("expected 10,000 crossed-bootstrap replicates")

    return {
        "archived": archived,
        "collision_rows": collision_rows,
        "reanalysis": reanalysis,
        "mechanism_by_series": dict(mechanism_by_series),
        "attempted_rows": attempted_rows,
        "inference": inference,
        "input_hashes": observed_hashes,
        "panel_counts": panel_counts,
    }


def _style() -> dict[str, Any]:
    return {
        "font.family": "DejaVu Sans",
        "font.size": 7.0,
        "axes.titlesize": 8.2,
        "axes.titleweight": "semibold",
        "axes.labelsize": 7.2,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "legend.fontsize": 7.0,
        "axes.linewidth": 0.65,
        "grid.linewidth": 0.45,
        "lines.linewidth": 1.15,
        "lines.markersize": 4.2,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.transparent": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }


def _finish_axis(ax: Any, *, axis: str = "y") -> None:
    ax.grid(True, axis=axis, color="#E6E6E6", zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _render_pdf(fig: Any, *, title: str) -> bytes:
    payload = io.BytesIO()
    fig.savefig(
        payload,
        format="pdf",
        dpi=300,
        metadata={
            "Title": title,
            "Author": "ChainRadioNet collaboration",
            "Subject": "Hash-pinned quantitative evidence",
            "Keywords": "dynamic spectrum access; DQN; reproducibility",
            "Creator": "CRN-2026-003 deterministic publication-figure generator",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(fig)
    return payload.getvalue()


def render_archived(data: dict[str, Any]) -> bytes:
    archived = data["archived"]
    with plt.rc_context(_style()):
        fig, axes = plt.subplots(
            2,
            2,
            figsize=(FIGURE_WIDTH_IN, FIGURE_HEIGHTS_IN["fig_archived_evidence.pdf"]),
            layout="constrained",
        )

        ax = axes[0, 0]
        assay = archived["assay"]
        jitter = np.linspace(-0.12, 0.12, 8)
        trained = np.asarray(assay["trained_full_core_agreement"], dtype=float) * 100
        untrained = np.asarray(assay["untrained_constructed_agreement"], dtype=float) * 100
        ax.scatter(jitter, trained, s=26, color=BLUE, edgecolor="white", linewidth=0.45, zorder=3)
        ax.scatter(1 + jitter, untrained, s=26, color=ORANGE, edgecolor="white", linewidth=0.45, zorder=3)
        controls = np.asarray(
            [assay["quality_greedy_heldout_agreement"], assay["round_robin_heldout_agreement"]]
        ) * 100
        ax.scatter([2, 3], controls, s=35, marker="D", color=GREEN, edgecolor="white", linewidth=0.45, zorder=3)
        ax.axhline(100, color=GREY, linewidth=0.7, linestyle=":")
        ax.set_xticks([0, 1, 2, 3], ["Trained\nDQN", "Untrained\nDQN", "Quality-\ngreedy", "Round\nrobin"])
        ax.set_ylabel("Best masked-priority agreement (%)")
        ax.set_ylim(30, 103)
        ax.set_title("(a) Masked-priority assay")
        ax.text(0, 96.8, "8/8 exact", ha="center", va="top")
        ax.text(1, float(untrained.min()) - 2.8, "0/8 exact", ha="center", va="top")
        _finish_axis(ax)

        ax = axes[0, 1]
        crossing = archived["off_archive_crossings"]
        x = np.arange(len(crossing))
        problems = np.asarray(
            [row["problems"] / row["problem_denominator"] * 100 for row in crossing]
        )
        masks = np.asarray([row["masks"] / row["mask_denominator"] * 100 for row in crossing])
        width = 0.37
        bars_problem = ax.bar(x - width / 2, problems, width, color=VERMILLION, label="Problems / 129", zorder=2)
        bars_mask = ax.bar(
            x + width / 2,
            masks,
            width,
            facecolor="white",
            edgecolor=GREY,
            hatch="///",
            linewidth=0.9,
            label="Masks / 63",
            zorder=2,
        )
        ax.set_xticks(x, [str(row["training_seed"]) for row in crossing], rotation=42, ha="right")
        ax.set_ylabel("Off-support crossings (%)")
        ax.set_xlabel("Training seed")
        ax.set_ylim(0, 29)
        ax.set_title("(b) Targeted boundary search")
        ax.legend(loc="upper left", frameon=False, ncol=2, columnspacing=0.8, handlelength=1.3)
        for bars, key in ((bars_problem, "problems"), (bars_mask, "masks")):
            for bar, row in zip(bars, crossing, strict=True):
                ax.annotate(
                    str(row[key]),
                    (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 2),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                )
        _finish_axis(ax)

        ax = axes[1, 0]
        labels = ["Homogeneous\nDQN", "Mixed\ncheckpoints", "Round\nrobin", "Quality-\ngreedy", "Uniform\nlegal"]
        observed = archived["s1_spread"]["observed"]
        analytical = archived["s1_spread"]["analytical"]
        observed_values = [
            observed["homogeneous_dqn"],
            observed["mixed_checkpoints"],
            observed["round_robin"],
            observed["quality_greedy"],
        ]
        observed_bars = ax.bar(np.arange(4), observed_values, width=0.64, color=BLUE, alpha=0.82, label="Archived mean", zorder=2)
        control_bar = ax.bar(4, analytical["uniform_legal"], width=0.64, facecolor="white", edgecolor=GREEN, hatch="///", linewidth=0.9, label="Analytical control", zorder=2)
        exact_marks = ax.scatter(
            [0, 1],
            [analytical["homogeneous_dqn"], analytical["mixed_checkpoints"]],
            marker="_",
            s=180,
            linewidth=2.1,
            color=ORANGE,
            zorder=4,
            label="Exact expectation",
        )
        ax.set_xticks(np.arange(5), labels)
        ax.set_ylabel("Unique channels per cycle, $U$")
        ax.set_ylim(0, 6.35)
        ax.set_title("(c) S1 spread: observed and analytical")
        ax.legend(
            [observed_bars, exact_marks, control_bar],
            ["Observed", "Exact", "Analytical control"],
            loc="upper center",
            frameon=False,
            ncol=3,
            columnspacing=0.7,
            handlelength=1.2,
            handletextpad=0.35,
        )
        _finish_axis(ax)

        ax = axes[1, 1]
        stages = archived["generator_path"]["stages"]
        labels = [row["stage"] for row in stages]
        fractions = np.asarray([row["fraction"] for row in stages], dtype=float) * 100
        y = np.arange(len(stages))
        colors = [ORANGE, ORANGE] + [LIGHT_GREY] * (len(stages) - 2)
        ax.barh(y, fractions, color=colors, height=0.62, zorder=2)
        ax.set_yticks(y, labels)
        ax.invert_yaxis()
        ax.set_xlabel("Matched row pairs changed (%)")
        ax.set_xlim(0, 105)
        ax.set_title("(d) Dispersion-contrast propagation")
        for index, pct in enumerate(fractions):
            if pct == 0:
                ax.annotate("0", (0, index), xytext=(3, 0), textcoords="offset points", va="center")
            else:
                ax.text(pct - 1.4, index, f"{pct:.2f}%", ha="right", va="center")
        ax.text(
            103,
            len(stages) - 0.42,
            f"$n={int(archived['generator_path']['rows']):,}$ row pairs",
            ha="right",
            va="bottom",
        )
        _finish_axis(ax, axis="x")

        return _render_pdf(fig, title="Archived finite-domain policy evidence")


def _effect_panel(
    ax: Any,
    rows: list[dict[str, str]],
    *,
    title: str,
    margin: float | None,
) -> None:
    cell_rows = {
        row["factor_id"]: row
        for row in rows
        if row["record_type"] == "cell_effect"
    }
    checkpoint = [float(row["value"]) for row in rows if row["record_type"] == "checkpoint_effect"]
    macro = next(row for row in rows if row["record_type"] == "equal_cell_macro_effect")
    y_cell = np.arange(len(FACTOR_ORDER), dtype=float)
    values = np.asarray([float(cell_rows[f]["value"]) for f in FACTOR_ORDER])
    lower = np.asarray([float(cell_rows[f]["lower_bound"]) for f in FACTOR_ORDER])
    upper = np.asarray([float(cell_rows[f]["upper_bound"]) for f in FACTOR_ORDER])
    ax.errorbar(
        values,
        y_cell,
        xerr=np.vstack([values - lower, upper - values]),
        fmt="o",
        color=BLUE,
        ecolor=SKY,
        elinewidth=1.2,
        capsize=2.0,
        label="Cell mean (5th–95th bootstrap)",
        zorder=3,
    )
    macro_y = 8.65
    macro_v = float(macro["value"])
    macro_lo = float(macro["lower_bound"])
    macro_hi = float(macro["upper_bound"])
    ax.errorbar(
        [macro_v],
        [macro_y],
        xerr=[[macro_v - macro_lo], [macro_hi - macro_v]],
        fmt="D",
        color=BLACK,
        ecolor=BLACK,
        capsize=2.3,
        label="Equal-cell macro",
        zorder=4,
    )
    seed_y = 9.75 + np.linspace(-0.18, 0.18, len(checkpoint))
    ax.scatter(checkpoint, seed_y, s=14, facecolor="white", edgecolor=GREY, linewidth=0.65, label="20 checkpoint effects", zorder=3)
    ax.axvline(0, color=BLACK, linewidth=0.75)
    if margin is not None:
        ax.axvline(margin, color=VERMILLION, linewidth=0.9, linestyle="--", label=f"Superiority margin = {margin:.2f}")
        ax.text(
            margin,
            0.985,
            rf"$\delta={margin:.2f}$",
            transform=ax.get_xaxis_transform(),
            ha="right",
            va="top",
            rotation=90,
            color=VERMILLION,
        )
    ax.set_yticks(
        [*y_cell, macro_y, 9.75],
        [*[FACTOR_LABELS[f] for f in FACTOR_ORDER], "Macro", "Checkpoints"],
    )
    ax.invert_yaxis()
    ax.set_xlabel(r"Goodput difference, $\Delta$ (packets/slot)")
    ax.set_title(title)
    _finish_axis(ax, axis="x")


def render_effects(data: dict[str, Any]) -> bytes:
    collision_rows = data["collision_rows"]
    inference = data["inference"]
    with plt.rc_context(_style()):
        fig, axes = plt.subplots(
            2,
            2,
            figsize=(FIGURE_WIDTH_IN, FIGURE_HEIGHTS_IN["fig_collision_effects.pdf"]),
            layout="constrained",
        )
        _effect_panel(
            axes[0, 0],
            [row for row in collision_rows if row["panel"] == "a"],
            title="(a) Double-DQN: full $-$\nphase-orthogonal",
            margin=0.05,
        )
        axes[0, 0].set_ylabel(r"Factor cell: process and $(\rho,\lambda)$")
        _effect_panel(
            axes[0, 1],
            [row for row in collision_rows if row["panel"] == "b"],
            title="(b) Double-DQN: full $-$\noutcome-history-removed",
            margin=None,
        )
        axes[0, 1].tick_params(labelleft=False)
        axes[0, 1].set_ylabel("Factor cell (same order as panel a)")

        ax = axes[1, 0]
        policy_cell_rows: dict[str, list[float]] = defaultdict(list)
        for row in collision_rows:
            if row["panel"] == "c" and row["record_type"] == "factorial_cell_mean":
                policy_cell_rows[row["policy"]].append(float(row["value"]))
        y = np.arange(len(POLICY_ORDER), dtype=float)
        for idx, policy in enumerate(POLICY_ORDER):
            values = np.asarray(policy_cell_rows[policy], dtype=float)
            if len(values) != 8:
                raise ValueError(f"policy {policy} lacks eight factor-cell means")
            ax.scatter(values, np.full(8, idx), s=15, facecolor="white", edgecolor=BLUE, linewidth=0.65, zorder=3)
            ax.scatter([float(values.mean())], [idx], s=30, marker="D", color=BLACK, edgecolor="white", linewidth=0.35, zorder=4)
        ax.set_yticks(y, [POLICY_LABELS[p] for p in POLICY_ORDER])
        ax.invert_yaxis()
        ax.set_xlabel("Goodput (packets/slot)")
        ax.set_ylabel("Policy")
        ax.set_title("(c) Eight cell means and equal-cell policy macro")
        _finish_axis(ax, axis="x")

        ax = axes[1, 1]
        comparisons = inference["inference"]["comparisons"]
        sensitivity_specs = [
            ("phase_orthogonal", "leave_one_training_unit", "Full − phase\nomit training seed", BLUE),
            ("phase_orthogonal", "leave_one_environment_block_shift_preserving", "Full − phase\nomit environment block", SKY),
            ("history_free_dqn", "leave_one_training_unit", "Full − outcome-history-removed\nomit training seed", VERMILLION),
            ("history_free_dqn", "leave_one_environment_block_shift_preserving", "Full − outcome-history-removed\nomit environment block", ORANGE),
        ]
        for idx, (comparison, deletion, _label, color) in enumerate(sensitivity_specs):
            record = comparisons[comparison]
            point = float(record["point_effect"])
            low, high = record["sensitivity"][deletion]["range"]
            deviations = np.asarray([float(low) - point, float(high) - point])
            ax.hlines(idx, deviations[0], deviations[1], color=color, linewidth=3.0, zorder=2)
            ax.plot(deviations, [idx, idx], marker="|", linestyle="None", color=BLACK, markersize=8, markeredgewidth=1.0, zorder=3)
            ax.scatter([0], [idx], s=24, marker="D", color=BLACK, edgecolor="white", linewidth=0.35, zorder=4)
            ax.text(deviations[1] + 0.00035, idx, f"[{deviations[0]:+.4f}, {deviations[1]:+.4f}]", va="center")
        ax.axvline(0, color=BLACK, linewidth=0.75)
        ax.set_yticks(np.arange(len(sensitivity_specs)), [spec[2] for spec in sensitivity_specs])
        ax.invert_yaxis()
        ax.set_xlim(-0.0145, 0.0145)
        ax.set_xlabel("Change from all-data estimate (packets/slot)")
        ax.set_ylabel("Deletion diagnostic")
        ax.set_title("(d) One-block deletion sensitivity ranges")
        _finish_axis(ax, axis="x")

        return _render_pdf(fig, title="Prospective collision-aware goodput effects")


def render_mechanisms(data: dict[str, Any]) -> bytes:
    reanalysis = data["reanalysis"]
    mechanism = data["mechanism_by_series"]
    inference = data["inference"]
    with plt.rc_context(_style()):
        fig, axes = plt.subplots(
            2,
            2,
            figsize=(FIGURE_WIDTH_IN, FIGURE_HEIGHTS_IN["fig_collision_mechanisms.pdf"]),
            layout="constrained",
        )

        ax = axes[0, 0]
        seeds = [int(row["training_seed"]) for row in mechanism["seven_action_attempt_ceiling"]]
        ceiling = np.asarray([float(row["agreement"]) for row in mechanism["seven_action_attempt_ceiling"]]) * 100
        attempted = np.asarray([float(row["agreement"]) for row in mechanism["attempted_only_calibration_selected"]]) * 100
        x = np.arange(1, 21)
        ax.plot(x, ceiling, "o-", color=VERMILLION, label="Seven-action attempt ceiling")
        ax.plot(x, attempted, "s-", color=BLUE, label="Attempted-only validation agreement")
        ax.axhline(float(ceiling.mean()), color=VERMILLION, linewidth=0.75, linestyle=":")
        ax.axhline(float(attempted.mean()), color=BLUE, linewidth=0.75, linestyle=":")
        tick_positions = [1, 5, 9, 13, 17, 20]
        ax.set_xticks(tick_positions, [str(seeds[i - 1]) for i in tick_positions], rotation=35, ha="right")
        ax.set_xlabel("Training seed (six labels shown)")
        ax.set_ylabel("Validation agreement (%)")
        ax.set_ylim(0, 66)
        ax.set_title("(a) Full Double-DQN: abstention and attempted-only fit")
        # Both curves leave a deliberate mid-range band. Keeping the legend
        # there preserves the data scale without covering either series.
        ax.legend(
            loc="center",
            bbox_to_anchor=(0.5, 0.43),
            frameon=True,
            facecolor="white",
            edgecolor="none",
            framealpha=0.96,
        )
        _finish_axis(ax)

        ax = axes[0, 1]
        checkpoint_records = reanalysis["attempted_only_logical_audit"]["per_checkpoint"]
        dqn_values = np.asarray(
            [row["calibration_selected_validation_agreement"] for row in checkpoint_records], dtype=float
        ) * 100
        positive = reanalysis["coordinate_positive_control"]
        rng_offsets = np.linspace(-0.13, 0.13, len(dqn_values))
        ax.scatter(rng_offsets, dqn_values, s=20, facecolor="white", edgecolor=BLUE, linewidth=0.7, zorder=3)
        ax.scatter([0], [dqn_values.mean()], s=38, marker="D", color=BLUE, edgecolor="white", linewidth=0.4, zorder=4)
        ax.scatter([1], [positive["logical_exact_agreement"] * 100], s=50, marker="o", color=GREEN, edgecolor="white", linewidth=0.4, zorder=4)
        ax.scatter([2], [positive["physical_best_order_agreement"] * 100], s=50, marker="X", color=VERMILLION, edgecolor="white", linewidth=0.4, zorder=4)
        ax.set_xticks(
            [0, 1, 2],
            [
                "Full Double-DQN:\nlogical attempts",
                "Ascending control:\nlogical",
                "Same control:\nphysical-pooled†",
            ],
        )
        ax.set_xlabel("Policy and coordinate treatment")
        ax.set_ylabel("Best masked-priority agreement (%)")
        ax.set_ylim(0, 108)
        ax.set_title("(b) Coordinate-aware certificate and control")
        ax.text(0.02, 0.15, "cyclic: 20/20", transform=ax.transAxes, ha="left", color=BLUE)
        ax.text(1, 92, "exact; one order", ha="center", va="top", color=GREEN)
        ax.text(0.98, 0.15, "cyclic (diagnostic)", transform=ax.transAxes, ha="right", color=VERMILLION)
        ax.text(0.99, 0.01, "†coordinate mixing is not a policy-mechanism test", transform=ax.transAxes, ha="right", va="bottom")
        _finish_axis(ax)

        ax = axes[1, 0]
        selected_policies = (
            "collision_aware_dqn",
            "history_free_dqn",
            "phase_orthogonal",
            "quality_greedy_persistent",
        )
        short_labels = (
            "Full Double-DQN",
            "Outcome-history\nremoved Double-DQN",
            "Phase-orthogonal",
            "Quality-greedy",
        )
        metrics = (
            ("pooled_attempt_fraction_given_eligible", "Attempts /\neligible"),
            ("pooled_collision_attempt_rate", "Collisions /\nattempts"),
            ("pooled_conditional_link_failure_rate", "Failure /\nsingleton"),
            ("opportunity_utilization", "Delivery /\nceiling"),
            ("delivery_jain_fairness_all_15_nodes", "Jain /\n15 nodes"),
        )
        operational = inference["operational"]["policies"]
        base_x = np.arange(len(metrics), dtype=float)
        width = 0.19
        colors = (BLUE, ORANGE, GREEN, PURPLE)
        for p_index, (policy, label, color) in enumerate(zip(selected_policies, short_labels, colors, strict=True)):
            macro = operational[policy]["macro"]
            values = [float(macro[key]) * 100 for key, _ in metrics]
            ax.bar(base_x + (p_index - 1.5) * width, values, width, label=label, color=color, zorder=2)
        ax.set_xticks(base_x, [label for _, label in metrics])
        ax.set_ylabel("Rate or index (%)")
        # Reserve an explicit legend band above the tallest (sub-100%) bar.
        ax.set_ylim(0, 120)
        ax.set_xlabel("Metric (first 3 pooled; last 2 equal-cell means)")
        ax.set_title("(c) Operating metrics with explicit denominators")
        ax.legend(
            loc="upper center",
            frameon=True,
            facecolor="white",
            edgecolor="none",
            framealpha=0.96,
            ncol=2,
            columnspacing=0.8,
            handletextpad=0.45,
        )
        _finish_axis(ax)

        ax = axes[1, 1]
        load_cells = inference["load_bound"]["cells"]
        by_rho: dict[float, dict[str, float]] = {}
        for row in load_cells:
            rho = float(row["occupancy_probability"])
            by_rho.setdefault(rho, {})[f"offered_{row['arrival_probability']}"] = float(row["offered_packets_per_slot"])
            by_rho[rho]["service_bound"] = float(row["quality_aware_optimistic_service_upper_bound"])
        rhos = [0.3, 0.6]
        x = np.arange(2)
        bounds = [by_rho[r]["service_bound"] for r in rhos]
        bars = ax.bar(x, bounds, width=0.58, color=SKY, edgecolor=BLUE, linewidth=0.8, label="Optimistic service bound", zorder=2)
        low = [by_rho[r]["offered_0.1"] for r in rhos]
        high = [by_rho[r]["offered_0.25"] for r in rhos]
        ax.scatter(x - 0.10, low, marker="o", s=35, color=GREEN, edgecolor="white", linewidth=0.4, label=r"Offered: $15\lambda=1.50$", zorder=4)
        ax.scatter(x + 0.10, high, marker="D", s=35, color=VERMILLION, edgecolor="white", linewidth=0.4, label=r"Offered: $15\lambda=3.75$", zorder=4)
        for bar, bound in zip(bars, bounds, strict=True):
            ax.text(bar.get_x() + bar.get_width() / 2, bound - 0.10, f"{bound:.3f}", ha="center", va="top", color=BLACK)
        for idx, bound in enumerate(bounds):
            if high[idx] > bound:
                ax.annotate("overloaded", (idx + 0.10, high[idx]), xytext=(0, 6), textcoords="offset points", ha="center", color=VERMILLION)
        ax.set_xticks(x, [r"$\rho=0.30$", r"$\rho=0.60$"])
        ax.set_xlabel(r"Occupancy probability, $\rho$")
        ax.set_ylabel("Packets per slot")
        # The upper band is reserved for the legend, above every datum and
        # overload annotation.
        ax.set_ylim(0, 5.35)
        ax.set_title("(d) Offered load versus analytical service bound")
        ax.legend(
            loc="upper center",
            frameon=True,
            facecolor="white",
            edgecolor="none",
            framealpha=0.96,
            ncol=1,
        )
        _finish_axis(ax)

        return _render_pdf(fig, title="Collision-aware policy mechanism and operating diagnostics")


def render_all(data: dict[str, Any]) -> dict[str, bytes]:
    return {
        "fig_archived_evidence.pdf": render_archived(data),
        "fig_collision_effects.pdf": render_effects(data),
        "fig_collision_mechanisms.pdf": render_mechanisms(data),
    }


def _canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def build_manifest(data: dict[str, Any], figures: dict[str, bytes]) -> bytes:
    manifest = {
        "schema_version": 1,
        "study_id": "crn-2026-003-finite-domain-dsa",
        "artifact_role": "deterministic_publication_derivatives",
        "generator": str(Path("studies/crn-2026-003-finite-domain-dsa/analysis") / Path(__file__).name),
        "generator_sha256": _sha256_file(Path(__file__).resolve()),
        "source_manifest": "studies/crn-2026-003-finite-domain-dsa/evidence/gold/figure_inputs/source_manifest.json",
        "source_manifest_sha256": _sha256_file(SOURCE_MANIFEST),
        "source_sha256": data["input_hashes"],
        "rendering": {
            "width_inches": FIGURE_WIDTH_IN,
            "minimum_configured_text_size_pt": MIN_FONT_PT,
            "font_family": "DejaVu Sans",
            "pdf_fonttype": 42,
            "fixed_pdf_creation_and_modification_dates": True,
            "rasterized_artists": False,
        },
        "figures": {
            name: {
                "sha256": _sha256_bytes(payload),
                "bytes": len(payload),
                "width_inches": FIGURE_WIDTH_IN,
                "height_inches": FIGURE_HEIGHTS_IN[name],
            }
            for name, payload in figures.items()
        },
        "panel_semantics": {
            "fig_archived_evidence.pdf": {
                "a": "Held-out masked-priority agreement for trained checkpoints and negative controls.",
                "b": "Targeted off-archive crossings reported against explicit 129-problem and 63-mask denominators.",
                "c": "Archived S1 spread and exact analytical reconstruction/control; bars and analytical marks retain distinct meanings.",
                "d": "Matched-row propagation of the generator-dispersion contrast (7,200,000 pairs).",
            },
            "fig_collision_effects.pdf": {
                "a": (
                    "Full prospective collision-aware Double-DQN minus "
                    "phase-orthogonal goodput: eight cell effects with sealed "
                    "5th--95th bootstrap intervals, 20 checkpoint effects, and "
                    "equal-cell macro."
                ),
                "b": (
                    "Full prospective collision-aware Double-DQN minus "
                    "outcome-history-removed Double-DQN under the same display "
                    "grammar."
                ),
                "c": "Eight factorial-cell goodput means and their arithmetic equal-cell macro for all nine policies; the centralized policy is descriptive only.",
                "d": "Leave-one-training-unit and shift-preserving leave-one-environment-block ranges, expressed as change from the all-data point estimate.",
            },
            "fig_collision_mechanisms.pdf": {
                "a": (
                    "Per-checkpoint seven-action attempt ceiling versus "
                    "calibration-selected attempted-only logical validation "
                    "agreement for the 20 full prospective Double-DQN "
                    "checkpoints; dotted lines are equal-checkpoint/equal-factor "
                    "means."
                ),
                "b": (
                    "Attempted-only logical-coordinate certificates for the 20 "
                    "full prospective Double-DQN checkpoints and the 6x63 "
                    "ascending-priority positive control. Physical pooling of "
                    "that known control is displayed only as a diagnostic "
                    "demonstration of coordinate confounding, never as a "
                    "policy-mechanism result."
                ),
                "c": "Attempt and failure rates use pooled event counts with named denominators; opportunity utilization and 15-node Jain fairness are equal-cell macro means.",
                "d": "Verified optimistic quality-aware service upper bound versus offered load 15 lambda; overload is asserted only where offered load exceeds the upper bound.",
            },
        },
        "validation": {
            "sealed_collision_record_count": 150,
            "sealed_panel_record_counts": data["panel_counts"],
            "logical_attempted_only_checkpoints": 20,
            "positive_control_cases": 378,
            "bootstrap_replicates": data["inference"]["inference"]["method"]["replicates"],
        },
    }
    return _canonical_json_bytes(manifest)


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            os.fchmod(handle.fileno(), 0o644)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def generate(*, output_dir: Path, check: bool) -> dict[str, str]:
    data = load_and_validate()
    figures = render_all(data)
    outputs = dict(figures)
    outputs["publication_figures.manifest.json"] = build_manifest(data, figures)
    for name, payload in outputs.items():
        path = output_dir / name
        if check:
            if not path.is_file() or path.is_symlink():
                raise FileNotFoundError(f"missing generated artifact for --check: {path}")
            if path.read_bytes() != payload:
                raise ValueError(f"stale or non-deterministic generated artifact: {path}")
        else:
            _atomic_write(path, payload)
    return {name: _sha256_bytes(payload) for name, payload in outputs.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--check", action="store_true", help="render in memory and verify committed artifacts byte-for-byte")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    hashes = generate(output_dir=args.output_dir.resolve(), check=args.check)
    print(json.dumps(hashes, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
