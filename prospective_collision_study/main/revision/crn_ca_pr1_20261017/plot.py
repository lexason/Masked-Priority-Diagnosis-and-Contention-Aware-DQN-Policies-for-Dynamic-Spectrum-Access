"""Create the prespecified, audit-backed collision-aware results figure."""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from main.revision.crn_ca_pr1_20261017.config import (  # noqa: E402
    ALL_POLICIES,
    COORDINATION_BLIND_POLICY_NAME,
    DEFAULT_RESULTS_ROOT,
    FACTORS,
    HISTORY_FREE_POLICY_NAME,
    LEARNED_POLICY_NAME,
    ORACLE_POLICY_NAME,
    PRIMARY_BASELINE_NAME,
    PRIMARY_GOODPUT_MARGIN,
    PRIORITY_AGREEMENT_THRESHOLD,
    PROTOCOL_VERSION,
    STUDY_ID,
    TRAINING_SEEDS,
)
from main.revision.crn_ca_pr1_20261017.provenance import (  # noqa: E402
    sha256_file,
    verify_freeze,
)
from main.revision.crn_ca_pr1_20261017.validation import validate  # noqa: E402

PRIMARY_METRIC = "network_goodput"
POLICY_LABELS = {
    LEARNED_POLICY_NAME: "CA-DQN (full)",
    HISTORY_FREE_POLICY_NAME: "History-free DQN",
    COORDINATION_BLIND_POLICY_NAME: "Coordination-blind DQN",
    PRIMARY_BASELINE_NAME: "Phase-orthogonal",
    "quality_greedy_persistent": "Quality-greedy persistent",
    "round_robin": "Round robin",
    "uniform_legal": "Uniform legal",
    "static_priority": "Static priority",
    ORACLE_POLICY_NAME: "Centralized quality matching†",
}
POLICY_COLORS = {
    LEARNED_POLICY_NAME: "#4E79A7",
    HISTORY_FREE_POLICY_NAME: "#F28E2B",
    COORDINATION_BLIND_POLICY_NAME: "#E15759",
    PRIMARY_BASELINE_NAME: "#76B7B2",
    "quality_greedy_persistent": "#59A14F",
    "round_robin": "#B07AA1",
    "uniform_legal": "#9C755F",
    "static_priority": "#BAB0AC",
    ORACLE_POLICY_NAME: "#111111",
}
POLICY_MARKERS = {
    LEARNED_POLICY_NAME: "o",
    HISTORY_FREE_POLICY_NAME: "s",
    COORDINATION_BLIND_POLICY_NAME: "^",
    PRIMARY_BASELINE_NAME: "D",
    "quality_greedy_persistent": "v",
    "round_robin": "P",
    "uniform_legal": "X",
    "static_priority": "<",
    ORACLE_POLICY_NAME: "*",
}
FIGURE_DATA_COLUMNS = (
    "panel",
    "record_type",
    "source_table",
    "metric",
    "treatment",
    "comparator",
    "policy",
    "factor_id",
    "training_seed",
    "value",
    "lower_bound",
    "upper_bound",
    "reference",
    "exact_sign_flip_p",
    "exact_priority_representable",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty analysis table: {path}")
    return rows


def _require_columns(
    rows: list[dict[str, str]], required: set[str], *, table: str
) -> None:
    missing = sorted(required - set(rows[0]))
    if missing:
        raise ValueError(f"{table} lacks required columns: {missing}")


def _finite(row: dict[str, str], key: str, *, table: str) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid {key} in {table}: {row!r}") from error
    if not np.isfinite(value):
        raise ValueError(f"non-finite {key} in {table}: {row!r}")
    return value


def _as_bool(value: str, *, label: str) -> bool:
    normalised = value.strip().lower()
    if normalised == "true":
        return True
    if normalised == "false":
        return False
    raise ValueError(f"invalid Boolean for {label}: {value!r}")


def _factor_label(factor_id: str) -> str:
    factor = next(item for item in FACTORS if item.factor_id == factor_id)
    process = "I" if factor.occupancy_process == "iid" else "M"
    return (
        f"{process}\n"
        f"{factor.occupancy_probability:.1f}/{factor.arrival_probability:.2f}"
    )


def _effect_rows(
    rows: list[dict[str, str]], comparator: str
) -> list[dict[str, str]]:
    selected = [
        row
        for row in rows
        if row["treatment"] == LEARNED_POLICY_NAME
        and row["comparator"] == comparator
        and row["metric"] == PRIMARY_METRIC
    ]
    by_factor: dict[str, dict[str, str]] = {}
    for row in selected:
        factor_id = row["factor_id"]
        if factor_id in by_factor:
            raise ValueError(f"duplicate cell effect for {comparator}/{factor_id}")
        by_factor[factor_id] = row
    expected = {factor.factor_id for factor in FACTORS}
    if set(by_factor) != expected:
        raise ValueError(
            f"cell effects for {comparator} do not match the frozen factorial design"
        )
    return [by_factor[factor.factor_id] for factor in FACTORS]


def _checkpoint_rows(
    rows: list[dict[str, str]], comparator: str
) -> list[dict[str, str]]:
    selected = [
        row
        for row in rows
        if row["comparator"] == comparator and row["metric"] == PRIMARY_METRIC
    ]
    by_seed: dict[int, dict[str, str]] = {}
    for row in selected:
        try:
            seed = int(row["training_seed"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"invalid training seed in checkpoint effects: {row!r}"
            ) from error
        if seed in by_seed:
            raise ValueError(f"duplicate checkpoint effect for {comparator}/{seed}")
        by_seed[seed] = row
    if set(by_seed) != set(TRAINING_SEEDS):
        raise ValueError(
            f"checkpoint effects for {comparator} do not match the frozen seed set"
        )
    return [by_seed[seed] for seed in TRAINING_SEEDS]


def _comparison_row(
    rows: list[dict[str, str]], comparator: str
) -> dict[str, str]:
    selected = [
        row
        for row in rows
        if row["treatment"] == LEARNED_POLICY_NAME
        and row["comparator"] == comparator
        and row["metric"] == PRIMARY_METRIC
    ]
    if len(selected) != 1:
        raise ValueError(
            f"expected one macro comparison for {comparator}; found {len(selected)}"
        )
    return selected[0]


def _factorial_matrix(rows: list[dict[str, str]]) -> np.ndarray:
    selected = [row for row in rows if row["metric"] == PRIMARY_METRIC]
    expected = {
        (policy, factor.factor_id) for policy in ALL_POLICIES for factor in FACTORS
    }
    observed: dict[tuple[str, str], float] = {}
    for row in selected:
        key = (row["policy"], row["factor_id"])
        if key in observed:
            raise ValueError(f"duplicate factorial mean for {key}")
        observed[key] = _finite(row, "mean", table="factorial_means.csv")
    if set(observed) != expected:
        raise ValueError("network-goodput factorial means are not a complete 9 x 8 grid")
    return np.asarray(
        [
            [observed[(policy, factor.factor_id)] for factor in FACTORS]
            for policy in ALL_POLICIES
        ],
        dtype=np.float64,
    )


def _mechanism_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    by_seed: dict[int, dict[str, str]] = {}
    for row in rows:
        try:
            seed = int(row["training_seed"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid training seed in mechanism table: {row!r}") from error
        if seed in by_seed:
            raise ValueError(f"duplicate mechanism certificate for {seed}")
        by_seed[seed] = row
    if set(by_seed) != set(TRAINING_SEEDS):
        raise ValueError("mechanism certificates do not match the frozen seed set")
    ordered = [by_seed[seed] for seed in TRAINING_SEEDS]
    for row in ordered:
        score = _finite(
            row,
            "validation_best_equal_cell_agreement",
            table="mechanism_certificates.csv",
        )
        if not 0.0 <= score <= 1.0:
            raise ValueError("mechanism agreement is outside [0, 1]")
        _as_bool(
            row["exact_priority_representable"],
            label="exact_priority_representable",
        )
    return ordered


def _figure_record(**values: object) -> dict[str, object]:
    return {column: values.get(column, "") for column in FIGURE_DATA_COLUMNS}


def _write_figure_data(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(FIGURE_DATA_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)


def _effect_panel(
    axis: Any,
    *,
    cell_rows: list[dict[str, str]],
    checkpoint_rows: list[dict[str, str]],
    comparison_row: dict[str, str],
    comparator: str,
    panel_title: str,
    reference: float,
) -> None:
    x = np.arange(len(FACTORS), dtype=np.float64)
    point = np.asarray(
        [_finite(row, "difference", table="cell_effects.csv") for row in cell_rows]
    )
    lower = np.asarray(
        [
            _finite(row, "one_sided_95_lcb", table="cell_effects.csv")
            for row in cell_rows
        ]
    )
    upper = np.asarray(
        [
            _finite(row, "one_sided_95_ucb", table="cell_effects.csv")
            for row in cell_rows
        ]
    )
    if np.any(lower > upper):
        raise ValueError(f"reversed cell bounds for {comparator}")
    color = POLICY_COLORS[LEARNED_POLICY_NAME]
    axis.vlines(x, lower, upper, color=color, linewidth=1.05, alpha=0.90)
    axis.scatter(
        x,
        point,
        color=color,
        edgecolor="white",
        linewidth=0.35,
        s=22,
        zorder=3,
        label="Cell effect (5th–95th bounds)",
    )

    macro_x = float(len(FACTORS) + 0.25)
    checkpoint_effects = np.asarray(
        [
            _finite(row, "paired_macro_difference", table="checkpoint_effects.csv")
            for row in checkpoint_rows
        ]
    )
    jitter = np.linspace(-0.22, 0.22, len(checkpoint_effects))
    axis.scatter(
        macro_x + jitter,
        checkpoint_effects,
        facecolors="none",
        edgecolors="#666666",
        linewidth=0.45,
        s=11,
        zorder=2,
        label="Checkpoint effect",
    )
    macro = _finite(comparison_row, "difference", table="comparisons.csv")
    macro_lower = _finite(
        comparison_row, "one_sided_95_lcb", table="comparisons.csv"
    )
    macro_upper = _finite(
        comparison_row, "one_sided_95_ucb", table="comparisons.csv"
    )
    if macro_lower > macro_upper:
        raise ValueError(f"reversed macro bounds for {comparator}")
    axis.vlines(
        macro_x, macro_lower, macro_upper, color="#111111", linewidth=1.35, zorder=3
    )
    axis.scatter(
        [macro_x],
        [macro],
        marker="D",
        color="#111111",
        edgecolor="white",
        linewidth=0.35,
        s=26,
        zorder=4,
        label="Equal-cell macro effect",
    )

    axis.axhline(0.0, color="#555555", linewidth=0.75, zorder=0)
    if reference != 0.0:
        axis.axhline(
            reference,
            color="#D55E00",
            linestyle="--",
            linewidth=0.9,
            label=f"Superiority margin = {reference:.2f}",
            zorder=0,
        )
    p_value = _finite(
        comparison_row, "exact_sign_flip_p_raw", table="comparisons.csv"
    )
    axis.text(
        0.015,
        0.975,
        f"macro $\\Delta$={macro:.3f}; exact $p$={p_value:.3g}",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=6.2,
    )
    axis.set_title(panel_title, loc="left", fontweight="bold")
    axis.set_ylabel("$\\Delta$ network goodput\n(deliveries / slot)")
    axis.set_xticks(
        [*x, macro_x],
        [*[_factor_label(factor.factor_id) for factor in FACTORS], "Macro\n+ seeds"],
        rotation=34,
        ha="right",
        rotation_mode="anchor",
    )
    axis.set_xlim(-0.55, macro_x + 0.55)
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.45, zorder=0)
    axis.legend(
        loc="lower left",
        fontsize=5.35,
        frameon=False,
        handletextpad=0.35,
        borderaxespad=0.25,
    )


def generate(results_root: Path) -> dict[str, Any]:
    """Generate a four-panel vector figure only after all production gates pass."""
    freeze = verify_freeze()
    validation = validate(results_root, smoke=False)
    if validation.get("status") != "PASS":
        raise ValueError("production validation did not pass")

    analysis_dir = results_root / "analysis"
    source_paths = {
        "summary": analysis_dir / "summary.json",
        "comparisons": analysis_dir / "comparisons.csv",
        "cell_effects": analysis_dir / "cell_effects.csv",
        "checkpoint_effects": analysis_dir / "checkpoint_effects.csv",
        "factorial_means": analysis_dir / "factorial_means.csv",
        "mechanism_certificates": analysis_dir / "mechanism_certificates.csv",
    }
    missing = [str(path) for path in source_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing required analysis artifacts: {missing}")

    data_path = analysis_dir / "figure_collision_aware_data.csv"
    figure_path = analysis_dir / "fig_collision_aware_results.pdf"
    manifest_path = analysis_dir / "FIGURE_MANIFEST.json"
    outputs = (data_path, figure_path, manifest_path)
    existing = [str(path) for path in outputs if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite figure artifacts: {existing}")

    analysis_summary = json.loads(source_paths["summary"].read_text(encoding="utf-8"))
    if analysis_summary.get("study_id") != STUDY_ID:
        raise ValueError("analysis summary study ID mismatch")
    if analysis_summary.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("analysis summary protocol version mismatch")
    if analysis_summary.get("analysis_status") != "COMPLETE":
        raise ValueError("analysis summary is not complete")

    comparisons = _read_csv(source_paths["comparisons"])
    cell_effects = _read_csv(source_paths["cell_effects"])
    checkpoint_effects = _read_csv(source_paths["checkpoint_effects"])
    factorial_means = _read_csv(source_paths["factorial_means"])
    mechanisms = _read_csv(source_paths["mechanism_certificates"])
    _require_columns(
        comparisons,
        {
            "treatment",
            "comparator",
            "metric",
            "difference",
            "one_sided_95_lcb",
            "one_sided_95_ucb",
            "exact_sign_flip_p_raw",
        },
        table="comparisons.csv",
    )
    _require_columns(
        cell_effects,
        {
            "treatment",
            "comparator",
            "metric",
            "factor_id",
            "difference",
            "one_sided_95_lcb",
            "one_sided_95_ucb",
        },
        table="cell_effects.csv",
    )
    _require_columns(
        checkpoint_effects,
        {
            "metric",
            "comparator",
            "role",
            "training_seed",
            "paired_macro_difference",
        },
        table="checkpoint_effects.csv",
    )
    _require_columns(
        factorial_means,
        {"factor_id", "policy", "metric", "mean"},
        table="factorial_means.csv",
    )
    _require_columns(
        mechanisms,
        {
            "training_seed",
            "validation_best_equal_cell_agreement",
            "exact_priority_representable",
        },
        table="mechanism_certificates.csv",
    )

    phase_cells = _effect_rows(cell_effects, PRIMARY_BASELINE_NAME)
    history_cells = _effect_rows(cell_effects, HISTORY_FREE_POLICY_NAME)
    phase_checkpoints = _checkpoint_rows(checkpoint_effects, PRIMARY_BASELINE_NAME)
    history_checkpoints = _checkpoint_rows(
        checkpoint_effects, HISTORY_FREE_POLICY_NAME
    )
    phase_comparison = _comparison_row(comparisons, PRIMARY_BASELINE_NAME)
    history_comparison = _comparison_row(comparisons, HISTORY_FREE_POLICY_NAME)
    policy_matrix = _factorial_matrix(factorial_means)
    mechanism_rows = _mechanism_rows(mechanisms)

    figure_data: list[dict[str, object]] = []
    effect_specifications = (
        (
            "a",
            PRIMARY_BASELINE_NAME,
            phase_cells,
            phase_checkpoints,
            phase_comparison,
            PRIMARY_GOODPUT_MARGIN,
        ),
        (
            "b",
            HISTORY_FREE_POLICY_NAME,
            history_cells,
            history_checkpoints,
            history_comparison,
            0.0,
        ),
    )
    for panel, comparator, cells, checkpoints_rows, comparison, reference in (
        effect_specifications
    ):
        for factor, row in zip(FACTORS, cells, strict=True):
            figure_data.append(
                _figure_record(
                    panel=panel,
                    record_type="cell_effect",
                    source_table="cell_effects.csv",
                    metric=PRIMARY_METRIC,
                    treatment=LEARNED_POLICY_NAME,
                    comparator=comparator,
                    factor_id=factor.factor_id,
                    value=_finite(row, "difference", table="cell_effects.csv"),
                    lower_bound=_finite(
                        row, "one_sided_95_lcb", table="cell_effects.csv"
                    ),
                    upper_bound=_finite(
                        row, "one_sided_95_ucb", table="cell_effects.csv"
                    ),
                    reference=reference,
                )
            )
        for seed, row in zip(TRAINING_SEEDS, checkpoints_rows, strict=True):
            figure_data.append(
                _figure_record(
                    panel=panel,
                    record_type="checkpoint_effect",
                    source_table="checkpoint_effects.csv",
                    metric=PRIMARY_METRIC,
                    treatment=LEARNED_POLICY_NAME,
                    comparator=comparator,
                    training_seed=seed,
                    value=_finite(
                        row,
                        "paired_macro_difference",
                        table="checkpoint_effects.csv",
                    ),
                    reference=reference,
                )
            )
        figure_data.append(
            _figure_record(
                panel=panel,
                record_type="equal_cell_macro_effect",
                source_table="comparisons.csv",
                metric=PRIMARY_METRIC,
                treatment=LEARNED_POLICY_NAME,
                comparator=comparator,
                value=_finite(comparison, "difference", table="comparisons.csv"),
                lower_bound=_finite(
                    comparison, "one_sided_95_lcb", table="comparisons.csv"
                ),
                upper_bound=_finite(
                    comparison, "one_sided_95_ucb", table="comparisons.csv"
                ),
                reference=reference,
                exact_sign_flip_p=_finite(
                    comparison, "exact_sign_flip_p_raw", table="comparisons.csv"
                ),
            )
        )
    for policy_index, policy in enumerate(ALL_POLICIES):
        for factor_index, factor in enumerate(FACTORS):
            figure_data.append(
                _figure_record(
                    panel="c",
                    record_type="factorial_cell_mean",
                    source_table="factorial_means.csv",
                    metric=PRIMARY_METRIC,
                    policy=policy,
                    factor_id=factor.factor_id,
                    value=float(policy_matrix[policy_index, factor_index]),
                )
            )
    for seed, row in zip(TRAINING_SEEDS, mechanism_rows, strict=True):
        figure_data.append(
            _figure_record(
                panel="d",
                record_type="checkpoint_mechanism",
                source_table="mechanism_certificates.csv",
                policy=LEARNED_POLICY_NAME,
                training_seed=seed,
                value=_finite(
                    row,
                    "validation_best_equal_cell_agreement",
                    table="mechanism_certificates.csv",
                ),
                reference=PRIORITY_AGREEMENT_THRESHOLD,
                exact_priority_representable=_as_bool(
                    row["exact_priority_representable"],
                    label="exact_priority_representable",
                ),
            )
        )
    _write_figure_data(data_path, figure_data)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.0,
            "axes.labelsize": 7.4,
            "axes.titlesize": 8.1,
            "axes.linewidth": 0.65,
            "xtick.labelsize": 6.0,
            "ytick.labelsize": 6.4,
            "legend.fontsize": 5.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )
    figure, axes = plt.subplots(2, 2, figsize=(7.16, 5.80), constrained_layout=True)
    _effect_panel(
        axes[0, 0],
        cell_rows=phase_cells,
        checkpoint_rows=phase_checkpoints,
        comparison_row=phase_comparison,
        comparator=PRIMARY_BASELINE_NAME,
        panel_title="(a) CA-DQN − phase-orthogonal",
        reference=PRIMARY_GOODPUT_MARGIN,
    )
    _effect_panel(
        axes[0, 1],
        cell_rows=history_cells,
        checkpoint_rows=history_checkpoints,
        comparison_row=history_comparison,
        comparator=HISTORY_FREE_POLICY_NAME,
        panel_title="(b) CA-DQN − history-free DQN",
        reference=0.0,
    )

    axis = axes[1, 0]
    y = np.arange(len(ALL_POLICIES), dtype=np.float64)
    cell_jitter = np.linspace(-0.16, 0.16, len(FACTORS))
    for index, policy in enumerate(ALL_POLICIES):
        values = policy_matrix[index]
        axis.scatter(
            values,
            y[index] + cell_jitter,
            color=POLICY_COLORS[policy],
            marker=POLICY_MARKERS[policy],
            s=10 if policy != ORACLE_POLICY_NAME else 18,
            alpha=0.70,
            linewidth=0.25,
            edgecolor="white" if policy != ORACLE_POLICY_NAME else "#111111",
            zorder=2,
        )
        axis.scatter(
            [float(np.mean(values))],
            [y[index]],
            color=POLICY_COLORS[policy],
            marker="*" if policy == ORACLE_POLICY_NAME else "D",
            s=43 if policy == ORACLE_POLICY_NAME else 25,
            linewidth=0.45,
            edgecolor="white" if policy != ORACLE_POLICY_NAME else "#111111",
            zorder=3,
        )
    axis.axhline(len(ALL_POLICIES) - 1.5, color="#777777", linewidth=0.7)
    axis.set_yticks(y, [POLICY_LABELS[policy] for policy in ALL_POLICIES])
    axis.invert_yaxis()
    axis.set_title(
        "(c) Policy operating points across 8 cells",
        loc="left",
        fontweight="bold",
    )
    axis.set_xlabel("Network goodput (deliveries / slot)")
    axis.grid(axis="x", color="#D9D9D9", linewidth=0.45, zorder=0)
    axis.scatter([], [], color="#777777", s=10, label="Cell mean")
    axis.scatter(
        [], [], color="#777777", marker="D", s=25, label="Equal-cell mean"
    )
    axis.legend(loc="upper right", frameon=False, ncol=2, handletextpad=0.3)
    axis.set_xlabel(
        "Network goodput (deliveries / slot)\n"
        "† Centralized oracle is descriptive; no superiority inference"
    )

    axis = axes[1, 1]
    agreement = np.asarray(
        [
            _finite(
                row,
                "validation_best_equal_cell_agreement",
                table="mechanism_certificates.csv",
            )
            for row in mechanism_rows
        ]
    )
    representable = np.asarray(
        [
            _as_bool(
                row["exact_priority_representable"],
                label="exact_priority_representable",
            )
            for row in mechanism_rows
        ],
        dtype=bool,
    )
    checkpoints = np.arange(1, len(TRAINING_SEEDS) + 1)
    axis.scatter(
        checkpoints[~representable],
        agreement[~representable],
        color="#0072B2",
        edgecolor="white",
        linewidth=0.4,
        s=25,
        label="Not exactly representable",
        zorder=3,
    )
    axis.scatter(
        checkpoints[representable],
        agreement[representable],
        color="#D55E00",
        marker="X",
        edgecolor="#111111",
        linewidth=0.35,
        s=31,
        label="Exactly representable",
        zorder=4,
    )
    axis.axhline(
        PRIORITY_AGREEMENT_THRESHOLD,
        color="#D55E00",
        linestyle="--",
        linewidth=0.9,
        label=f"Threshold = {PRIORITY_AGREEMENT_THRESHOLD:.2f}",
    )
    axis.set_title(
        "(d) Best fixed-priority validation agreement",
        loc="left",
        fontweight="bold",
    )
    axis.set_xlabel("Final checkpoint (prespecified seed order)")
    axis.set_ylabel("Equal-cell agreement")
    axis.set_xticks(checkpoints, [str(value) for value in checkpoints], rotation=0)
    axis.set_xlim(0.35, len(TRAINING_SEEDS) + 0.65)
    axis.set_ylim(0.0, 1.015)
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.45, zorder=0)
    axis.legend(loc="lower right", frameon=False, handletextpad=0.35)

    pdf_buffer = io.BytesIO()
    figure.savefig(
        pdf_buffer,
        format="pdf",
        bbox_inches="tight",
        metadata={
            "Title": f"{STUDY_ID}: collision-aware preregistered results",
            "Author": "ChainRadioNet-AI study pipeline",
            "Subject": (
                "Prespecified efficacy, ablation, policy, and mechanism results"
            ),
            "Keywords": (
                "dynamic spectrum access; collision-aware DQN; reproducibility"
            ),
            "Creator": "ChainRadioNet-AI deterministic plotting stage",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(figure)
    with figure_path.open("xb") as handle:
        handle.write(pdf_buffer.getvalue())

    evaluation_summary = results_root / "evaluation" / "summary.csv"
    if not evaluation_summary.is_file():
        raise FileNotFoundError(evaluation_summary)
    source_hashes = {
        path.relative_to(results_root).as_posix(): sha256_file(path)
        for path in (*source_paths.values(), evaluation_summary)
    }
    manifest = {
        "schema_version": 2,
        "study_id": STUDY_ID,
        "protocol_version": PROTOCOL_VERSION,
        "frozen_inputs_sha256": freeze.get("aggregate_frozen_inputs_sha256", ""),
        "source_summary": evaluation_summary.relative_to(results_root).as_posix(),
        "source_summary_sha256": sha256_file(evaluation_summary),
        "analysis_summary": source_paths["summary"]
        .relative_to(results_root)
        .as_posix(),
        "analysis_summary_sha256": sha256_file(source_paths["summary"]),
        "source_sha256": dict(sorted(source_hashes.items())),
        "figure_data": data_path.relative_to(results_root).as_posix(),
        "figure_data_sha256": sha256_file(data_path),
        "figure": figure_path.relative_to(results_root).as_posix(),
        "figure_sha256": sha256_file(figure_path),
        "panels": 4,
        "panel_descriptions": {
            "a": (
                "Full collision-aware DQN minus phase-orthogonal network-goodput "
                "effects by factorial cell, with crossed-bootstrap bounds, the "
                "prespecified 0.05 margin, checkpoint effects, and macro estimate."
            ),
            "b": (
                "Full collision-aware DQN minus history-free DQN network-goodput "
                "effects by factorial cell, with crossed-bootstrap bounds, the zero "
                "reference, checkpoint effects, and macro estimate."
            ),
            "c": (
                "All nine policies' eight factorial-cell network-goodput means and "
                "equal-cell means; the centralized oracle is descriptive only."
            ),
            "d": (
                "Per-checkpoint validation best fixed-priority agreement and exact "
                "finite-support priority representability against the 0.95 threshold."
            ),
        },
        "policy_labels": {
            policy: POLICY_LABELS[policy] for policy in ALL_POLICIES
        },
        "policy_colors": {
            policy: POLICY_COLORS[policy] for policy in ALL_POLICIES
        },
        "path_convention": "all paths are relative to the results root",
    }
    with manifest_path.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return manifest

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    args = parser.parse_args()
    print(json.dumps(generate(args.results_root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
