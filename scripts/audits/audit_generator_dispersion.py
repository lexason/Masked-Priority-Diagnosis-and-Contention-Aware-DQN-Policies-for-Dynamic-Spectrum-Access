"""Locate where the factorial dispersion perturbation stops affecting outcomes.

Pairs every snr0 trace with its snrm5 counterpart while holding process,
occupancy, checkpoint and environment seed fixed.  The archived traces expose
observation hashes, legal masks, actions, selected quality and acceptance, so
the audit can distinguish a perturbation that never reaches observations from
one that reaches observations but not decisions or endpoint labels.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRACE_ROOT = PROJECT_ROOT / "traces/factorial/runs"
DEFAULT_OUTPUT = PROJECT_ROOT / "reproduced/generator_dispersion_path_audit.json"
TRAINING_SEEDS = (42, 7, 1007, 2007, 101, 202, 303, 404)
ENVIRONMENT_SEEDS = (2610000, 2620000, 2630000, 2640000, 2650000)


def _path(
    trace_root: Path,
    process: str,
    occupancy: int,
    level: str,
    seed: int,
    env: int,
) -> Path:
    return (
        trace_root
        / f"{process}_o{occupancy}_{level}"
        / f"dqn_trainseed_{seed}"
        / f"env_{env}.csv.gz"
    )


def _audit_pair(path_0: Path, path_m5: Path) -> Counter:
    if not path_0.is_file() or not path_m5.is_file():
        raise FileNotFoundError(
            f"Required paired raw traces are missing: {path_0} / {path_m5}. "
            "Check --trace-root and obtain the complete factorial trace archive."
        )
    counts: Counter = Counter()
    with (
        gzip.open(path_0, "rt", encoding="utf-8", newline="") as left_stream,
        gzip.open(path_m5, "rt", encoding="utf-8", newline="") as right_stream,
    ):
        left_reader = csv.DictReader(left_stream)
        right_reader = csv.DictReader(right_stream)
        for left, right in zip(left_reader, right_reader, strict=True):
            key_left = (left["node_id"], left["step"])
            key_right = (right["node_id"], right["step"])
            if key_left != key_right:
                raise ValueError(f"Row alignment differs: {key_left} != {key_right}")
            counts["rows"] += 1
            counts["observation_hash_changed"] += (
                left["observation_hash"] != right["observation_hash"]
            )
            counts["legal_mask_changed"] += left["legal_action_mask"] != right["legal_action_mask"]
            counts["action_changed"] += left["channel_selected"] != right["channel_selected"]
            counts["attempt_changed"] += left["attempted"] != right["attempted"]
            counts["acceptance_changed"] += left["accepted"] != right["accepted"]
            quality_left = float(left["selected_quality"])
            quality_right = float(right["selected_quality"])
            quality_delta = abs(quality_left - quality_right)
            counts["selected_quality_changed"] += quality_delta > 0.0
            counts["selected_quality_abs_delta_sum"] += quality_delta
            counts["selected_quality_abs_delta_max"] = max(
                counts["selected_quality_abs_delta_max"], quality_delta
            )
    return counts


def run(
    *,
    trace_root: Path = DEFAULT_TRACE_ROOT,
    output: Path = DEFAULT_OUTPUT,
) -> dict[str, object]:
    if not trace_root.is_dir():
        raise FileNotFoundError(
            f"Factorial raw-trace root not found: {trace_root}. "
            "Pass --trace-root pointing to the directory containing P_o30_snr0 and "
            "the other factorial cells."
        )
    strata: dict[str, dict[str, float | int]] = {}
    total: Counter = Counter()
    overall_quality_abs_delta_max = 0.0
    for process in ("P", "U"):
        for occupancy in (30, 60):
            aggregate: Counter = Counter()
            stratum_quality_abs_delta_max = 0.0
            for training_seed in TRAINING_SEEDS:
                for environment_seed in ENVIRONMENT_SEEDS:
                    pair = _audit_pair(
                        _path(
                            trace_root,
                            process,
                            occupancy,
                            "snr0",
                            training_seed,
                            environment_seed,
                        ),
                        _path(
                            trace_root,
                            process,
                            occupancy,
                            "snrm5",
                            training_seed,
                            environment_seed,
                        ),
                    )
                    stratum_quality_abs_delta_max = max(
                        stratum_quality_abs_delta_max,
                        float(pair.pop("selected_quality_abs_delta_max", 0.0)),
                    )
                    aggregate.update(pair)
            aggregate["selected_quality_abs_delta_max"] = stratum_quality_abs_delta_max
            rows = aggregate["rows"]
            strata[f"{process}_o{occupancy}"] = dict(sorted(aggregate.items())) | {
                "observation_hash_change_fraction": aggregate["observation_hash_changed"] / rows,
                "legal_mask_change_fraction": aggregate["legal_mask_changed"] / rows,
                "action_change_fraction": aggregate["action_changed"] / rows,
                "acceptance_change_fraction": aggregate["acceptance_changed"] / rows,
                "selected_quality_change_fraction": aggregate["selected_quality_changed"] / rows,
                "selected_quality_mean_absolute_delta": (
                    aggregate["selected_quality_abs_delta_sum"] / rows
                ),
            }
            total.update(
                {
                    key: value
                    for key, value in aggregate.items()
                    if key != "selected_quality_abs_delta_max"
                }
            )
            overall_quality_abs_delta_max = max(
                overall_quality_abs_delta_max,
                stratum_quality_abs_delta_max,
            )

    rows = total["rows"]
    result: dict[str, object] = {
        "pairing": {
            "factors_held_fixed": [
                "occupancy process",
                "nominal occupancy",
                "DQN training seed",
                "environment master seed",
                "node",
                "cycle",
            ],
            "contrast": "snr_db=0 minus snr_db=-5 generator-dispersion configuration",
            "checkpoint_count": len(TRAINING_SEEDS),
            "environment_count_per_cell": len(ENVIRONMENT_SEEDS),
        },
        "aggregate": {
            key: value
            for key, value in sorted(total.items())
            if key != "selected_quality_abs_delta_max"
        }
        | {
            "selected_quality_abs_delta_max": overall_quality_abs_delta_max,
            "observation_hash_change_fraction": total["observation_hash_changed"] / rows,
            "legal_mask_change_fraction": total["legal_mask_changed"] / rows,
            "action_change_fraction": total["action_changed"] / rows,
            "acceptance_change_fraction": total["acceptance_changed"] / rows,
            "selected_quality_change_fraction": total["selected_quality_changed"] / rows,
            "selected_quality_mean_absolute_delta": total["selected_quality_abs_delta_sum"] / rows,
        },
        "strata": strata,
        "interpretation": (
            "The dispersion contrast reaches the generated observation/quality path, but it "
            "does not change occupancy-derived legal masks, archived DQN actions, attempts, "
            "or acceptance labels.  The exact endpoint null is therefore not an unused-field "
            "artifact; it is an insensitivity of the audited policy/outcome path."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--trace-root",
        type=Path,
        default=DEFAULT_TRACE_ROOT,
        help="factorial raw-trace directory containing the eight factor cells",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="destination JSON audit file",
    )
    args = parser.parse_args()
    trace_root = args.trace_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    try:
        result = run(trace_root=trace_root, output=output)
    except FileNotFoundError as exc:
        parser.error(str(exc))
    print(json.dumps(result["aggregate"], indent=2, sort_keys=True))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
