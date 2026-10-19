"""Positive controls for the masked-priority equivalence audit.

The principal audit finds exact masked-priority equivalence for eight trained
DQN checkpoints.  This program checks that the same fitting procedure does not
mechanically return exact equivalence for every policy operating behind a hard
legal-action mask.

Two controls are evaluated:

* archived Quality--Greedy and Round Robin actions on the same S1/S2
  calibration and held-out environment seeds used by the DQN audit; and
* the deterministically reconstructed, untrained online DQN for each training
  seed on a fixed constructed-state grid covering all 63 non-empty masks.

The untrained construction exactly follows ``DQNAgent(seed=seed)``.  For every
mask, current occupancy is fixed by that mask, the 24 populated non-mask
features are sampled independently from U[0,1], and compatibility coordinates
30--127 remain zero.  The result is a finite assay-sensitivity control, not a
claim about the distribution of untrained-policy behavior in deployment.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import itertools
import json
import sys
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch

SCRIPT = Path(__file__).resolve()


def _discover_project_root() -> Path:
    """Return the nearest checkout containing the DQN implementation."""

    candidates = (Path.cwd().resolve(), *SCRIPT.parents)
    for candidate in candidates:
        if (candidate / "ai/agents/dqn_agent.py").is_file():
            return candidate
    return Path.cwd().resolve()


DEFAULT_PROJECT_ROOT = _discover_project_root()

N_CHANNELS = 6
TRAINING_SEEDS = (42, 7, 1007, 2007, 101, 202, 303, 404)
CONTROL_POLICIES = ("quality_greedy", "round_robin")
CALIBRATION_ENVS = (2_610_000, 2_620_000, 2_630_000)
HELDOUT_ENVS = (2_640_000, 2_650_000)
SCENARIOS = ("S1", "S2")
SAMPLES_PER_MASK = 512
CONSTRUCTED_STATE_SEED_BASE = 20_261_019

DEFAULT_TRACE_ROOT = DEFAULT_PROJECT_ROOT / "traces/core/runs"
DEFAULT_OUTPUT = DEFAULT_PROJECT_ROOT / "reproduced/priority_positive_controls_audit.json"


def _load_dqn_agent(project_root: Path) -> type:
    source = project_root / "ai/agents/dqn_agent.py"
    if not source.is_file():
        raise FileNotFoundError(
            f"DQN project source not found at {source}. "
            "Pass --project-root pointing to the ChainRadioNet checkout."
        )
    project = str(project_root)
    if project not in sys.path:
        sys.path.insert(0, project)
    try:
        from ai.agents.dqn_agent import DQNAgent
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            f"Could not import DQNAgent from {source}. Install the project's Python "
            "dependencies and pass the correct --project-root."
        ) from exc
    return DQNAgent


def _mask_bits(raw: str) -> int:
    values = json.loads(raw)
    if len(values) != N_CHANNELS:
        raise ValueError(f"Expected {N_CHANNELS} mask entries, found {len(values)}")
    return sum(int(bool(value)) << channel for channel, value in enumerate(values))


def _predict(order: tuple[int, ...], mask_bits: int) -> int:
    for channel in order:
        if mask_bits & (1 << channel):
            return channel
    raise ValueError("Priority prediction requested for an empty legal set")


def _fit(
    counts: Counter[tuple[int, int]],
) -> tuple[tuple[int, ...], int, int]:
    scored: list[tuple[int, tuple[int, ...]]] = []
    for order in itertools.permutations(range(N_CHANNELS)):
        matches = sum(
            count
            for (mask_bits, action), count in counts.items()
            if _predict(order, mask_bits) == action
        )
        scored.append((matches, order))
    best_matches = max(score for score, _order in scored)
    best_orders = [order for score, order in scored if score == best_matches]
    return min(best_orders), best_matches, len(best_orders)


def _score(
    order: tuple[int, ...], counts: Counter[tuple[int, int]]
) -> tuple[int, int, float]:
    matches = sum(
        count
        for (mask_bits, action), count in counts.items()
        if _predict(order, mask_bits) == action
    )
    attempts = sum(counts.values())
    return matches, attempts, matches / attempts


def _summarise_counts(counts: Counter[tuple[int, int]]) -> dict[str, int]:
    masks = {mask_bits for mask_bits, _action in counts}
    multi_action_masks = sum(
        len({action for observed_mask, action in counts if observed_mask == mask_bits}) > 1
        for mask_bits in masks
    )
    return {
        "distinct_nonempty_masks": len(masks),
        "masks_with_multiple_observed_actions": multi_action_masks,
        "observed_mask_action_pairs": len(counts),
    }


def _trace_counts(paths: Iterable[Path]) -> Counter[tuple[int, int]]:
    counts: Counter[tuple[int, int]] = Counter()
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(
                f"Required raw trace is missing: {path}. Check --trace-root and "
                "obtain the complete row-level trace archive."
            )
        with gzip.open(path, mode="rt", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            required = {"legal_action_mask", "channel_selected", "attempted"}
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise ValueError(f"Missing required columns in {path}")
            for row in reader:
                if int(row["attempted"]) == 0:
                    continue
                mask_bits = _mask_bits(row["legal_action_mask"])
                action = int(row["channel_selected"])
                if not mask_bits & (1 << action):
                    raise ValueError(f"Illegal archived action in {path}")
                counts[(mask_bits, action)] += 1
    return counts


def _archived_control(policy: str, trace_root: Path) -> dict[str, object]:
    calibration = _trace_counts(
        trace_root / scenario / policy / f"env_{environment}.csv.gz"
        for scenario in SCENARIOS
        for environment in CALIBRATION_ENVS
    )
    heldout = _trace_counts(
        trace_root / scenario / policy / f"env_{environment}.csv.gz"
        for scenario in SCENARIOS
        for environment in HELDOUT_ENVS
    )
    order, calibration_best, tied_orders = _fit(calibration)
    calibration_matches, calibration_attempts, calibration_agreement = _score(
        order, calibration
    )
    heldout_matches, heldout_attempts, heldout_agreement = _score(order, heldout)
    if calibration_matches != calibration_best:
        raise AssertionError("Fitted and recomputed calibration scores disagree")
    return {
        "policy": policy,
        "fitted_priority_high_to_low": ">".join(map(str, order)),
        "tied_best_orders": tied_orders,
        "calibration": {
            "attempts": calibration_attempts,
            "matches": calibration_matches,
            "agreement": calibration_agreement,
            **_summarise_counts(calibration),
        },
        "heldout_without_refitting": {
            "attempts": heldout_attempts,
            "matches": heldout_matches,
            "agreement": heldout_agreement,
            **_summarise_counts(heldout),
        },
    }


def _constructed_states(seed: int) -> tuple[np.ndarray, np.ndarray]:
    generator = torch.Generator(device="cpu").manual_seed(
        CONSTRUCTED_STATE_SEED_BASE + seed
    )
    mask_bits = torch.arange(1, 1 << N_CHANNELS).repeat_interleave(
        SAMPLES_PER_MASK
    )
    channels = torch.arange(N_CHANNELS)
    legal = ((mask_bits[:, None] >> channels) & 1).to(dtype=torch.bool)
    states = torch.zeros((len(mask_bits), 128), dtype=torch.float32)
    states[:, :N_CHANNELS] = (~legal).to(dtype=torch.float32)
    states[:, N_CHANNELS : 5 * N_CHANNELS] = torch.rand(
        (len(mask_bits), 4 * N_CHANNELS), generator=generator
    )
    return states.numpy(), legal.numpy()


def _untrained_control(seed: int, dqn_agent_class: type) -> dict[str, object]:
    # DQNAgent applies torch.manual_seed(seed) before constructing its online
    # network, exactly as the archived training entry point does.
    agent = dqn_agent_class(
        node_id=f"untrained_control_{seed}", n_channels=N_CHANNELS, seed=seed
    )
    network = agent._online  # Read-only access to the initial executable network.
    network.eval()
    states, legal = _constructed_states(seed)
    with torch.inference_mode():
        q_values = network(torch.from_numpy(states))[:, :N_CHANNELS]
        q_values = q_values.masked_fill(~torch.from_numpy(legal), float("-inf"))
        actions = q_values.argmax(dim=1).cpu().numpy()

    counts: Counter[tuple[int, int]] = Counter()
    for mask_row, action in zip(legal, actions, strict=True):
        mask_bits = sum(
            int(is_legal) << channel for channel, is_legal in enumerate(mask_row)
        )
        counts[(mask_bits, int(action))] += 1

    order, matches, tied_orders = _fit(counts)
    attempts = sum(counts.values())
    return {
        "training_seed": seed,
        "state_rng_seed": CONSTRUCTED_STATE_SEED_BASE + seed,
        "states": attempts,
        "samples_per_nonempty_mask": SAMPLES_PER_MASK,
        "fitted_priority_high_to_low": ">".join(map(str, order)),
        "tied_best_orders": tied_orders,
        "matches": matches,
        "agreement": matches / attempts,
        **_summarise_counts(counts),
    }


def run(
    *,
    project_root: Path = DEFAULT_PROJECT_ROOT,
    trace_root: Path = DEFAULT_TRACE_ROOT,
    output: Path = DEFAULT_OUTPUT,
) -> dict[str, object]:
    if not trace_root.is_dir():
        raise FileNotFoundError(
            f"Core raw-trace root not found: {trace_root}. "
            "Pass --trace-root pointing to the directory containing S1 and S2."
        )
    dqn_agent_class = _load_dqn_agent(project_root)
    archived_controls = [
        _archived_control(policy, trace_root) for policy in CONTROL_POLICIES
    ]
    untrained_controls = [
        _untrained_control(seed, dqn_agent_class) for seed in TRAINING_SEEDS
    ]
    result: dict[str, object] = {
        "schema_version": 1,
        "purpose": (
            "Positive controls showing that exhaustive priority fitting does not "
            "mechanically yield exact equivalence for every hard-masked policy"
        ),
        "archived_controls": {
            "scenarios": list(SCENARIOS),
            "calibration_environment_seeds": list(CALIBRATION_ENVS),
            "heldout_environment_seeds": list(HELDOUT_ENVS),
            "records": archived_controls,
        },
        "untrained_dqn_controls": {
            "construction": (
                "DQNAgent initial online network reconstructed from each training seed; "
                "all 63 nonempty masks; occupancy fixed by mask; 24 dynamic features "
                "iid U[0,1]; coordinates 30--127 zero"
            ),
            "samples_per_mask": SAMPLES_PER_MASK,
            "state_rng_seed_base": CONSTRUCTED_STATE_SEED_BASE,
            "records": untrained_controls,
        },
        "interpretation_boundary": (
            "Failure of exact priority equivalence in these controls establishes assay "
            "specificity on the stated finite domains. It does not identify which "
            "training component caused the final DQN action maps."
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
        "--project-root",
        type=Path,
        default=DEFAULT_PROJECT_ROOT,
        help="ChainRadioNet checkout containing ai/agents/dqn_agent.py",
    )
    parser.add_argument(
        "--trace-root",
        type=Path,
        default=DEFAULT_TRACE_ROOT,
        help="core raw-trace directory containing scenario folders S1 and S2",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="destination JSON audit file",
    )
    args = parser.parse_args()
    try:
        result = run(
            project_root=args.project_root.expanduser().resolve(),
            trace_root=args.trace_root.expanduser().resolve(),
            output=args.output.expanduser().resolve(),
        )
    except (FileNotFoundError, RuntimeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
