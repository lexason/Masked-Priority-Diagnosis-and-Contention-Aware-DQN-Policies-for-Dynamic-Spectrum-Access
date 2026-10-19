"""Adversarially probe feature sensitivity of the eight frozen DQNs.

This audit complements, but does not replace, the finite random-intervention
test in :mod:`audit_priority_collapse`.  For every checkpoint, non-empty legal
mask, and legal alternative to the fitted priority action, it minimizes the
corresponding pairwise Q-margin by projected gradient descent over the 24
populated non-mask feature coordinates.  Occupancy is fixed by the mask, the
remaining compatibility coordinates stay at zero, and all varied coordinates
are constrained to [0, 1].

The search is deliberately reported as a falsification attempt rather than a
global certificate: a failure to find a crossing does not prove that none
exists in a piecewise-linear network.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
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
RESTARTS = 128
STEPS = 100
STEP_SIZE = 0.02
RANDOM_BATCH_SIZE = 10_000
BALANCED_RANDOM_DRAWS = 1_000_000
STRESS_RANDOM_DRAWS = 10_000_000
STRESS_SEED = 202
INPUT_START = N_CHANNELS
INPUT_STOP = 5 * N_CHANNELS

DEFAULT_CHECKPOINT_ROOT = DEFAULT_PROJECT_ROOT / "checkpoints"
DEFAULT_PRIORITY_AUDIT = DEFAULT_PROJECT_ROOT / "reproduced/priority_collapse_audit.json"
DEFAULT_OUTPUT = DEFAULT_PROJECT_ROOT / "reproduced/feature_adversarial_audit.json"


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


def _orders(priority_audit: Path) -> dict[int, tuple[int, ...]]:
    if not priority_audit.is_file():
        raise FileNotFoundError(
            f"Priority-audit JSON not found: {priority_audit}. Run "
            "audit_priority_collapse.py first or pass --priority-audit."
        )
    payload = json.loads(priority_audit.read_text(encoding="utf-8"))
    result: dict[int, tuple[int, ...]] = {}
    for record in payload["per_checkpoint"]:
        result[int(record["training_seed"])] = tuple(
            int(value) for value in record["fitted_priority_high_to_low"].split(">")
        )
    if set(result) != set(TRAINING_SEEDS):
        raise RuntimeError("Priority audit does not contain the expected checkpoints")
    return result


def _selected(order: tuple[int, ...], mask_bits: int) -> int:
    for channel in order:
        if mask_bits & (1 << channel):
            return channel
    raise ValueError("A non-empty legal mask is required")


def _checkpoint(checkpoint_root: Path, seed: int) -> Path:
    checkpoint = checkpoint_root / f"seed_{seed}" / f"dqn_seed{seed}_final.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Required DQN checkpoint is missing: {checkpoint}. "
            "Pass --checkpoint-root pointing to the directory containing seed_* folders."
        )
    return checkpoint


def _random_cube_search(
    network: torch.nn.Module,
    order: tuple[int, ...],
    *,
    draws: int,
    generator_seed: int,
) -> dict[str, object]:
    """Stream joint [0,1]^24 draws without retaining the full tensor."""

    if draws % RANDOM_BATCH_SIZE:
        raise ValueError("Random-search draws must be a multiple of the batch size")
    generator = torch.Generator(device="cpu").manual_seed(generator_seed)
    flip_counts: Counter[tuple[int, int, int]] = Counter()
    maximum_winning_delta = 0.0
    maximum_witness: dict[str, object] | None = None
    with torch.no_grad():
        for _ in range(draws // RANDOM_BATCH_SIZE):
            mask_bits = torch.randint(
                1,
                1 << N_CHANNELS,
                (RANDOM_BATCH_SIZE,),
                generator=generator,
            )
            bit_values = 1 << torch.arange(N_CHANNELS)
            legal = (mask_bits[:, None] & bit_values[None, :]) != 0
            states = torch.zeros((RANDOM_BATCH_SIZE, 128), dtype=torch.float32)
            states[:, :N_CHANNELS] = (~legal).to(dtype=torch.float32)
            states[:, INPUT_START:INPUT_STOP] = torch.rand(
                (RANDOM_BATCH_SIZE, INPUT_STOP - INPUT_START),
                generator=generator,
            )
            expected = torch.full((RANDOM_BATCH_SIZE,), -1, dtype=torch.int64)
            for channel in order:
                take = (expected < 0) & legal[:, channel]
                expected[take] = channel
            if bool((expected < 0).any().item()):
                raise AssertionError("Nonempty mask did not yield a priority action")
            q_values = network(states)[:, :N_CHANNELS]
            executed = q_values.masked_fill(~legal, float("-inf")).argmax(dim=1)
            flipped = executed != expected
            for row_index in torch.nonzero(flipped, as_tuple=False).flatten().tolist():
                mask = int(mask_bits[row_index].item())
                priority_action = int(expected[row_index].item())
                executed_action = int(executed[row_index].item())
                flip_counts[(mask, priority_action, executed_action)] += 1
                delta_tensor = (
                    q_values[row_index, executed_action]
                    - q_values[row_index, priority_action]
                )
                delta = float(delta_tensor.item())
                if delta > maximum_winning_delta:
                    maximum_winning_delta = delta
                    maximum_witness = {
                        "legal_mask_bits": mask,
                        "priority_action": priority_action,
                        "executed_action": executed_action,
                        "winning_q_delta": delta,
                        "dynamic_features": [
                            float(value)
                            for value in states[
                                row_index, INPUT_START:INPUT_STOP
                            ].tolist()
                        ],
                    }
    return {
        "draws": draws,
        "generator_seed": generator_seed,
        "flips": sum(flip_counts.values()),
        "flip_rate": sum(flip_counts.values()) / draws,
        "distinct_flip_masks": len({key[0] for key in flip_counts}),
        "flip_counts": [
            {
                "legal_mask_bits": mask,
                "priority_action": priority,
                "executed_action": executed,
                "count": count,
            }
            for (mask, priority, executed), count in sorted(flip_counts.items())
        ],
        "maximum_winning_q_delta": maximum_winning_delta,
        "maximum_witness": maximum_witness,
    }


def _audit_checkpoint(
    seed: int,
    order: tuple[int, ...],
    checkpoint_root: Path,
    dqn_agent_class: type,
) -> dict[str, object]:
    agent = dqn_agent_class(
        node_id=f"feature_audit_{seed}", n_channels=N_CHANNELS, seed=seed
    )
    agent.load_checkpoint(_checkpoint(checkpoint_root, seed))
    network = agent._online  # Read-only audit of the archived executable.
    network.eval()

    balanced_random = _random_cube_search(
        network,
        order,
        draws=BALANCED_RANDOM_DRAWS,
        generator_seed=20_261_018 + seed,
    )

    problems: list[tuple[int, int, int]] = []
    for mask_bits in range(1, 1 << N_CHANNELS):
        selected = _selected(order, mask_bits)
        for alternative in range(N_CHANNELS):
            if alternative != selected and mask_bits & (1 << alternative):
                problems.append((mask_bits, selected, alternative))

    n_problems = len(problems)
    if n_problems != 129:
        raise AssertionError(f"Expected 129 mask-target problems, found {n_problems}")

    rng = np.random.default_rng(20_261_017 + seed)
    states = np.zeros((n_problems, RESTARTS, 128), dtype=np.float32)
    selected_actions = np.empty((n_problems, RESTARTS), dtype=np.int64)
    alternative_actions = np.empty((n_problems, RESTARTS), dtype=np.int64)
    for problem_index, (mask_bits, selected, alternative) in enumerate(problems):
        occupancy = np.asarray(
            [not bool(mask_bits & (1 << channel)) for channel in range(N_CHANNELS)],
            dtype=np.float32,
        )
        states[problem_index, :, :N_CHANNELS] = occupancy
        states[problem_index, :, INPUT_START:INPUT_STOP] = rng.random(
            (RESTARTS, INPUT_STOP - INPUT_START), dtype=np.float32
        )
        # Include three deterministic starting points in every problem.
        states[problem_index, 0, INPUT_START:INPUT_STOP] = 0.5
        states[problem_index, 1, INPUT_START:INPUT_STOP] = 0.0
        states[problem_index, 2, INPUT_START:INPUT_STOP] = 1.0
        selected_actions[problem_index, :] = selected
        alternative_actions[problem_index, :] = alternative

    x = torch.from_numpy(states.reshape(-1, 128)).requires_grad_(True)
    selected_tensor = torch.from_numpy(selected_actions.reshape(-1))
    alternative_tensor = torch.from_numpy(alternative_actions.reshape(-1))
    row = torch.arange(x.shape[0])

    def margins() -> torch.Tensor:
        q_values = network(x)[:, :N_CHANNELS]
        return q_values[row, selected_tensor] - q_values[row, alternative_tensor]

    initial_margin = margins()
    initial_margin.sum().backward()
    if x.grad is None:
        raise RuntimeError("Feature audit did not produce input gradients")
    initial_dynamic_gradient = x.grad[:, INPUT_START:INPUT_STOP].detach()
    initial_gradient_l1 = initial_dynamic_gradient.abs().sum(dim=1)
    nonzero_initial_gradients = int(torch.count_nonzero(initial_gradient_l1 > 0).item())
    max_initial_gradient_l1 = float(initial_gradient_l1.max().item())
    median_initial_gradient_l1 = float(initial_gradient_l1.median().item())
    x.grad = None

    initial_matrix = initial_margin.detach().reshape(n_problems, RESTARTS)
    best, best_restart = initial_matrix.min(dim=1)
    best_states = x.detach().reshape(n_problems, RESTARTS, 128)[
        torch.arange(n_problems), best_restart
    ].clone()
    initial_minimum = float(best.min().item())
    for _ in range(STEPS):
        margin = margins()
        margin.sum().backward()
        if x.grad is None:
            raise RuntimeError("Projected-gradient search lost its gradients")
        with torch.no_grad():
            dynamic = x[:, INPUT_START:INPUT_STOP]
            dynamic.sub_(STEP_SIZE * x.grad[:, INPUT_START:INPUT_STOP].sign())
            dynamic.clamp_(0.0, 1.0)
        x.grad = None
        with torch.no_grad():
            current_matrix = margins().reshape(n_problems, RESTARTS)
            current, current_restart = current_matrix.min(dim=1)
            improved = current < best
            if bool(improved.any().item()):
                current_states = x.detach().reshape(n_problems, RESTARTS, 128)[
                    torch.arange(n_problems), current_restart
                ]
                best_states[improved] = current_states[improved]
                best[improved] = current[improved]

    crossings = best <= 0.0
    minimum_index = int(torch.argmin(best).item())
    mask_bits, selected, alternative = problems[minimum_index]
    witnesses: list[dict[str, object]] = []
    for problem_index in torch.nonzero(crossings, as_tuple=False).flatten().tolist():
        witness_mask, witness_selected, witness_alternative = problems[problem_index]
        witness = best_states[problem_index]
        with torch.no_grad():
            q_values = network(witness)[:N_CHANNELS]
        legal_channels = [
            channel
            for channel in range(N_CHANNELS)
            if witness_mask & (1 << channel)
        ]
        executed = max(legal_channels, key=lambda channel: (float(q_values[channel]), -channel))
        witnesses.append(
            {
                "legal_mask_bits": witness_mask,
                "legal_channels": legal_channels,
                "priority_action": witness_selected,
                "target_alternative": witness_alternative,
                "executed_action": executed,
                "target_pairwise_q_margin": float(best[problem_index].item()),
                "dynamic_features": [
                    float(value)
                    for value in witness[INPUT_START:INPUT_STOP].tolist()
                ],
            }
        )
    distinct_crossing_masks = len(
        {int(witness["legal_mask_bits"]) for witness in witnesses}
    )
    return {
        "training_seed": seed,
        "fitted_priority_high_to_low": ">".join(str(value) for value in order),
        "mask_target_problems": n_problems,
        "restarts_per_problem": RESTARTS,
        "gradient_steps": STEPS,
        "step_size": STEP_SIZE,
        "initial_states": n_problems * RESTARTS,
        "initial_minimum_pairwise_q_margin": initial_minimum,
        "minimum_pairwise_q_margin_found": float(best.min().item()),
        "problems_with_nonpositive_margin_found": int(crossings.sum().item()),
        "distinct_masks_with_nonpositive_margin_found": distinct_crossing_masks,
        "initial_gradients_with_nonzero_l1": nonzero_initial_gradients,
        "initial_gradients_total": n_problems * RESTARTS,
        "maximum_initial_dynamic_gradient_l1": max_initial_gradient_l1,
        "median_initial_dynamic_gradient_l1": median_initial_gradient_l1,
        "minimum_margin_problem": {
            "legal_mask_bits": mask_bits,
            "priority_action": selected,
            "target_alternative": alternative,
        },
        "crossing_witnesses": witnesses,
        "balanced_random_cube_search": balanced_random,
    }


def run(
    *,
    project_root: Path = DEFAULT_PROJECT_ROOT,
    checkpoint_root: Path = DEFAULT_CHECKPOINT_ROOT,
    priority_audit: Path = DEFAULT_PRIORITY_AUDIT,
    output: Path = DEFAULT_OUTPUT,
) -> dict[str, object]:
    missing_checkpoints = [
        checkpoint_root / f"seed_{seed}" / f"dqn_seed{seed}_final.pt"
        for seed in TRAINING_SEEDS
        if not (
            checkpoint_root / f"seed_{seed}" / f"dqn_seed{seed}_final.pt"
        ).is_file()
    ]
    if missing_checkpoints:
        preview = ", ".join(str(path) for path in missing_checkpoints[:2])
        suffix = ", and others" if len(missing_checkpoints) > 2 else ""
        raise FileNotFoundError(
            f"Required DQN checkpoint(s) missing: {preview}{suffix}. "
            "Pass --checkpoint-root pointing to the directory containing seed_* folders."
        )
    dqn_agent_class = _load_dqn_agent(project_root)
    orders = _orders(priority_audit)
    records = [
        _audit_checkpoint(seed, orders[seed], checkpoint_root, dqn_agent_class)
        for seed in TRAINING_SEEDS
    ]
    total_problems = sum(int(record["mask_target_problems"]) for record in records)
    total_initial_states = sum(int(record["initial_states"]) for record in records)
    total_crossings = sum(
        int(record["problems_with_nonpositive_margin_found"]) for record in records
    )
    checkpoint_mask_pairs_with_crossings = sum(
        int(record["distinct_masks_with_nonpositive_margin_found"])
        for record in records
    )
    total_nonzero_gradients = sum(
        int(record["initial_gradients_with_nonzero_l1"]) for record in records
    )
    stress_agent = dqn_agent_class(
        node_id=f"feature_stress_{STRESS_SEED}",
        n_channels=N_CHANNELS,
        seed=STRESS_SEED,
    )
    stress_agent.load_checkpoint(_checkpoint(checkpoint_root, STRESS_SEED))
    stress_agent._online.eval()
    stress_random = _random_cube_search(
        stress_agent._online,
        orders[STRESS_SEED],
        draws=STRESS_RANDOM_DRAWS,
        generator_seed=20_261_220,
    )
    result: dict[str, object] = {
        "schema_version": 1,
        "scope": {
            "checkpoints": len(TRAINING_SEEDS),
            "nonempty_masks_per_checkpoint": 63,
            "varied_feature_coordinates": INPUT_STOP - INPUT_START,
            "feature_bounds": [0.0, 1.0],
            "fixed_coordinates": (
                "current occupancy is fixed by the legal mask; coordinates 30--127 are zero"
            ),
        },
        "method": {
            "description": (
                "targeted projected-gradient minimisation of every fitted-priority "
                "versus legal-alternative Q-margin"
            ),
            "restarts_per_problem": RESTARTS,
            "deterministic_starts": ["all 0.5", "all 0", "all 1"],
            "random_starts_per_problem": RESTARTS - 3,
            "gradient_steps": STEPS,
            "step_size": STEP_SIZE,
            "interpretation": (
                "falsification search, not a formal global robustness certificate"
            ),
        },
        "aggregate": {
            "mask_target_problems": total_problems,
            "initial_states": total_initial_states,
            "projected_gradient_updates": total_initial_states * STEPS,
            "problems_with_nonpositive_margin_found": total_crossings,
            "checkpoint_mask_pairs_with_nonpositive_margin_found": (
                checkpoint_mask_pairs_with_crossings
            ),
            "initial_gradients_with_nonzero_l1": total_nonzero_gradients,
            "initial_gradients_total": total_initial_states,
            "minimum_pairwise_q_margin_found": min(
                float(record["minimum_pairwise_q_margin_found"]) for record in records
            ),
            "balanced_random_cube_draws": len(TRAINING_SEEDS) * BALANCED_RANDOM_DRAWS,
            "balanced_random_cube_flips": sum(
                int(record["balanced_random_cube_search"]["flips"])
                for record in records
            ),
        },
        "seed_202_random_cube_stress": stress_random,
        "per_checkpoint": records,
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
        "--checkpoint-root",
        type=Path,
        default=DEFAULT_CHECKPOINT_ROOT,
        help="checkpoint directory containing seed_<seed> folders",
    )
    parser.add_argument(
        "--priority-audit",
        type=Path,
        default=DEFAULT_PRIORITY_AUDIT,
        help="priority_collapse_audit.json used to recover fitted orders",
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
            checkpoint_root=args.checkpoint_root.expanduser().resolve(),
            priority_audit=args.priority_audit.expanduser().resolve(),
            output=args.output.expanduser().resolve(),
        )
    except (FileNotFoundError, RuntimeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
