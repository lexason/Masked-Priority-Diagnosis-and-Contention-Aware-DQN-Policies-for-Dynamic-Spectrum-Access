# CRN-CA-PR1 implementation readiness report

Date: 2026-10-17 (Australia/Sydney)
Status: v2 implementation independently approved and **ready for source
freeze; not frozen and not launched**

## Why an isolated environment was required

The repository's general-purpose trainer/evaluator cannot answer the new
collision question without changing its estimand. It creates one independently
seeded occupancy generator per node, accepts a lease solely from the selecting
node's local quality threshold, does not reject duplicate simultaneous channel
choices, and puts only the designated training node's transitions into replay.
Changing those modules would also contaminate antecedent studies. The new code
therefore lives entirely in `main/revision/crn_ca_pr1_20261017` and writes only
to `results/crn_ca_pr1_20261017`.

## Implemented controls

- Shared policy-independent occupancy, arrival, fading, and link-realisation
  stream, with an incremental trace SHA-256.
- Simultaneous actions and destructive collision resolution before any policy
  history update.
- Persistent packet queues and a per-run conservation identity.
- Explicit abstention and local action-dependent delivery reward.
- Shared Double-DQN with all-node replay counts exposed in metadata.
- Twenty final-iterate training seeds, three seed-paired observation variants,
  24 disjoint evaluation seeds per cell, and no outcome-based checkpoint
  selection.
- Full 2 x 2 x 2 workload factorial; five decentralised comparators including
  a phase-orthogonal scheduler with equivalent coordination cues; and a
  centralized maximum-quality matching oracle reported only as headroom.
- Each training factor is crossed nearly equally with all six cyclic channel
  permutations; evaluation uses each permutation exactly four times in every cell.
- Exact all-720 priority-order scoring, precedence certificates, balanced
  calibration/validation splits, crossed block inference, and compact
  mask/action sufficient statistics rather than pseudoreplicated rows.
- Fail-closed source/dependency/runtime freeze, production overwrite guards,
  complete Cartesian validation, verifiable result seal, and audit-backed
  vector figures.

## Verification state before freeze

The v1 transition kernel passed independent code inspection and a separate
9,600-slot property probe across every factorial cell and label shift. That
review confirmed simultaneous destructive collision semantics, local
action-dependent link delivery, exact queue conservation, legal masks,
action-independent exogenous RNG consumption, Double-DQN target masking, and
equal all-node replay. It also found weaknesses in the old comparator,
inference, validation, and seal; v1 was therefore never frozen or run.

For v2, the machine-readable training plan now contains 60 final checkpoints
(20 seeds x 3 variants), each with 400 x 250 slots and 1,500,000 inserted node
transitions. The evaluation plan contains 12,672 runs across eight cells, 24
environment blocks, four complete label cycles, 60 learned checkpoints, and six
comparator policies. The locked-dependency pre-freeze suite passes all 47 tests
and Ruff. Two clean, independently created three-variant smoke runs each
produced the exact 72 expected gradient updates per checkpoint and identical
exogenous trace hashes across variants. Their checkpoints, episode tables, and
evaluation summaries are byte-identical across reruns; metadata and validation
report hashes differ only because `wall_seconds` is deliberately retained as
operational provenance. Both smoke validations pass, the exact 60-checkpoint
and 12,672-run plans pass, and the final independent source/protocol audit is
GO. Smoke outcomes cannot justify a claim and are excluded from result sealing.

## Resource estimate

The disposable training benchmark completed 80 environment steps (72 gradient
updates and 1,200 node transitions) in approximately 0.13 s using two CPU
threads. A full-length 3,300-slot learned-policy evaluation run took
approximately 1.53 s using one CPU thread. Allowing heavily for process spawn,
full replay, filesystem work, and CPU contention, the intended 24-core tower
budget is:

| Stage | Workers | Conservative wall clock |
|---|---:|---:|
| Final v2 implementation review + two smokes | 1--3 | 30--90 min |
| Sixty 100,000-step training runs | 8 x 2 threads | 2--8 h |
| 12,672 paired evaluations | 12 x 1 thread | 1--3 h |
| Validation, 10,000 block bootstraps, figures, seal | 1 | 30--90 min |
| Total expected after freeze | -- | 4--12 h |
| Conservative contingency ceiling | -- | <20 h |

Peak replay storage is about 55 MiB per training worker before process/runtime
overhead; eight workers fit within 30 GiB. Expected scientific artifacts remain
well below 1 GiB. The programme therefore retains a substantial margin against
the 48-hour deadline even with the stronger seed count and two learned
ablations.

## Freeze decision

All pre-freeze implementation gates are complete: lint, synthetic tests, exact
plan checks, two deterministic end-to-end smokes, and independent
source/protocol re-audit. The next permitted action is `provenance.py init`,
followed immediately by `provenance.py verify` and the production commands in
`README.md`. Any inventoried source, dependency, or runtime change after that
point invalidates the manifest and must stop the run. No favourable smoke
behaviour may be cited as a result or used to exclude a production seed.
