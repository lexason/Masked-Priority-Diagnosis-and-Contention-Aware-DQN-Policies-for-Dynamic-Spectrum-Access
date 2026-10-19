# CRN-CA-PR1-20261017: prospective collision-aware protocol

Status: implementation candidate; production execution is prohibited until the
source freeze is initialised and verified. This protocol is outcome-independent
and belongs only to the namespace `main/revision/crn_ca_pr1_20261017`.

## Research question

Can a shared-parameter Double-DQN supplied with local outcome history improve
network goodput under simultaneous decentralised access relative to a strong,
collision-free distributed scheduler that receives the same node-identity,
availability, and common-phase cues? Does the combined local outcome-history
block contribute to that effect, and is the final policy distinguishable on
finite evaluation support from every one of the 6! masked fixed-priority
channel orders?

The study is a new prospective experiment. It neither changes nor reuses
outcomes from any discovery or replication study. Null, negative, and mixed
results will be reported without seed replacement, checkpoint selection, or
factor suppression.

## Fixed system model

- Fifteen secondary nodes and six orthogonal channels act in 100-ms slots.
- One exogenous occupancy process is shared by every node in a slot. Each node
  observes the same availability pattern and its own correlated local quality.
- Bernoulli packet arrivals enter persistent per-node queues. Packets are never
  silently dropped. The conservation identity is checked in every run.
- All actions are chosen before any outcome is resolved. If two or more nodes
  choose the same available physical channel, every such transmission fails.
- A singleton transmission succeeds when its presampled uniform link draw is
  below the selected node/channel quality. Thus delivery is local and
  action-dependent. Occupied channels are masked; abstention is explicit.
- The learned state comprises availability and quality, per-channel collision
  and delivery EWMAs, previous action/outcome, queue size, node identity, and a
  common slot-phase one-hot. It contains no future trace value or other node's
  chosen action.
- Local rewards are +1 delivery, -0.50 collision, -0.50 singleton link failure,
  -0.02 voluntary defer, and 0 for a forced defer or no packet.
- A single Double-DQN is shared across nodes. Exactly all 15 transitions enter
  uniform replay after every slot; one gradient update is attempted per slot.

## Training and selection

Twenty fixed optimisation seeds are listed in `config.py`. Each seed trains
three separately optimised, seed-paired final iterates for 400 episodes of 250
slots: the full collision-aware observation, a history-free intervention, and
a coordination-blind intervention. The history-free intervention zeros the
collision/delivery EWMAs and previous-outcome block while retaining previous
action, node identity, and common phase. The coordination-blind intervention
zeros node identity and common phase while retaining local outcome history.
Network architecture, optimiser, rewards, environment stream, replay schedule,
and all other inputs are unchanged. Episodes rotate deterministically and
nearly equally through the full crossing of all eight factorial cells and six
cyclic channel-label permutations.
The training environment uses the disjoint 41-million seed namespace.

The only admissible checkpoint is the final iterate. Intermediate performance,
evaluation performance, or manuscript convenience may not select, replace, or
exclude a seed. Hyperparameters, state fields, reward values, and architecture
are frozen in `config.py` and its canonical protocol digest.

## Confirmatory evaluation

The complete 2 x 2 x 2 factorial is:

1. occupancy dynamics: independent Bernoulli or temporally correlated Markov;
2. stationary occupancy probability: 0.30 or 0.60;
3. per-node packet-arrival probability: 0.10 or 0.25 per slot.

Every cell uses the same 24 evaluation seeds, disjoint from training and all
known antecedent namespaces. Each run has 300 burn-in slots and 3,000 measured
slots. The 24 seeds carry the six cyclic logical-to-physical channel
permutations exactly four times, preventing a favourable channel number from being
confounded with an action-head index. Every policy is replayed from a fresh
environment with the same factor, seed, and permutation; equality of exogenous
trace hashes is a hard validation gate.

All 60 learned checkpoints are evaluated. Six prespecified comparators are
evaluated once per trace. The primary comparator is phase-orthogonal access:
for phase `h`, node rank is `(node_id-h) mod 15`; a queued node transmits on the
ranked idle logical channel iff its rank is smaller than the number of idle
channels. It is decentralised, rotating, collision-free, and uses no cue denied
to the learned policy. The remaining decentralised comparators are
saturation-safe quality-greedy persistent access with
`p=min(1,idle_channels/15)`, node-phased round robin, uniform legal selection,
and masked static priority. A central maximum-current-quality one-to-one
matching policy is reported only as a descriptive oracle/headroom reference;
it is never a superiority target. No comparator is tuned on evaluation
outcomes.

## Outcomes and inference

The primary physical endpoint is network goodput, `D/T`, in successfully
delivered packets per measured slot. Because queues persist and packets have no
deadline or drop event, `D/(B0+A)` is named the finite-window service fraction,
not packet-delivery ratio, and is secondary. Other diagnostics are collision
attempt rate; singleton attempts; link failures conditional on singleton
attempts; opportunity utilisation relative to the available queued-channel
ceiling; mean and ending backlog; deferrals; action counts; and delivery Jain
fairness. Fairness is undefined, rather than one, when every node has zero
deliveries, and a zero-service indicator is retained.

For baseline `B`, the exact equal-cell estimand is

`Delta_B = (1/(R*C*E)) sum_r sum_c sum_e [Y_full(r,c,e)-Y_B(c,e)]`,

where `R=20` independent final checkpoints, `C=8` fixed equally weighted
factorial cells, and `E=24` crossed environment blocks. For a learned ablation,
the same seed index is paired before averaging. The baseline value is reused
within an environment block and is not treated as 20 independent observations.

Ten thousand preregistered bootstrap draws resample the training-seed index
once and the environment index once per draw. The same environment resample is
used jointly across all cells and policies. Environment resampling is
stratified by the six fixed label shifts, preserving four equally weighted
shift strata members per draw. Primary decisions use a one-sided 95% percentile
lower confidence bound (the 0.05 quantile) and an exact one-sided seed-level
sign-flip test conditional on the complete evaluation cube.

The three required gates for the strongest collision-aware efficacy claim are:

1. Full DQN minus phase-orthogonal network goodput has lower confidence bound
   above the prespecified practically important margin of 0.05 packet/slot and
   exact one-sided `p<0.05`. The margin means at least one additional delivered
   packet every 20 slots at the confidence boundary.
2. Seed-paired full DQN minus history-free DQN network goodput has lower
   confidence bound above zero and exact one-sided `p<0.05`. Without this gate,
   no improvement is attributed to the combined local outcome-history block.
   Because the intervention jointly removes collision EWMA, delivery EWMA, and
   previous outcome, this gate does not isolate any one component.
3. Every full checkpoint is non-representable by a masked fixed-priority order
   on complete finite evaluation support, every checkpoint's held-out
   best-order agreement is below 0.95, and the one-sided 95% block-bootstrap
   upper bound for the equal-seed macro agreement is below 0.95.

For gate 3, the first 12 environment indices (two complete label cycles) form
the calibration set and the final 12 form the validation set. All 720 channel
orders are scored; calibration ties are retained and validation uses the most
conservative (highest-agreement) tied order. A full-support precedence graph
adds an edge from every selected channel to every other available channel.
Eligible abstention or a directed cycle makes exact priority representation
impossible; acyclicity, topological-order uniqueness, edge support, best order,
tie count, and a count digest are all recorded. This is finite-support evidence,
not proof over unobserved states.

Full-versus-coordination-blind effects are diagnostic. Secondary goodput
claims versus quality-greedy persistent, round robin, uniform legal, and static
priority form one Holm-adjusted family and additionally require positive
confidence bounds. The centralized oracle is descriptive. Absolute and
relative effects, checkpoint effects, and all eight cell-specific effects are
reported regardless of sign; no cell, seed, or endpoint may be suppressed.
The required gates are an intersection-union claim and are not rescued by a
favourable secondary comparison.

The inferential unit for learned-policy variability is the independently
trained final checkpoint; environment seeds form crossed, paired evaluation
blocks. Slot and node rows are never treated as independent replicates.

## Integrity gates

Before production: locked-dependency lint, unit tests, end-to-end smoke,
resource benchmark, plan-count audit, then `provenance.py init` and
`provenance.py verify`. The freeze binds every study source, `pyproject.toml`,
`uv.lock`, the canonical protocol payload, and runtime package versions. After
a freeze exists, any inventoried change aborts production. Training must
produce exactly 60 checkpoints with equal all-node replay counts, exact
episode schedules, final-iterate metadata, and action-independent trace hashes
across the three paired variants. Evaluation must contain exactly 12,672
unique Cartesian runs (60 learned plus six comparator runs in each of 192
factor/environment blocks), complete group membership, common trace hashes,
exact packet conservation, and balanced channel permutations. Analysis begins
only after validation. Results are sealed only after validation, analysis,
figures, and their manifests exist and pass; a separate seal-verification
command must reproduce every sealed hash.

Smoke outputs are stored below `results/crn_ca_pr1_20261017/smoke`, are excluded
from confirmatory analysis and result sealing, and cannot justify a claim.

## Interpretation limits

This is a synthetic, slotted, six-channel testbed experiment, not over-the-air
proof. Node identity and common slot phase are available coordination cues;
the phase-orthogonal comparator and coordination-blind ablation expose rather
than conceal their effect. Results do not establish performance for anonymous
or asynchronous radios.
The N210 evidence elsewhere in the repository cannot be used to relabel these
collision outcomes as hardware measurements. Claims must name the factorial,
the finite seed sets, and any failed endpoint.
