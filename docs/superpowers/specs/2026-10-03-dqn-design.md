# DQN implementation design

## Purpose and existing decisions

Implement the RL method already specified in `EXPERIMENT_PLAN.md` section 6 and
`experiment_config.yaml`, following the user's instruction to align previous
results and continue implementation. This is the third competing method; the
exact solver remains an evaluation service. Preserve the frozen corpus and the
existing reference and baseline CSVs.

The frozen corpus contains 17,700 instances. Training and validation each contain
equal synthetic and real shares. Final evaluation uses the same 3,300 test/E3/E4
instances as the baselines. No test score controls training or checkpoint choice.

## Alignment ruling

The September 29 evaluation-service design supersedes the earlier `U = V*`
sentence: `value` is the certified integer optimum, while `upper_bound` retains
the fractional-knapsack bound. Checkpoint selection uses the stored fractional
`U` in `V/U`; exact optimality gaps use `V*`. Correct the stale documentation and
configuration wording, without rewriting reference solutions. Record SHA-256
digests of data, manifest, reference table, configuration and code with each run.

## Environment and model

Actions are 0 (skip) and 1 (include). Episodes follow the saved `item_order` and
take exactly n transitions. Selected indices always refer to the original arrays.
Include is valid iff integer weight fits remaining integer capacity. Reject an
invalid action rather than silently treating it as skip. Rewards are value divided
by the original instance value sum; evaluation totals remain integers.

`State(instance, position, remaining_capacity)` is an immutable lightweight view;
replay stores views, not repeated copies of item arrays. Terminal next state is
`None`. A batch exposes `items`, `valid`, `context`, and `action_mask`: padded
normalized remaining item pairs, a valid-item mask, the current pair plus three
context scalars, and valid actions. For zero capacity use denominator 1; all
include actions remain masked. Terminal observations are never fed to the model.

`QNetwork` uses Linear(2,64)-ReLU-Linear(64,64)-ReLU, masked mean/max pooling,
then Linear(133,128)-ReLU-Linear(128,2). Context is current normalized weight/value,
remaining capacity / max(C,1), remaining count / n, and log(1+n). No source label,
ID or optimal solution is an input. Exact Q ties choose skip. The same mask applies
to random exploration, inference, and target-network maximization.

Inference may precompute item embeddings and suffix mean/max pools while model
weights are fixed. This is algebraically equivalent to encoding remaining items
at every step. Tests must compare cached and ordinary inference. Batched validation
is allowed; latency measurements use one full solution call including encoding.

## Training

Use PyTorch on CPU by default (one thread, deterministic seeds), with optional
explicit device selection. Consume all optimizer and budget values from YAML.
Implement ordinary DQN with a copied target network, Huber loss, Adam and clipped
gradients; do not introduce Double DQN, imitation or reward changes in this phase.

`BalancedSampler.next_block()` samples one episode per available (source,n),
shuffles block order, and cycles available (family,alpha) cells evenly. A block's
source exposure is balanced for the frozen corpus. Stop only after completing the
block that crosses the requested budget. Replay keeps two bounded pools, each
half the configured total capacity, and samples without replacement within each
source. Updates wait until both source quotas and warm-up are satisfied.

Run validation at the first episode boundary after each interval and at run end.
Use source-equal mean V/U (exclude and count zero-optimum instances), additionally
record source/cell scores. Save the earliest highest-scoring checkpoint; save a
last checkpoint separately. Training checkpoints include effective config,
provenance and step; this phase does not promise exact interrupted-run resume.

## Outputs and execution scope

Separate CLI commands: `train`, `evaluate`, and `check-data`. Each run writes to a
new output directory and refuses overwrites. `train --pilot --seed 0` requests the
configured 50,000-transition pilot. The default full budget remains 300,000 for
seeds 0,1,2; a pilot is not a final experiment. A small explicit transition override
is allowed for integration tests and is recorded as a custom run.

Outputs: `run.json`, `training.csv`, `validation.csv`, `best.pt`, `last.pt`, and
training exposure counts. Evaluation writes baseline-compatible raw columns plus
checkpoint metadata and validates all selected sets independently. Final test
evaluation is a separate, explicit command; this implementation phase runs unit
tests, a small integration run, and the pilot, without declaring superiority or a
completed three-seed comparison. The later report will include learning cost,
per-source/distribution/size quality, seed variability and the baseline comparison.

## Acceptance

Test original-index mapping, zero/exact capacity, rewards, invalid actions,
masking in all three uses, terminal targets, variable sizes up to 500, padding
invariance, cached inference equivalence, balanced sampling/replay, deterministic
seeding, fractional-bound checkpoint scoring, earlier checkpoint ties, data
fingerprint rejection, output guards, and a real optimizer/checkpoint/CLI cycle.
Run the existing suite with the project's venv and a fresh workspace temporary
directory. Keep pilot evidence and implementation status distinct from final results.

Implementation reference: [PyTorch DQN tutorial](https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html).
