# DQN implementation and execution

The implementation follows the frozen experimental design: sequential skip/include
decisions, feasible-action masking, a shared item encoder with mean/max pooling,
balanced episode sampling, 32/32 synthetic/real replay, and ordinary DQN targets.
The architecture handles all planned sizes, including the n=500 stress set.

Verified on 2026-10-03: **473 tests passed**. All three formal seeds completed
300,440 transitions, and all 9,900 held-out DQN selections passed independent
feasibility/value/hash checks. See [the full experimental report](EXPERIMENT_REPORT.md)
for results and reproduction commands. The earlier seed-0 pilot completed 50,320
transitions and remains documented separately in [the pilot report](RL_PILOT_REPORT.md).
Formal best checkpoints for seeds 0/1/2 are steps 250,100 / 300,020 / 150,020.

## Alignment with completed work

| Artifact | Existing contents | RL use |
| --- | --- | --- |
| `data_manifest.json` and processed JSONL | 12,000 train, 2,400 validation, 2,400 test, 600 E3, 300 E4 | Verify file SHA-256 and record them per run |
| `reference_results.csv` | 17,700 certified integer-DP references | Use `value` as V* for exact gaps and `upper_bound` as U for checkpoint selection |
| `baseline_results.csv` | 3,300 greedy and 33,000 random rows | Preserve these results; evaluate DQN on the same instance IDs |

The reference's fractional U exceeds V* on 13,033 instances. This is expected.
Validation selects the highest source-equal mean V/U, with earlier checkpoints
winning exact ties. It does not optimize the test set. All returned item indices
refer to the original stored arrays, although decisions follow `item_order`.

The existing main-test mean exact gaps are approximately **1.16% for greedy** and
**43.09% for random**, from the saved result files. Random has ten trials per
instance. These are baseline results, not evidence that DQN improves on either.

## Environment

Use the project's existing virtual environment. Install its pinned dependencies:

```powershell
.\venv\Scripts\python.exe -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

The extra RL dependency is [PyTorch 2.8.0 CPU](https://pytorch.org/get-started/previous-versions/).
The recorded implementation uses one CPU thread and deterministic seeds by default.
Each run records actual Python/PyTorch/NumPy versions, device, thread count, source
code hashes and effective configuration. CUDA is optional via `--device cuda` with
an appropriate separately installed PyTorch build; CPU is the verification target.

## Verify inputs and tests

```powershell
.\venv\Scripts\python.exe -m experiment.rl.cli check-data
.\venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp results/pytest-rl-check
```

Use a fresh `--basetemp` path if needed. Pytest owns that temporary directory.
`check-data` verifies the saved corpus fingerprints and reference coverage; it does
not regenerate data or solve instances again. Training reads only the train and
validation JSONL partitions. No checkpoint receives reference optima as input.

## Pilot, then full training

```powershell
.\venv\Scripts\python.exe -m experiment.rl.cli train --pilot --seed 0 --out results/dqn-pilot-seed0
```

The pilot requests 50,000 transitions and finishes its current balanced block.
With the current eight source/size pairs, blocks contain 740 transitions, so the
completed run takes **50,320** transitions. Inspect finite losses, validation V/U,
source exposure and elapsed time. A pilot is not a completed final experiment.

After reviewing the pilot, use the fixed 300,000-transition budget for each seed:

```powershell
.\venv\Scripts\python.exe -m experiment.rl.cli train --seed 0 --out results/dqn-seed0
.\venv\Scripts\python.exe -m experiment.rl.cli train --seed 1 --out results/dqn-seed1
.\venv\Scripts\python.exe -m experiment.rl.cli train --seed 2 --out results/dqn-seed2
```

Defaults come from `experiment_config.yaml`. `--transitions N` labels the run as a
custom diagnostic. Output directories must not exist; repeated commands never
overwrite earlier runs. Checkpoints support inference; exact interrupted-training
resume is not implemented. A fresh run restarts from its seed.

## Run outputs

| File | Purpose |
| --- | --- |
| `run.json` | Actual transitions, updates, source/size exposure, unique instances, effective config, runtime, hashes and status |
| `training.csv` | Update step, Huber loss, epsilon and gradient norm before clipping |
| `validation.csv` | Step, equal-source score, per-source scores, new-best flag and validation timing |
| `validation_details.json` | Per-cell scores and validation exact gaps |
| `best.pt` | Earliest highest validation-score model |
| `last.pt` | Model at the completed training budget |

Validation occurs at step zero, at the first episode boundary crossing each
25,000-step interval, and at completion. The initial model can remain best if later
training degrades; the result must be reported honestly. Test inference is a separate
command, so training alone does not expose test scores.

## Evaluate a frozen full-run checkpoint

```powershell
.\venv\Scripts\python.exe -m experiment.rl.cli evaluate --checkpoint results/dqn-seed0/best.pt --out results/dqn-seed0-evaluation
```

Repeat for seeds 1 and 2. Each evaluation produces `rl_results.csv` with one row
per test/E3/E4 instance and `evaluation.json` with input and checkpoint provenance.
Raw columns match the baselines; extra columns record selected original indices,
checkpoint digest/step and run kind. Every selection is independently checked for
feasibility and integer value/weight totals. Per-instance latency includes encoding
and solution construction, excludes file I/O, and follows a warm-up call. Batched
validation timings are not used as inference latencies.

Pilot/custom checkpoints require `--allow-diagnostic` to evaluate and remain
labelled as diagnostic. Input/config mismatches stop evaluation.

## Remaining experiment work

Run all three full-budget seeds, evaluate their frozen best checkpoints, average
random trials within each instance, and compare methods on identical instance IDs.
Report exact gaps, feasibility, latency, training cost and robustness by source,
family, size and capacity. Preserve separate seed results and paired uncertainty
estimates. A full comparison report and figures follow those runs; implementation
tests or a pilot do not substitute for them.

Implementation design: [DQN design](docs/superpowers/specs/2026-10-03-dqn-design.md).
Algorithm reference: [official PyTorch DQN tutorial](https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html).
