# DQN implementation and pilot report

**Date:** 2026-10-03  
**Scope:** Implementation verification and a seed-0 pilot. Full three-seed test-set
comparisons and the final experimental report are not yet complete.

## Alignment with previous results

All five processed corpus files match `data_manifest.json`. The 17,700 certified
reference rows and 36,300 baseline rows are preserved, with no recorded baseline
feasibility violations. The baseline table covers 3,300 evaluation instances:
one greedy solution and ten random trials per instance.

The reference table distinguishes certified optimum `value` (V*) from fractional
`upper_bound` (U). The implementation retains V/U for validation checkpoint
selection and uses V* for exact optimality gaps. README/protocol/configuration
wording was aligned with that existing reference-service decision. The frozen
data generation algorithm and the existing result files were not regenerated.

## Verification

- Full project suite: **449 passed, 0 failed, 0 skipped**, in 161.01 seconds.
- RL-specific tests: **47 passed**, including real optimizer updates, deterministic
  repeated runs, checkpoint reload, CLI evaluation on fixtures, action masking,
  terminal targets, original-index mapping, padding, source balance and n=500 inputs.
- Dependency check: no broken requirements in the existing project `venv`.
- An old path test was corrected to support pytest temporary directories both
  inside and outside the repository; production path serialization was unchanged.
- The trained best checkpoint was independently reloaded and used to reconstruct
  all 2,400 validation solutions. Feasibility, integer totals, input/code hashes and
  the exact saved validation score were reproduced.

## Pilot settings and exposure

The pilot uses the existing YAML settings, seed 0, Python 3.12.10,
PyTorch 2.8.0+cpu, Windows 11 and one PyTorch CPU thread. It trains on the fixed
12,000-instance training corpus and selects checkpoints on all 2,400 validation
instances. No actual test/E3/E4 DQN scores were calculated.

| Quantity | Measured value |
| --- | --- |
| Requested transitions | 50,000 |
| Actual transitions, completing the balanced block | 50,320 |
| Optimizer updates | 11,331 |
| Episodes | 544 |
| Unique training instances seen | 530 |
| Episodes per source | 272 synthetic / 272 real |
| Transitions per source | 25,160 synthetic / 25,160 real |
| Episodes per source/size pair | 68 |
| Wall time | 164.61 seconds |

Every logged loss and gradient norm was finite. Source balance was checked against
the saved run metadata. Wall time includes validation and was measured while the
regression suite ran concurrently; it is diagnostic timing, not an isolated final
performance benchmark.

## Validation learning progress

The score below is the equal-source mean V/U. U is a fractional upper bound, so
this must not be described as the fraction of the certified optimum attained.

| Environment transitions | Synthetic V/U | Real V/U | Equal-source V/U | Selected best? |
| --- | --- | --- | --- | --- |
| 0 | 0.093362 | 0.033988 | 0.063675 | Initial checkpoint |
| 25,010 | 0.741992 | 0.761401 | 0.751697 | Superseded |
| 50,000 | 0.795430 | 0.824214 | **0.809822** | **Yes** |
| 50,320 | 0.765645 | 0.750044 | 0.757845 | No |

At the selected checkpoint, validation exact optimality gaps are **19.4258%** for
synthetic data and **16.6080%** for real data, giving an equal-source mean of
**18.0169%**. These are validation results and are not directly comparable with
the previously reported main-test baseline gaps.

The pilot demonstrates a working learning pipeline and substantial improvement
over initialization. It does **not** establish convergence or superiority over
greedy. The final score fell after the best checkpoint, and some cells remain
weak: synthetic correlated instances with n=20 and alpha=0.25 average V/U=0.193480.
This supports retaining validation-based checkpoint selection and reporting
per-cell results in the full experiment.

## Local evidence

Generated run artifacts live in the git-ignored results directory:

- [Run metadata and exposure](results/dqn-pilot-seed0/run.json)
- [Validation history](results/dqn-pilot-seed0/validation.csv)
- [Training update log](results/dqn-pilot-seed0/training.csv)
- [Per-cell validation scores](results/dqn-pilot-seed0/validation_details.json)
- [Independent verification](results/dqn-pilot-seed0/verification.json)
- [Best model checkpoint](results/dqn-pilot-seed0/best.pt)
- [Existing-artifact audit](results/rl-alignment-2026-10-03.json)
- [Exact environment package snapshot](results/rl-environment-freeze.txt)

Best-checkpoint SHA-256:
`c60d55e11446bb3af4c71e9ac9270bf18bc0b94efa295f7b3222f32625155428`.

## Next experiment stage

Use the configured 300,000-transition budget for each of seeds 0, 1 and 2,
finishing each current balanced block. Freeze the budget and settings before
viewing test performance. Evaluate each seed's validation-selected checkpoint on
the same 3,300 instances as the baselines, then produce paired comparisons,
per-cell robustness results, isolated timing measurements and the full report.
Keep a failed-to-outperform-greedy result if that is what the measurements show.

The commands and artifact formats are in [RL_IMPLEMENTATION.md](RL_IMPLEMENTATION.md).
