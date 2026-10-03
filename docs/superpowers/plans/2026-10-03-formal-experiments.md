# Formal experiments and report

Authority: `EXPERIMENT_PLAN.md`, especially Sections 5–10; user instruction to continue after the verified pilot.

## Fixed decisions before test evaluation

- Keep the frozen corpus, exact references, original baselines and DQN implementation unchanged.
- Run seeds 0, 1 and 2 for the configured 300,000 transitions, completing the final balanced block. No budget extension or test-driven tuning in this experiment.
- Select each checkpoint independently by equal-source validation V/U; retain all three seeds in the report.
- Run training concurrently, one CPU thread per process. Report concurrent wall times explicitly; measure single-instance inference sequentially after training finishes.
- Preserve the original baseline CSV. Recompute baselines to a new file for current-machine timing and verify every non-timing result against the original.
- Put analysis code outside `experiment/` so training-code fingerprints remain identical across seeds while analysis is developed.
- Use 1,000 paired bootstrap resamples: real source-date clusters, synthetic instances within cells, equal weighting of the two source means. Separate data uncertainty from training-seed variation.
- Treat E4b as a subset of E1; E3 shares real source records with E1. Optional E5 and new data sources remain outside this completed core study.

## Tasks and verification ledger

- [x] Freeze input/code hashes and launch the three formal runs using the existing virtual environment.
- [x] Implement and test strict result joins, per-instance random averaging, source-balanced summaries and paired uncertainty.
- [x] Verify all three runs completed, checkpoint provenance and validation selection; then evaluate E1–E4b.
- [x] Verify new baseline logical results match the frozen results and all RL selections are feasible.
- [x] Produce reproducible figures, machine-readable tables and an English experimental report, with limitations and negative results retained.
- [x] Review the analysis independently, run relevant tests and update README / experiment status.

Working-tree ruling: continue in the user's existing checkout containing the preceding uncommitted implementation. A new checkout would omit the implementation and local data being continued. No commits or publishing are requested.

Shared interfaces: training/evaluation use unchanged corpus and config hashes; analysis joins by instance ID and distinguishes certified `value` from fractional `upper_bound`; figures consume saved analysis summaries and validation CSVs.

Verified so far: all three runs completed 300,440 transitions, 3,248 episodes and 73,861 updates each. Best checkpoints (seeds 0/1/2) are at steps 250,100 / 300,020 / 150,020; their validation scores were reproduced with the original batch boundaries. Fresh baseline timing preserved every non-timing field across all 36,300 rows. The consolidated suite passed 473 tests in 179.11 seconds. Timing evaluations run sequentially after training and testing have finished.

Review rulings: retain legitimate step-zero selected checkpoints in analysis; suppress mixed-unit raw-value summaries; preserve original-price E4b threshold semantics; use the same validation batch boundaries when reproducing training scores. All are implemented in analysis only, leaving the frozen training code unchanged.

Final outcome: 9,900 DQN results on 3,300 distinct held-out/stress instances, zero violations. Main exact gaps: random 43.089%, greedy 1.159%, DQN three-seed mean 9.521% (seed SD 1.213 percentage points). E3 DQN gap 31.641% reveals poor unseen-size generalization. Five PNG/SVG figure pairs were rendered and visually checked; compact tables and audits are saved under `report_tables/`. The independent reviewer reproduced all 888 summary rows, 8,807 numerical comparisons and 16 paired intervals, maximum numerical difference 4.26e-14, and checked the completed report. No substantive unresolved finding. Core E0–E4b study complete; optional E5, extra domains and slides are not claimed as delivered.
