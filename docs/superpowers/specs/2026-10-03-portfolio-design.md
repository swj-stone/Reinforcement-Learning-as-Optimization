# Experience-based strategy selection

The user authorizes improving the current system while preserving its DQN as a baseline. Final solution value has priority; runtime is reported separately.

## Scope and claims

Keep the static, fully observed 0/1 knapsack task. Build a contextual bandit: observe instance statistics, select one complete solver, receive its normalized final value, and update the selected action's empirical reward. This is one-step reinforcement learning for algorithm selection, not a stronger sequential DQN and not an online arrival experiment. Expected best selection cannot guarantee the best strategy on every instance.

Eight actions: density greedy, value first, light first, value/sqrt(weight), an adapted 1/e observation-threshold rule, eight-seed greedy search, eight-candidate exchange repair, and the frozen seed-0 DQN (chosen by the original validation scores). The threshold rule observes and rejects floor(n/e) stored-order items, then admits fitting items at least as dense as the observed maximum. Its classical secretary guarantee does not apply to this knapsack adaptation.

Context uses only n, capacity/sum(weights), density coefficient of variation, and max(weight)/capacity. Fixed bins are in portfolio_config.json. Source, family, identifiers, test references and candidate outcomes are excluded from context. Q estimates use observed per-action means with five global-mean pseudo-observations. Unseen contexts fall back to learned global action means.

Train three seeds for ten balanced epochs (120,000 selected-action observations each) on the existing 12,000 training instances. Epsilon decays linearly from 1 to 0.1. Each epoch alternates shuffled synthetic and real instances. The reward is V/U with U computed from the known input by fractional knapsack; exact DP labels are unnecessary for training. A cached simulator may precompute all candidate outcomes, but each learner update receives only the selected action's reward. Select checkpoints using source-balanced mean V/U on the existing validation set; keep the earlier checkpoint on ties. Choose the best fixed strategy on validation as an essential control.

## Evaluation

Freeze the design/config before creating fresh held-out instances or inspecting their solver outcomes. Generate 2,400 main, 600 n=500, and 300 extreme-tail cases, canonical-disjoint from all 17,700 old instances. Real cases reuse the original test-date pool; this is a follow-up within the same business cohort, not an independent temporal validation. No training/validation source rows or dates may overlap. Keep all old artifacts unchanged.

Compare random (mean of ten trials), greedy, all three original DQN seeds, each candidate, validation-best fixed strategy, each learned selector, greedy-guarded selectors, and the best result from running the entire portfolio. The latter provides the best value within these eight candidates, not a globally optimal answer. Charge the cost of all candidates; do not count an oracle selection as learned performance. Guarded selectors explicitly pay for greedy and the selected solver and derive their no-regression guarantee from that fallback.

Report exact DP gap, normalized quality, paired improvements, regret to portfolio best, chosen-action frequencies, feasibility, and single-instance timing. Bootstrap paired quality differences by synthetic strata and real source-date clusters. Report three training seeds separately and their descriptive mean; intervals are conditional on fitted models. A selector losing to the best fixed action is a valid result and must be stated. No test-driven retuning.

## Isolation

New code lives in portfolio/, outputs in new results/portfolio-v2* paths, compact published tables in portfolio_report_tables/, and the English report in PORTFOLIO_REPORT.md. Use the existing venv and installed dependencies. Preserve experiment/ code, experiment_config.yaml, original corpora, references, checkpoints, and EXPERIMENT_REPORT.md. No commits or deployment.
