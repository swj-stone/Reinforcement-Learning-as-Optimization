# Strategy Portfolio Implementation Plan

> **For agentic workers:** Use executing-plans or subagent-driven-development task by task. Track verification below.

**Goal:** Improve final solution value through a portfolio and an experience-based strategy selector, retaining the previous DQN baseline.

**Architecture:** Pure candidate solvers feed a contextual bandit with selected-action reward feedback. Separate training and held-out evaluation commands bind data, checkpoints and code to hashes.

**Tech Stack:** Existing Python venv, NumPy, PyTorch, pandas, pytest, Matplotlib.

**Spec:** [Design](../specs/2026-10-03-portfolio-design.md).

## Global constraints

- Value is the primary objective; no test-driven tuning.
- Existing experiment/ code, original inputs, checkpoints and report remain unchanged.
- Keep 50:50 source balance; fresh instances are not fresh real-world business records.
- All candidates obey integer capacity; DP is evaluation-only.

## Review focus

- A candidate must never duplicate an item or exceed capacity: solver property tests.
- Unseen context and zero-capacity cases must have defined predictions: bandit tests.
- No metadata or counterfactual rewards may enter a chosen-action update: bandit tests.
- Fresh holdout must not overwrite or duplicate the old corpus: holdout tests.
- Cached training and timed deployment must agree, and every reported result reconstructs: pipeline checks.

## Task 1: Candidate solvers

Files: portfolio/strategies.py; tests/test_portfolio_strategies.py.
Interface: candidate_names(), solve_strategy(name, instance), choose_best(solutions), returning BaselineSolution.

- [x] Write/run failing analytical and randomized feasibility tests.
- [x] Implement the seven deterministic strategies in the spec.
- [x] Run targeted tests and inspect counterexamples.

## Task 2: Contextual learner

Files: portfolio/bandit.py; tests/test_portfolio_bandit.py.
Interface: context_key(instance, bins); Bandit.predict/update/to_dict/from_dict; fit(rewards, contexts, sources, validation_rewards, validation_contexts, config, seed).

- [x] Test context invariance, unseen fallback, selected-action-only updates, complementary-context learning, serialization and deterministic balanced training.
- [x] Implement tabular estimates with global shrinkage and epsilon exploration.
- [x] Run targeted tests.

## Task 3: Fresh held-out corpus

Files: portfolio/holdout.py; tests/test_portfolio_holdout.py.
Interface: build_holdout(out: Path, repo: Path), refusing existing outputs.

- [x] Implement and test namespaced generation with global canonical deduplication.
- [x] Record old/new date and source-row overlap explicitly.
- [x] Freeze config and generate only after the design is saved.

## Task 4: Reproducible experiment and report

Files: portfolio/run.py, portfolio/report.py, PORTFOLIO_REPORT.md, README.md; new results and published tables.

- [x] Build train/validation reward cache, fit three seeds and freeze selected models.
- [x] Evaluate fresh holdout with exact references and all fixed/learned/portfolio controls.
- [x] Reconstruct every solution; compare frozen original hashes; run complete pytest suite.
- [x] Independently review implementation/results, publish English report and concise Chinese findings.

Validation: 559 tests passed in 180.11 s. Independent audit reconstructed 92,400 outcomes and 24 bootstrap intervals. Added a runnable single-instance CLI in portfolio/solve.py, with seven focused tests, to expose best/learned/guarded modes. The value-priority default runs all eight candidates. See PORTFOLIO_REPORT.md for results and attribution.
