# Classroom Report Notebook Plan

**Goal:** Deliver an extensive, executed English report.ipynb that explains the full course project, with short runnable demonstrations and readable figures.

**Scope:** The research subject is RL-based selection of optimization strategies. Random, greedy and the frozen DQN are primary comparisons. Best fixed, guarded selection and full-portfolio search are supplementary controls. Preserve all original experiments and acknowledge their limitations.

**Design:** A narrative notebook organized into background, mathematical model, data, baselines, sequential DQN, contextual bandit, experimental protocol, results, robustness, examples, limitations, reproducibility and presentation Q&A. Small code cells read local artifacts or run deterministic teaching examples. No network downloads, full training or overwriting experiment outputs on Run All. Save executed outputs and an HTML companion for presentation.

**Files:** report.ipynb; report.html; analysis/build_report_notebook.py; analysis/notebook_visuals.py; analysis/execute_report_notebook.py; assets/notebook/ small auditable demonstration bundle; requirements-notebook.txt. Preserve author/supervisor names already in project docs.

**Verification:** Execute every code cell in a clean kernel from the existing project venv; validate nbformat, inspect emitted figures and HTML, check result assertions and provenance. The run must work without access to full Git-ignored datasets/checkpoints by using a small hashed teaching bundle and published result tables. Full reproduction commands are documentation only.

- [x] Audit sources and prepare compact teaching artifacts with source hashes.
- [x] Write extensive English narrative and short demonstrations, with clear measured-versus-illustrative labels.
- [x] Create presentation figures and a guided live-demo route.
- [x] Execute all cells, export HTML, inspect content and plots, and independently review claims.
- [x] Deliver notebook, HTML and concise instructions for the existing venv.

Verification evidence: `results/notebook-verification/execution.json` records 30 successful code cells and 13 embedded figures; `content-review.json` records resolved review findings; `portability.json` records a second successful run from a 33-file minimal copy with original-project reads blocked except for the existing venv. Browser inspection confirmed that all 13 HTML images loaded and formulas rendered without MathJax errors. The frozen experiment provenance check passed. Notebook and HTML links and kernel instructions are included in README.md.
