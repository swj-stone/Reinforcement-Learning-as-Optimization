# Experimental Plan: Reinforcement Learning as Optimization

**Course:** Artificial Intelligence in Operations Research — Seminar  
**Author:** Wenjie SHI  
**Supervisor:** Chi ZHAO  
**Status:** Protocol for implementation; no experimental results are claimed.  
**Schedule:** Two weeks, with optional extensions after the core experiment.

## 1. Objective and Fixed Scope

Compare exactly three methods: **random selection, greedy selection, and reinforcement learning**. Test how their performance changes with data source, distribution, instance size, and severe imbalance.

Use a single-resource 0/1 knapsack model:

`maximize sum(v_i * x_i), subject to sum(w_i * x_i) <= C, x_i in {0, 1}`

Here, `w_i` is a positive resource requirement, `v_i` is a positive benefit, and `C` is the capacity. Resource units may be handling units or volume, depending on the source. Each instance uses one consistent unit system. Routing, multiple simultaneous constraints, online arrivals, and uncertain cargo realization are outside the core experiment.

The required source mixture is **50% synthetic and 50% real-data-derived instances in both training and validation**. The main test set will use the same balance. Public business records are available; private warehouse or airline data is not currently available.

## 2. Synthetic Data: Explicit Generation Recipes

For every family, generate `n in {20, 50, 100, 200}` and `alpha in {0.25, 0.50, 0.75}`. Set `C = floor(alpha * sum(w_i))`. Use integer requirements and values, and store the generator seed and family with every instance.

`U(a,b)` below means an inclusive discrete uniform draw. Independent draws are used unless a relationship is explicitly specified.

| Family | Generation rule | What it tests |
| --- | --- | --- |
| Uniform | Independently draw `w_i ~ U(1,100)` and `v_i ~ U(1,100)` | A simple reference distribution |
| Approximately normal | Independently draw weight and value from `Normal(50,15^2)`, round to the nearest integer, and clip to `[1,100]` | Concentration near typical values |
| Correlated | Draw `w_i ~ U(1,100)` and `e_i ~ U(-10,10)`; set `v_i = max(1,w_i+e_i)` | Similar value-to-resource ratios |
| Severely skewed | Start with `w_i ~ U(1,20)` and density `d_i ~ U(1,5)`. Select `k=max(1,floor(0.05*n+0.5))` indices uniformly without replacement and replace their weights with `U(200,500)`. Independently select another `k` indices and replace their densities with `U(30,60)`. Set `v_i=w_i*d_i`. | Rare heavy items, rare high-value-density items, and their possible overlap |

The two rare-index sets in the skewed family are sampled independently and may overlap. Each set contains approximately 5% of items, subject to integer rounding. This is **item-distribution imbalance**, not a classification-label imbalance. Do not remove the extreme observations after generation. Record actual top-5% weight/value shares, maximum-to-median ratios, and coefficient of variation to describe the generated skew rather than assuming a particular concentration.

Use separate generator streams for train/validation/test: master seeds `1001/2001/3001`. Derive per-instance seeds from the split, family, size, capacity ratio, and instance index. Reject duplicate canonical instance hashes across splits, including copies with reordered items.

## 3. Public Real Data and the Optimization Mapping

### 3.1 Core source: UCI Online Retail

Obtain the original workbook from the official [UCI Online Retail page](https://archive.ics.uci.edu/dataset/352/online+retail). Relevant source fields are `InvoiceNo`, `StockCode`, `Quantity`, `InvoiceDate`, and `UnitPrice`. Save its download date and SHA-256 checksum.

The following is our proposed adaptation, not a claim about the retailer's actual optimization process:

| Model component | Definition |
| --- | --- |
| One selectable item | One positive transaction line, treated as an indivisible fulfillment lot |
| Resource `w_i` | `Quantity`, interpreted as handling-unit demand |
| Benefit `v_i` | `Quantity * UnitPrice`, converted to integer pence using decimal arithmetic and round-half-up |
| Capacity `C` | `floor(alpha * sum(w_i))`, an experimentally imposed handling-unit budget |
| Business interpretation | Select fulfillment lots to maximize sales value under a handling budget |

A lot is accepted or rejected in full; different lines from the same invoice may be selected separately. Keeping invoices intact during **data splitting** does not impose an all-or-nothing invoice constraint in the optimization model. Sales value is not profit, and quantity is not measured cargo weight. Describe this source as **real-transaction-derived warehouse allocation**, not as a recorded warehouse loading benchmark.

Cleaning rules:

1. Preserve the raw workbook and assign each row a stable source-row ID.
2. Remove cancellation invoices (case-insensitive prefix `C`), nonpositive quantities/prices, missing required fields, and invalid timestamps. Require parsed quantities to be positive integers; reject fractional or nonnumeric quantities instead of silently rounding them. Log counts by reason.
3. Restrict the core experiment to product codes consisting of five digits optionally followed by letters. This excludes non-product codes such as postage and manual adjustments. Audit excluded codes/descriptions on training records, publish the rule and exclusion counts, and freeze any correction before generating the final validation/test corpora.
4. Flag exact duplicate rows; remove only exact full-record duplicates under a documented rule. Retain legitimate repeated products and distinct order lines. Reject lots whose converted integer benefit is zero.
5. Preserve positive outliers. Do not trim difficult examples using test-set quantiles.
6. Exclude customer identifiers and descriptive text from model features; retain only necessary provenance fields for grouping and auditing.

### 3.2 Additional domains

| Domain | Public source and use | Inclusion condition |
| --- | --- | --- |
| Parcel dispatch | [Amazon Last Mile Routing Challenge](https://registry.opendata.aws/amazon-last-mile-challenges/); its [official schema](https://github.com/MIT-CAVE/rc-cli/blob/main/templates/data_structures.md) includes package dimensions and planned service time | Optional separate dispatch experiment: group packages by stop, use total package volume as resource and package count as benefit, and impose a volume budget. Verify group sizes and split by whole route. This is an adaptation, not a revenue or routing benchmark. |
| Air cargo | [Dubai Customs public airway-bill catalog](https://www.gslb.dubaipulse.gov.ae/data/dc-records/customs_airway_bill_detail-open?organisation=dubai-customs&page=24&service=dc-records) is a candidate for a field audit | Verify access, row meaning, units, date/manifest grouping, and an objective field before use. A quantity objective may be possible if documented; monetary revenue and flight grouping have not been verified. |

For a revenue-based air-cargo experiment, require shipment-level weight or volume, shipment revenue/value, a timestamp, and a flight/manifest identifier. Aggregate airport tonnage is not a substitute. If revenue is simulated, label the experiment as semi-synthetic and keep it outside the core real-data quota. The core plan proceeds with UCI alone and does not depend on obtaining airline records.

Any additional source must be registered before rebuilding and freezing splits. If two real sources are included, allocate the real half equally between them where supported: for example, 50% synthetic + 25% warehouse + 25% parcel dispatch. Do not mix incompatible units inside one instance.

## 4. Splits, Instance Construction, and the 50:50 Rule

### Split raw records first

Assign each invoice to the calendar date of its earliest transaction. Sort the distinct invoice dates; allocate the earliest 70% of dates to training, the next 15% to validation, and the remaining dates to test. Use floor rounding for the first two boundaries. All rows of one invoice inherit that assignment, even if its timestamps cross midnight. Record actual row and invoice percentages, which need not equal the date percentages.

For each partition, build a candidate pool for each assigned source date. To construct a real instance, choose an eligible date uniformly and sample `n` distinct transaction lines without replacement from that day's pool. Repeat with independently seeded samples to meet the quotas. Records may recur in different instances **within the same split**; related instances are not independent business observations. No source row or invoice may cross partitions. Store all parent IDs and canonical instance hashes. Reject duplicate model-input hashes across splits, even if different source rows happen to produce identical item pairs and capacity.

For both data sources, derive and store one random item permutation per instance from its seed. Random selection and RL use this same order. Greedy explicitly sorts it according to its own decision rule.

This tests generalization to later business dates, with potential temporal distribution shift. It is not a claim of generalization to unseen products or customers.

### Freeze these target counts

| Split | Synthetic instances | Real-derived instances | Total |
| --- | --- | --- | --- |
| Train | 6,000 | 6,000 | 12,000 |
| Validation | 1,200 | 1,200 | 2,400 |
| Main test | 1,200 | 1,200 | 2,400 |

Synthetic quotas cover `4 families * 4 sizes * 3 capacity ratios = 48` cells: **125 training and 25 validation/test instances per cell**. Real quotas cover `4 sizes * 3 ratios = 12` cells: **500 training and 100 validation/test instances per cell**. Thus every size/capacity cell is 50:50 by source after summing synthetic families.

Before generating the final corpus, audit the number of eligible dates and unique instances in every split/cell. A date with fewer than `n` valid lots is ineligible; do not duplicate lines inside an instance to fill it. If a size/capacity cell cannot meet its real-data quota, set a smaller validation/test quota `q` per synthetic family for that cell, use `4q` real instances, and use five times those counts in training. Choose `q <= 25` that all three partitions support; if `q=0`, mark the cell unavailable. Document deviations and report both count-weighted and cell-macro results. Never replace missing real data with generated records under a real-data label.

The 50:50 rule means **half of the complete knapsack instances**, not half of individual items or bytes. Example: a 100-instance batch contains 50 synthetic knapsacks and 50 real-derived knapsacks.

## 5. Fair Definitions of the Three Methods

| Method | Exact protocol |
| --- | --- |
| Random | Hold the stored item permutation fixed. If an item fits, include it with probability 0.5; otherwise skip. Run 10 trials with independent inclusion coins; derive each coin-flip seed deterministically from evaluation seed `6001`, instance ID, and trial index. Average outcomes rather than selecting the best trial. |
| Greedy | Sort by descending `v_i/w_i`; break ties by lower `w_i`, then stable item ID. Include every item that fits and continue after a rejected item. |
| RL | Train one DQN architecture on the mixed training corpus. At test time choose the feasible action with the highest Q-value with exploration disabled; break exact ties by skipping. |

All methods receive the same complete candidate information and capacity. Each has one construction pass per reported solution; random repetition estimates its variability and is not extra best-of-ten search. Timing includes sorting, feature construction, action selection, and feasibility checks, but excludes loading the shared corpus from disk. Report one-trial random latency, not the duration of all ten trials as a single solution.

## 6. RL Environment and Initial Training Configuration

An episode considers each item once. The raw state contains all item requirements/values, the current index, and remaining capacity. An include action subtracts the resource requirement and earns the item's value; skip earns zero. After the last item, the episode terminates and the bootstrap term is zero. Skip is always valid; infeasible include actions are masked in exploration, inference, and the maximum used in the learning target.

Use `gamma=1`. For learning only, set reward to `v_i / sum_j(v_j)` when including an item. Evaluate final selections in the original units. This keeps per-instance rewards comparable while preserving the maximizing solution for each instance.

To support different sizes, encode each remaining item's normalized pair `(w_i/C, v_i/sum(v))` with a shared two-layer, 64-unit ReLU encoder. Concatenate its mean and max pooled embeddings with the current item's normalized pair, `remaining_capacity/C`, `remaining_count/n`, and `log(1+n)`. Feed this vector to a 128-unit ReLU layer with two Q-value outputs. Pool only unprocessed items, including the current item; exclude all processed items and padding. Terminal states bypass next-state evaluation. IDs and source labels are not prediction features.

| Setting | Initial value |
| --- | --- |
| Training seeds | `0, 1, 2` on the same frozen data splits |
| Optimizer / learning rate | Adam / `0.0003` |
| Loss / gradient clipping | Huber loss / global norm `10` |
| Replay capacity / batch size | `100,000` transitions / `64` |
| Replay source balance | `32` synthetic + `32` real transitions per update, after both pools contain enough samples |
| Warm-up / update frequency | `5,000` transitions / one update every `4` transitions |
| Target network | Copy online weights every `2,000` environment transitions |
| Exploration | Epsilon decreases linearly from `1.0` to `0.05` during the first `150,000` transitions, then stays at `0.05` |
| Initial run budget | `300,000` transitions per seed; complete the current balanced episode block |
| Validation | Every `25,000` transitions at an episode boundary; use no exploration |

Sample repeatedly from the fixed training corpus; 12,000 distinct available instances do not mean 12,000 episodes will be completed. A balanced block samples one instance for each available `(source, size)` pair, in random order, with supported capacity ratios and synthetic families cycled evenly across blocks. Exclude cells marked unavailable in the manifest. This preserves 50:50 episode exposure and balances transition counts by source. Report actual episodes, unique instances seen, and transitions per source/size. Replay batches additionally enforce 50:50 source exposure.

Compute and cache validation references before the pilot. Select each seed's checkpoint using the mean validation score `V/U`, where `U` is a fixed valid reference upper bound for that instance. For a certified optimum, `U=V*`; exclude trivial zero-optimum cases from this ratio and count them separately. Weight sources equally and report cell-macro scores if quotas changed. Resolve ties using the earlier checkpoint. Never select checkpoints using test performance.

Run a 50,000-transition pilot first to check stability and estimate compute time. Fix the final budget before viewing test metrics. If 300,000 transitions is insufficient and time allows, a validation-only decision may raise it to 1,000,000 for **all three seeds**; record the change. Report a failed-to-learn result honestly if the budget is exhausted.

## 7. Reference Solutions and Metrics

Use integer dynamic programming for small cases with manageable capacities, or [OR-Tools CP-SAT](https://developers.google.com/optimization/cp/cp_solver) with binary item variables for the general case. Limit each CP-SAT reference solve to 10 seconds, one worker, and a fixed seed. Cache its status, best feasible value `L`, upper bound `U`, runtime, and item-set hash. This is an evaluation service, not a fourth competing method.

If the solver has not certified optimality, retain a valid upper bound; the fractional-knapsack relaxation provides a fallback. Never treat a timed-out incumbent as the optimum. Convert data to the declared integer units before solving; do not discretize it differently for the competitors and the reference.

| Metric | Reporting rule |
| --- | --- |
| Feasibility | Selected resource must not exceed capacity; each item is selected at most once. Target: zero violations. |
| Exact optimality gap | For certified `V*>0`: `100*(V*-V)/V*`. Also report the certification rate. |
| Bound-based gap | When only `U>0` is known: `100*(U-V)/U`, explicitly labeled an upper bound on the true percentage gap. Report solver status alongside it. |
| Relative improvement | `100*(V_RL-V_greedy)/V_greedy` when the greedy value is positive; show the win/tie/loss rate as well. |
| Speed | Median and 95th-percentile single-instance latency on the same hardware; include method preprocessing. |
| Learning cost | Training wall time, transitions, hardware, and checkpoint-selection procedure |
| Robustness | Mean/median quality and variation by source, family, size, capacity ratio, and tail subset |

If no item fits, all methods must return the empty solution: mark the case trivial, verify feasibility, and exclude it from percentage denominators while reporting its frequency. Do not discard infeasible outputs from summary tables; count them as failures.

Average normalized per-instance quality, not raw monetary values mixed with synthetic scores. First average random trials within an instance and summarize RL seeds separately. The primary aggregate is one-half of the synthetic instance mean plus one-half of the real instance mean. For uncertainty, use 1,000 paired bootstrap resamples: resample whole source dates with all their derived real instances, and whole synthetic instances within cells, keeping method outputs paired. Recompute the two source means separately and combine them with equal weights in every resample. This accounts for within-date dependence without treating repeated source records as independent business observations. Report training-seed variation separately from data-sampling uncertainty.

## 8. Experiment Matrix

| ID | Data and run | Required output |
| --- | --- | --- |
| E0: correctness | Three-item example plus 100 generated instances with `n<=15`; enumerate all subsets for the reference | Verified optimum, reward accounting, masks, and zero constraint violations |
| E1: main comparison | Three methods on the frozen realized main-test corpus (target: 2,400 instances); three RL seeds and 10 random trials per instance | Balanced overall table plus separate synthetic/real tables; report actual manifest counts |
| E2: distribution and scale | Break down E1 by its four families, four sizes, and three capacity ratios | Gap-versus-size plots and distribution/imbalance comparison; no new training |
| E3: unseen size | Separate `n=500` test: 300 synthetic + 300 real instances; synthetic quota `4 families * 3 ratios * 25`, real quota `3 ratios * 100` | Generalization to larger instances, retaining 50:50 source balance |
| E4: stronger imbalance | 300 new synthetic instances at `n=200`: 100 per capacity ratio. Change the skewed generator to `k=2` rare weight indices with `U(500,2000)` and independently `k=2` rare density indices with `U(100,200)` | A synthetic-only 1% rare-item stress result, reported separately from mixed results |
| E4b: real tail slice | From the main real test set, flag instances containing a lot above the training-source 99th percentile of quantity or value density | Real-outlier performance; this is a subset of E1, not an independent dataset |
| E5: optional mixture ablation | Train on 6,000 distinct instances per condition: 6,000 synthetic, 6,000 real, or 3,000 of each. Match transitions and seeds; retrain all three conditions because the mixed subset differs from E1. | Evidence about source mixing with matched corpus size and compute; a separate training study |

E3 real instances must use eligible test-source dates and remain disjoint in canonical instance hash from the main set. They may reuse test source records; report this dependence. If `n=500` real quotas cannot be met, reduce the matching synthetic counts to preserve 50:50 or report the real arm unavailable. Never describe a synthetic-only test as mixed-data generalization.

Reserve stress-test streams with seeds `4001` (size) and `5001` (tail). They must not enter training, checkpoint selection, normalization fitting, or hyperparameter tuning. E4b thresholds are fixed using only training records. Test-driven changes require a newly held-out test set and a separate reported experiment.

For E5, draw balanced subsets only from training partitions and freeze them before training. If quotas were reduced, match all three conditions to the same smaller corpus size. Use the same mixed validation set and test sets across conditions. Single-source ablations use 64 transitions from that source per replay batch; the mixed condition retains 32 + 32. Record remaining differences in source-date diversity when interpreting the result.

## 9. Two-Week Execution Checklist

These are implementation tasks and acceptance conditions, not commands for software that already exists.

| Day | Action | Output and acceptance check |
| --- | --- | --- |
| 1 | Acquire UCI data; inspect fields, units, cancellations, and extreme values | Data-source note, raw-file checksum, cleaning report, and confirmed optimization mapping |
| 2 | Split source dates/invoices and audit eligible pools for every size | Split manifest; zero row/invoice overlap; recorded quota feasibility |
| 3 | Implement four generators and construct frozen mixed corpora | Target count tables, 50:50 checks by split and size/capacity cell, canonical-hash uniqueness |
| 4 | Implement random/greedy methods and the reference solver | E0 optimum checks pass; ties and random seeds are reproducible |
| 5 | Implement the environment, variable-size network, masking, and replay balance | Complete episodes at every size; raw/normalized rewards agree; no invalid action selected |
| 6 | Cache validation references; run the pilot; inspect validation learning curves and runtime | Decide and freeze budget/configuration without using test results |
| 7–8 | Train all three seeds and select validation checkpoints | Three saved models, logs, source-exposure counts, and validation-selection records |
| 9 | Run E1–E4b and compute cached reference values/bounds | Per-instance metrics with solver status; no cross-split leakage; feasibility report |
| 10 | Produce tables, figures, limitations, and seminar slides | Conclusions address all three methods and both sources, including unsuccessful RL outcomes |

Prioritize E0–E2 if compute time is short; list any unfinished stress tests explicitly. Optional additional datasets and E5 must not delay delivery of the core comparison.

## 10. Artifacts and Completion Criteria

The implementation should produce the following files or equivalent clearly named artifacts:

| Artifact | Required contents |
| --- | --- |
| `data_manifest.json` | Source links/checksums, cleaning rules, dates, seeds, counts, units, source/group IDs, and split/hash audit |
| `experiment_config.yaml` | Generator formulas, quotas, RL settings, budgets, evaluation rules, and pinned package versions |
| `instances/{train,validation,test}.jsonl` | Instance ID, source, family, `n`, capacity, resource/value arrays, item order, and source-row/group provenance |
| `checkpoints/` and training logs | Three selected RL models, step counts, validation history, and hardware/runtime metadata |
| `reference_results.csv` | Instance ID, certified status, optimum or bounds, and solver time |
| `metrics.csv` | Instance ID, method, seed/trial, selected value/resource, violations, normalized quality, and latency |
| `figures/` | Learning curves, source-specific quality, gap versus size, skew comparison, and quality versus runtime |
| Final report/slides | Method comparison, dataset adaptations, uncertainty, limitations, and reproducibility instructions |

Completion means the data audit passes, all three methods run on the same frozen core test set, the 50:50 requirement is verified, and conclusions are supported by saved measurements. No particular method is required to win. Dataset adaptations and unavailable air-cargo fields must remain explicit in the final report.
