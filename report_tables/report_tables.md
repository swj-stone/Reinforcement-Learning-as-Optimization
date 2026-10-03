### Training and selection

| Seed | Transitions | Episodes | Unique instances | Best step | Validation V/U | Concurrent wall time (s) |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 300440 | 3248 | 2846 | 250100 | 0.907155 | 1231.3 |
| 1 | 300440 | 3248 | 2829 | 300020 | 0.901160 | 1230.8 |
| 2 | 300440 | 3248 | 2830 | 150020 | 0.887261 | 1229.9 |

### E1: main results

| Method | Mean gap (%) | Median gap (%) | Mean V/U | Feasible |
| --- | --- | --- | --- | --- |
| Random | 43.089 | 44.269 | 0.561039 | 100.0% |
| Greedy | 1.159 | 0.117 | 0.973414 | 100.0% |
| DQN seed 0 | 8.505 | 6.611 | 0.902165 | 100.0% |
| DQN seed 1 | 9.194 | 7.358 | 0.895105 | 100.0% |
| DQN seed 2 | 10.864 | 8.103 | 0.880748 | 100.0% |
| DQN three-seed mean | 9.521 | 7.656 | 0.892673 | 100.0% |

### E1: source-specific exact gaps

| Method | Synthetic gap (%) | Real-derived gap (%) |
| --- | --- | --- |
| Random | 38.624 | 47.554 |
| Greedy | 1.013 | 1.305 |
| DQN seed 0 | 8.633 | 8.378 |
| DQN seed 1 | 9.051 | 9.337 |
| DQN seed 2 | 11.833 | 9.895 |
| DQN three-seed mean | 9.839 | 9.203 |

### E1: paired comparison to greedy

| DQN seed | Relative value improvement (%) | Win / tie / loss (%) | Gap difference (pp) | 95% paired interval (pp) |
| --- | --- | --- | --- | --- |
| 0 | -7.313 | 2.38 / 2.38 / 95.25 | 7.346 | [6.978, 7.767] |
| 1 | -8.084 | 1.46 / 2.50 / 96.04 | 8.035 | [7.717, 8.391] |
| 2 | -9.680 | 2.17 / 1.38 / 96.46 | 9.705 | [9.368, 10.113] |
| mean | -8.359 | 2.00 / 2.08 / 95.92 | 8.362 | [8.057, 8.707] |

### E2: size

| n | Random gap (%) | Greedy gap (%) | DQN gap (%) | DQN seed SD (pp) |
| --- | --- | --- | --- | --- |
| 20 | 42.920 | 2.344 | 11.104 | 2.450 |
| 50 | 42.992 | 1.213 | 9.710 | 1.617 |
| 100 | 43.156 | 0.691 | 9.068 | 0.848 |
| 200 | 43.288 | 0.390 | 8.203 | 0.316 |

### E2: synthetic family

| family | Random gap (%) | Greedy gap (%) | DQN gap (%) | DQN seed SD (pp) |
| --- | --- | --- | --- | --- |
| correlated | 22.418 | 0.563 | 9.487 | 2.357 |
| normal | 35.824 | 0.611 | 9.133 | 2.028 |
| skewed | 49.023 | 2.563 | 13.246 | 1.669 |
| uniform | 47.230 | 0.315 | 7.492 | 1.027 |

### E2: capacity ratio

| alpha | Random gap (%) | Greedy gap (%) | DQN gap (%) | DQN seed SD (pp) |
| --- | --- | --- | --- | --- |
| 0.25 | 45.597 | 1.198 | 12.180 | 2.058 |
| 0.5 | 38.420 | 0.917 | 9.013 | 1.623 |
| 0.75 | 45.251 | 1.363 | 7.370 | 0.830 |

### E3–E4b: stress sets

| Suite | N | Random gap (%) | Greedy gap (%) | DQN gap (%) | DQN seed SD (pp) | Paired DQN−greedy 95% CI (pp) |
| --- | --- | --- | --- | --- | --- | --- |
| E3 | 600 | 43.164 | 0.114 | 31.641 | 9.774 | [31.019, 32.064] |
| E4 | 300 | 45.023 | 3.117 | 15.982 | 1.433 | [11.814, 13.994] |
| E4b | 691 | 47.719 | 1.366 | 9.815 | 0.780 | [7.564, 9.371] |

### E1: isolated single-instance latency

| Method | Median (ms) | 95th percentile (ms) | Timed calls |
| --- | --- | --- | --- |
| Random | 0.0100 | 0.0240 | 24000 |
| Greedy | 0.0470 | 0.1350 | 2400 |
| DQN seed 0 | 5.7511 | 11.9834 | 2400 |
| DQN seed 1 | 5.7915 | 12.3028 | 2400 |
| DQN seed 2 | 5.7925 | 12.3168 | 2400 |
| DQN three-seed mean | 5.7567 | 12.2379 | 7200 |
