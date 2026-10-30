# Meta-analysis summary (n = 119 full-text papers)

Abstract-only (excluded): 4; reserves used: 4; per year: {2020: 11, 2021: 7, 2022: 20, 2023: 21, 2024: 21, 2025: 22, 2026: 17}

## Shares (Wilson 95% CI)

**R3**: ode45/adaptive-ODE 33/119 28% [20, 36]; fixed-step-euler 13/119 11% [6, 18]; zead/higher-order-formula 16/119 13% [8, 21]; simulink 6/119 5% [2, 11]; hardware 1/119 1% [0, 5]; not-stated 50/119 42% [34, 51]
**R4_step_reported**: yes 41/119 34% [27, 43]; no 77/119 65% [56, 73]; unclear 1/119 1% [0, 5]
**R5_discrete_analysis**: yes 21/119 18% [12, 25]; no 98/119 82% [75, 88]
**R6_gain_matched|compares**: yes 23/103 22% [15, 31]; no 67/103 65% [55, 74]; NA 7/103 7% [3, 13]; unclear 6/103 6% [3, 12]
**R6_harmonized|gain_baselines**: matched 18/96 19% [12, 28]; nominal 39/96 41% [31, 51]; nominal_claim 7/96 7% [4, 14]; different 4/96 4% [2, 10]; not_reported 28/96 29% [21, 39]; nominal_any 46/96 48% [38, 58]
**R8_baselines_rerun|compares**: yes 101/103 98% [93, 99]; unclear 2/103 2% [1, 7]
**R9**: none 54/119 45% [37, 54]; constant 2/119 2% [0, 6]; linear 4/119 3% [1, 8]; random 1/119 1% [0, 5]; harmonic 7/119 6% [3, 12]; multiple 49/119 41% [33, 50]; unclear 2/119 2% [0, 6]
**R10|time_claim**: yes 14/42 33% [21, 48]; no 28/42 67% [52, 79]
**R11_code_available**: yes 7/119 6% [3, 12]; no 110/119 92% [86, 96]; unclear 2/119 2% [0, 6]
**R12_physical_experiment**: yes 15/119 13% [8, 20]; no 103/119 87% [79, 92]; unclear 1/119 1% [0, 5]
**step_reported|ode45**: yes 11/33 33% [20, 50]; no 22/33 67% [50, 80]
**step_reported|discrete**: yes 27/30 90% [74, 97]; no 3/30 10% [3, 26]
**discrete_analysis|discrete**: yes 20/30 67% [49, 81]; no 10/30 33% [19, 51]
**reproducible minimum** (solver/scheme stated AND step or tolerance stated AND code released): 3/119 3% [1, 7]
**noise kinds tested (any)**: constant 45/119; linear 36/119; harmonic 35/119; random 23/119; other 1/119; polynomial 3/119
**R1_n_problems**: {'median': 2.0, 'q1': 2.0, 'q3': 3.75, 'n': 118, 'min': 1, 'max': 18}
**R2_max_dim**: {'median': 5.0, 'q1': 3.0, 'q3': 8.0, 'n': 112, 'min': 1, 'max': 4096}
**R7_n_baselines**: {'median': 2.0, 'q1': 1.0, 'q3': 3.0, 'n': 119, 'min': 0, 'max': 10}

## Trends (2020-2022 vs 2023-2026; difference with Newcombe 95% CI)

- R4 step reported: 26% [15, 42] (10/38) -> 38% [28, 49] (31/81); diff +12 pts [-7, +28]; stratum-weighted overall 34% (se 2)
- R3 solver/scheme stated: 47% [32, 63] (18/38) -> 63% [52, 73] (51/81); diff +16 pts [-3, +33]; stratum-weighted overall 58% (se 2)
- R3 discrete scheme: 26% [15, 42] (10/38) -> 25% [17, 35] (20/81); diff -2 pts [-19, +14]; stratum-weighted overall 25% (se 2)
- R5 discrete analysis: 21% [11, 36] (8/38) -> 16% [10, 26] (13/81); diff -5 pts [-22, +9]; stratum-weighted overall 17% (se 2)
- R11 code released: 8% [3, 21] (3/38) -> 5% [2, 12] (4/81); diff -3 pts [-16, +6]; stratum-weighted overall 6% (se 1)
- R12 physical experiment: 8% [3, 21] (3/38) -> 15% [9, 24] (12/81); diff +7 pts [-7, +18]; stratum-weighted overall 13% (se 2)

## Reliability (double-coded pairs: 16; disagreements 3, adjudicated 6)

- R1_n_problems: n 16, exact 100%, within +-1 100%
- R2_max_dim: n 15, exact 80%, within +-1 87%
- R3_sim_method: n 16, exact 100%, kappa 1.00
- R4_step_reported: n 16, exact 100%, kappa 1.00
- R5_discrete_analysis: n 16, exact 100%, kappa 1.00
- R6_gain_matched: n 16, exact 94%, kappa 0.90
- R7_n_baselines: n 16, exact 94%, within +-1 100%
- R8_baselines_rerun: n 16, exact 100%, kappa 1.00
- R9_noise_types: n 16, exact 100%, kappa 1.00
- R10_time_bound_checked: n 16, exact 100%, kappa 1.00
- R11_code_available: n 16, exact 100%, kappa 1.00
- R12_physical_experiment: n 16, exact 100%, kappa 1.00
