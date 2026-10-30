# ZNN-Bench experiment protocol (E1–E9 + robotics case study)

All sweeps: configs in `experiments/configs/E*.py` → `znnbench.runner` → `tasks.jsonl` → `python tools/memguard.py pool --tasks ... --workers 24 --max-gb 200` → parquet in `results/E*/` → `experiments/analyze_E*.py` → `figures/` + `results/tables/`.
Statistics: 3 problem seeds (instances) × B = 4 random initial states (unless stated); report median and IQR across the 12 trajectories, bootstrap 95% CI (2000 resamples) for headline comparisons; Kendall τ_b for ranking agreement. Default problem frequency 1 rad/s, κ = 10, T = 6 s (steady window = last 25 %).

Two step-size modes are always distinguished:
- **fixed-h**: h = γτ held constant while τ → 0 (the DTZNN convention, γ = h/τ);
- **fixed-γ**: γ held constant (the continuous-time convention, h = γτ → 0).

## E1 Order verification (what residual order does each formula deliver?)
Formulas: every registered causal discretization (Euler, Taylor–Zhang, ZeaD 5/6/N-instant, Uhlig orders 5–8, AB2–AB4) + RK4 (non-causal reference) + implicit variants. Model: linear ZNN. Problems: linear_system (n=4), matrix_inversion (n=4, n=16), sylvester (n=4), tvqp_eq, nonlinear_equations. τ ∈ {1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4}. h ∈ {formula's recommended/optimal h, 0.1}. Both modes.
Outputs: MSSRE vs τ curves; fitted slope (above the round-off floor) vs claimed order (Table); floor level. Hypotheses: slope = q+1 at fixed h, = q at fixed γ (one order lost).

## E2 Stability maps
For each (formula × activation) find max stable h by bisection on h ∈ (0, 3] (τ = 1e-2, fixed-h, T = 6, stability = no divergence and final residual < 10×initial floor). For linear activation compare with the theoretical interval from the root locus of the error recursion c_{-1}ξ^s + (c_0 + h)ξ^{s-1} + c_1ξ^{s-2} + … (computed in analysis). For nonlinear activations, also map stability vs initial error magnitude ‖e(0)‖ ∈ {0.1, 1, 10} (power-type tails shrink the stable region).

## E3 Convergence-time audit (finite/fixed/predefined-time claims)
Models: sign-bi-power, fixed-time, predefined-time activations, strictly predefined-time with time-varying gain (with/without saturation), varying-parameter. (a) Continuous-time: RK4 with τ = 1e-5 (or adaptive high accuracy), measure time to ‖E‖ < 1e-6 and compare with the theoretical bound → conservatism ratio. (b) Discrete-time: time-to-ε (1e-3, 1e-6) vs τ ∈ {1e-2, 1e-3, 1e-4} and h; predefined T_c ∈ {1, 0.3, 0.1, 0.03, 0.01} s relative to τ → show breakdown when T_c ≲ 10τ; time-varying gains ∝ 1/(T_c − t) → divergence unless saturated. Also show ε below the residual floor is never reached.

## E4 Fairness re-ranking
Panel of ~8 "proposed-vs-baseline" models (linear ZNN, power-sigmoid, sign-bi-power, fixed-time, predefined-time, VP-ZNN, strictly-PT TVG, fractional PT-ZNN, IEZNN) on matrix_inversion (n=4) and tvqp_eq.
Protocols: P0 "as published" (each model's own paper parameters, continuous-time RK4 τ = 1e-4); P1 matched linearised gain (γΦ'(0) equal; for non-Lipschitz Φ match the secant gain at ‖e(0)‖); P2 matched stability-limited step (each at 0.8 × its max stable h, discrete Euler τ = 1e-3); P3 matched peak ‖ẋ‖ (scale γ so the transient peak control effort equals a common budget); P4 discrete fixed τ ∈ {1e-2, 1e-3} with published parameters. Rank by time-to-1e-3 and by MSSRE; Kendall τ_b between P0 and each other protocol; how often the "proposed" model stays best.

## E5 Noise robustness under discretization
Noises: constant, linear ramp, Gaussian (per-step), harmonic, random-walk. Models: linear ZNN, IEZNN, DIEZNN, harmonic-rejection ZNN, fractional PT-ZNN, sign-bi-power. τ ∈ {1e-2, 1e-3, 1e-4}, h ∈ {0.1, 0.3}, Euler and Taylor–Zhang. Compare residual floors with continuous-time predictions (e.g. IEZNN → 0 under constant noise in continuous time).

## E6 Information realism (where does the time-derivative come from?)
Info: analytic, bd1, bd2, estimated (alpha-beta, Kalman, Savitzky–Golay). Data noise σ ∈ {0, 1e-8, 1e-6, 1e-4, 1e-2}. Formulas: Euler, Taylor–Zhang, highest-order ZeaD, highest-order Uhlig. τ grid as E1. Hypothesis: high-order advantage collapses (noise amplified ∝ σ/τ); best formula depends on regime → regime map figure.

## E7 Cost–accuracy Pareto
matrix_inversion n ∈ {4, 8, 16, 32, 64, 128}: standard ZNN (LU solve per step), explicit inversion ZNN (X ≈ A⁻¹), low-complexity ZNN, inverse-free ZNN, Newton tracking, prediction–correction, GNN. Wall-clock per step (median of 5 repeats, single thread: OMP/OPENBLAS/VECLIB threads = 1) and analytic FLOPs vs MSSRE → Pareto fronts.

## E8 Implicit vs explicit discretization of sign-type ZNN
Activations: sign, sign-bi-power (r ∈ {0.25, 0.5, 0.75}), fixed-time, predefined-time. Discretizations: Euler, implicit_euler_err, consistent (homogeneous), implicit Taylor–Zhang, semi-implicit. Problems: time-invariant (finite-step convergence check) and time-varying (linear_system, matrix_inversion). Metrics: MSSRE, chattering index, steps to exact convergence, stable h range.

## E9 GNN vs ZNN vs prediction–correction vs Uhlig vs Newton at matched cost
Problems: linear_system, tv_unconstrained_opt, tvqp_eq. Match per-step FLOPs (e.g. PC with C correction steps vs ZNN with higher-order look-ahead). Hypotheses: GNN lag floor O(1/γ) (and stability γτσ_max² < 2); ZNN ≈ PC ≈ Uhlig at matched cost.

## R Robotics case study
PUMA560 and Panda, circle and Lissajous paths, T = 10 s, control rates 100 Hz–1 kHz (τ = 1e-2…1e-3). Models: linear ZNN, sign-bi-power, predefined-time, IEZNN (+ constant noise), implicit sign-type. Metrics: position error (MSSRE, mm), peak joint velocity (vs typical limits ~2 rad/s), chattering of joint velocities, joint drift.
