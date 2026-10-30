# E7: cost–accuracy Pareto on time-varying matrix inversion

Files:

| file | role |
|---|---|
| `experiments/configs/E7.py` | accuracy sweep config (`CONFIG`, runner format). Run as a script, it builds both task sets |
| `experiments/tasks/E7_common.py` | method table (core + tuned arm + build-time extras + model variants), scales, label rule, FLOP models, `PrefetchDerivative` wrapper, censoring cost proxies, cluster bootstrap |
| `experiments/tasks/E7_timing.py` | clean per-step timing tasks (memguard-compatible: `--build`, `--chunk`; atomic parquet done files; per-round machine probe) |
| `experiments/analyze_E7.py` | statistics, figures, LaTeX tables, `summary.json`, `SUMMARY.md` |

**Scale selection.** Set the environment variable `E7_SCALE` (or `ZNN_SCALE`) to `mini` or `full`; the default is `full`. When `experiments/configs/E7.py` or `experiments/tasks/E7_timing.py` is run as a script, `--scale mini|full` does the same.

## 1. Protocol (as implemented, after the 2026-10-18 review)

**Problem.** `matrix_inversion` (E = A(t)X − I, Zhang & Ge 2005) with κ = 10, freq = 1 rad/s and n ∈ {4, 8, 16, 32, 64, 128}. Each problem seed gives a new A(t) (`"seed": "$seed"`). The singular values of A(t) lie in [0.8, 12] for **every** n (geometric spacing in [1, κ] × (1 ± 0.2)), so κ(A(t)) ∈ [6.7, 15] is fixed across n; ‖Ȧ‖₂ is n-independent too. The residual is the vec 2-norm (Frobenius) of E, which grows like √n at equal per-entry error, so MSSRE values are compared **within** n only.

**Accuracy (runner sweep).**
- τ = 1e-3, horizon T = 6 s (N = 6000 steps), MSSRE = max ‖E‖ over the last 25 % of the horizon; settle diagnostic `settle_ratio` (max over the first half of that window / max over the second half; > 2 flags a transient) and `mssre_w2`.
- 3 problem seeds × B = 4 initial states = 12 trajectories per (n, method). Initial states are U[−2, 2], except `znn_explicit_inverse`, which uses its `Model.x0` hook (Newton–Schulz start × (1 + 1 % noise); the method converges only locally, Getz–Marsden Thm 2.1).
- Diverged trajectories count as MSSRE = +∞; errors/timeouts are missing (NaN) and counted.
- `timing_repeats = 5` lean runs per spec give the runner's end-to-end wall per step (B = 4, data generator included) — a secondary measure for rank agreement only.

**Methods.** Protocol arm: every gain-using method at h = γτ = 0.3 (γ = 300) unless its stability interval ends below 0.3.

| label | model | discretization | h | info | note |
|---|---|---|---|---|---|
| `znn` | znn | euler | 0.3 | analytic | standard ZNN, one LU solve per step |
| `uhlig_3_4` | znn | uhlig_3_4 | **0.0255** = h_opt | analytic | h_max = 0.054 < 0.3 |
| `znn_explicit_inverse` | znn_explicit_inverse | euler | 0.3 | analytic | X ≈ A⁻¹, GEMMs only |
| `gnn` | gnn | euler | **0.007** | none | h σ_max² ≤ 1, half the Euler limit 2/σ_max² |
| `gnn_matched` | gnn | euler | 0.3 | none | h σ_max² = 19–43 ≫ 2: diverges (recorded, not timed) |
| `davidenko` | davidenko | euler | – | analytic | prediction only (reference) |
| `pc_newton_causal` | davidenko | pc_newton_causal | – | analytic | Simonetto NTT, causal information |
| `pc_gradient_causal` | davidenko | pc_gradient_causal | – | analytic | GTT, α = 0.5/L (auto) |
| `newton_tracking_causal` | (unused) | newton_tracking_causal | – | analytic | correction only |

Gain-sensitivity ("tuned") arm, same per-step operations as the protocol rows (so their cost ratio to the protocol row is also an **A/A timing control**): `znn_tuned` and `znn_explicit_inverse_tuned` at h = 1 (dead-beat Euler; Newton–Schulz with feed-forward), `gnn_tuned` at h = 0.0111 = 0.8 h_max.

Extras, picked up at build time from `znnbench.available()` (module/key matching `inverse_free`, `structured_extra`, `low_complex`, `lcc_`), run with registered defaults:
- discretizations `broyden_dtznn_zhang2012` (h = 0.3, M0 = identity) and `cdtznn_mahuang2025` (h_max = 0.172 → h_opt = 0.0465, M0 = newton_schulz), paired with `znn`; dense n² × n² schemes with `max_dim = 400`, so n ∈ {4, 8, 16} only;
- models `lcc_znn` (default activation exp-sig (13)), `hgnn_znn`, `fa_gnn_znn` (Euler; the GNN–ZNN hybrids at h = 0.007 and at the matched h = 0.3, which diverges because they need h(1 + σ_max²) < 2).
- **Model variant** (`E7_common.MODEL_VARIANTS`): `lcc_znn_linear` = `lcc_znn` with `activation='linear'`, labelled "LCCZNN (12), linear φ". It is the only converging member of the scalar-energy (inverse-free, O(mn)) class from U[−2, 2]. It coincides with the *secondary* restatement (Xiao et al. 2025, Eq. (10)) of the inverse-free ZNN of Chen et al. (2024), which is deliberately **not registered** (primary text unobtainable); results must be cited as LCCZNN (12) with linear φ, never as Chen et al.

**GNN-type models run with `info='none'`.** They never read dE/dt (`uses_dEdt = False`): identical dynamics, but `simulate` then skips the wasted Ȧ·X product (zoo finding 2).

**Primary cost: clean per-step time** (`E7_timing.py`).
- One task per (n, problem seed) holds every timed method at B ∈ {1, 4}. After one untimed warm-up run per spec (200 steps), R = 5 rounds; in each round every spec is timed once (lean `simulate`, 2000 steps, `record_every = steps`, no solution error, `online_stats = False`, GC disabled) in a fresh random order. All methods of a round are measured within seconds of each other, and cost ratios are paired per (seed, round) block.
- Time = `wall_dynamics / steps_run` (time inside `disc.step` only). `PrefetchDerivative` evaluates the memoised Ȧ(t_k) inside `data(t_k)`, before the step (without it the synthetic generator, ≈ 14 dense products, runs inside the step and dominates at n = 128; measured 440 → 176 µs/step for ZNN-Euler).
- Reported cost = median over the 15 B = 1 samples (3 seeds × 5 rounds), with IQR and paired cluster-bootstrap CIs.
- **Single thread:** the task env sets OMP/OPENBLAS/MKL/VECLIB/NUMEXPR threads = 1 (memguard merges `env` from tasks.jsonl; `E7_timing.py` also sets them before importing numpy). numpy/scipy use Accelerate. Every row records `cpu_over_wall` (mini: median 0.996, max 1.013).
- **Machine probe (new):** at the start of every round a fixed 128 × 128 dgemm and LU factor + solve are timed (`probe_gemm_us_all`, `probe_lu_us_all`). Accelerate's dgemm runs on the matrix (AMX) unit shared by a CPU cluster; concurrent processes slow it while CPU/wall stays 1.00 (§5, review item R1), so the probe — not `cpu_over_wall` — is the contention check. SUMMARY reports the fraction of rounds > 1.15 × the best probe.
- One-off cost inside every timed run: `pc_gradient_causal` estimates its Lipschitz constant (30 power iterations) at the first step of each run: ≈ 1 % of its per-step time at 2000 steps (≈ 7 % at the mini's 300).

**Secondary costs.**
1. Analytic FLOPs per step from `znnbench/costs.py`: `step_flops(problem, model, disc, B=1, info)['total']` (algorithmic view; `skip_unused_dEdt=False` is stored as `flops_costs_code`). Fallback: idealised leading-order counts `E7_common.IDEAL_OPS`.
2. Measured call-count FLOPs (`flops_exec`) from a `CountingProblem` profile (difference of two runs, 50/250 steps at full scale), LU factorisations counted on a private instance. It does not see discretization-internal work (Broyden/CDTZNN M-matrix updates), so it is used for the core methods only.
3. The runner's end-to-end lean wall (B = 4, data generation included), rank agreement only.

**Statistics.**
- Medians and IQR; paired, two-stage cluster bootstrap (2000 resamples, seed 20261018): seeds with replacement, then rounds (timing) or initial states (accuracy) within each drawn seed, the same draw for every method at that n (the B initial states of one seed share A(t), so trajectories are not independent; checklist item 5).
- Pre-specified contrasts at n_max (ratios of medians a/b) with 95 % and Bonferroni-simultaneous CIs (K = 9).
- **Pareto front** per n with ε-dominance (5 % ties on both axes) at the medians, plus the bootstrap membership frequency; protocol arm (core methods) and all ranked methods.
- **Kendall τ_b** between cost rankings (time vs costs.py / ideal / executed FLOPs / runner wall / B = 4) and of the MSSRE ranking at each n vs n_max. MSSRE ranks are censoring-aware (+∞ = worst, tied). p-values are descriptive only.
- **Scaling exponent** of cost vs n: all-points log–log LSQ and an asymptotic fit over the longest tail whose consecutive local slopes stay within **0.15** of the last one; 2-point fits and tails with last local slope < 2.5 are flagged (pre-asymptotic).
- **Overhead share** (new): the per-step time at the smallest n (n = 4, < 1e3 FLOPs ⇒ < 0.01 µs of arithmetic) is the method's interpreter + call-overhead floor; `overhead_share = floor / cost(n)` (an estimate, as numpy allocation/copies grow ~n²).
- **Censoring** (new): a cell with median MSSRE = +∞ is censored; it stays in every table, enters bootstrap/ranks as +∞, and is drawn in a "diverged" strip (MSSRE = ∞, half-filled marker of the entity, k/K diverged) at its own clean cost or, if it has none, at the cost of its operation-identical proxy (`E7_common.cost_proxy`: `<x>_matched` → `<x>`, `lcc_znn` → `lcc_znn_linear`; marked ‡). SUMMARY.md has a censoring section (censored / partially diverged / failed / timing-diverged / not timed).
- MSSRE < 1e-11 is flagged as at the round-off floor (dagger, grey band).

**Outputs** (`<results>` = `results/mini/E7` or `results/full/E7`).
- Figures (`--figdir`, pdf + png): `fig_E7_pareto` (core methods, keys group `baseline`, wall-clock, protocol front), `fig_E7_pareto_gain` (protocol → tuned arrows), `fig_E7_pareto_flops` (x = costs.py FLOPs / n³), `fig_E7_scaling` ((a) µs/step vs n, (b) paired ratio to ZNN-Euler, (c) effective GFLOP/s with the probe's 1-thread dgemm rate as ceiling), `fig_E7_pareto_extra[_k]` (extras vs the two references; censored extras in the strip).
- Tables (`--tabdir`, booktabs, compile with pdflatex — checked): `E7_pareto.tex`, `E7_headline.tex`, `E7_ratios.tex`, `E7_scaling.tex` (now with floor and overhead share), `E7_kendall.tex`.
- `<results>/summary.json`, `SUMMARY.md`, `E7_stats.csv`, `E7_contrasts.csv`.

## 2. Deviations from PLAN.md and why

1. **h is not 0.3 for every method:** `uhlig_3_4` and `cdtznn_mahuang2025` run at their h_opt (stable interval ends below 0.3); GNN and the GNN–ZNN hybrids run at a stability-scaled h = 0.007 (their matched h = 0.3 rows are kept and diverge); PC/Newton-tracking/Davidenko have no gain. A tuned arm (h = 1 for the ZNN-Euler pair, 0.8 h_max for GNN) separates gain effects from method effects.
2. **Wall-clock comes from dedicated timing tasks**, not only the runner's `timing_repeats`: the runner's wall includes the data generator, is measured on a batch of B = 4 that shares one LU factorisation (favouring LU methods), and drops `wall_dynamics`. Its numbers are kept as a secondary measure.
3. **Low-complexity / inverse-free ZNN:** `inverse_free_znn` is not registered (primary source unobtainable); `lcc_znn` with its default activation diverges at step 1 from U[−2, 2]; the class is therefore represented by `lcc_znn_linear` (labelled as LCCZNN (12), linear φ) plus the inverse-free DTZNNs.
4. **Timing horizon** 2000 steps × 5 rounds, not the 6000-step accuracy run (dense linear-algebra cost does not depend on the state; ≥ 15 ms per run).
5. **B = 4 clean timings** are recorded too (batched-cost discussion).
6. **Pareto analysis** uses ε-dominance and a bootstrap membership frequency.

## 3. Confounds

- **Gain.** Equal nominal h = γτ is not equal effective gain. GNN's effective Euler step is h σ_max² (19–43 at h = 0.3 → divergence; protocol h = 0.007 gives ≤ 1.0, tuned 0.0111). The hybrids' Euler limit is h(1 + σ_max²) < 2 (tested in `tests/test_structured_extra.py`); at 0.007 they sit at ≈ half of it. One Newton correction is a dead-beat (h = 1) step: PC-Newton at h = 0.3 vs ZNN-Euler at h = 0.3 is gain-confounded (0.30× MSSRE), against ZNN-Euler at h = 1 the MSSRE is identical (ratio 1.00) and ZNN is cheaper. Uhlig 3_4 / CDTZNN at h_opt follow the "own optimum" rule, not the matched one.
- **Information / initial state.** The explicit-inverse ZNN starts from the Newton–Schulz guess (uses A(0), local method); all others from U[−2, 2]. The steady-state MSSRE of the linear designs does not depend on x0; the settle ratio checks the window (0 unsettled cells in the mini). GNN/hybrids run with `info='none'` (they ignore dE/dt; identical dynamics).
- **Transients.** `pc_gradient_causal` contracts by ≈ 1 − 0.5/κ(t)² per step (time constant up to ≈ 0.45 s); with T = 6 s the window [4.5, 6] s is settled in the mini (T = 5, n ≤ 64). Check the settle line of the full SUMMARY at n = 128.
- **Unit dependence (checklist item 1).** LCCZNN's activation (13) contains exp(‖E‖²/2): its step-1 overflow from U[−2, 2] depends on the unit of E. FA-GNN-ZNN's fuzzy factor uses absolute breakpoints (u_max = 1, an ASSUMPTION of the zoo): its gain boost is unit-dependent. These conclusions are qualified as such; all other methods are linear in E.
- **Timing.** B = 1, one BLAS thread, data generator excluded. At small n the per-step time is interpreter/call overhead (≥ 50 % for n ≤ 32 in the mini, ≈ 12 % at n = 128 in a single-process micro-sample), so wall-clock contrasts below n_max measure the Python implementation. Concurrent processes inflate GEMM-heavy methods through the shared matrix unit, invisible to CPU/wall (R1). FLOPs misrank time because GEMM runs at ≈ 300 GFLOP/s (1 thread, n = 128) while the LU path runs at ≈ 50–80.
- **Pseudo-replication.** The B initial states of a seed share A(t) (and the linear designs forget x0): CIs resample seeds first; with 3 seeds the cluster bootstrap is coarse.

## 4. Mini validation (re-run after the review, 2026-10-18)

**Scale (`mini`).** Accuracy: n ∈ {4, 16, 32, 64}, seeds {0, 1}, B = 2, T = 5 s, `timing_repeats = 1` → 152 specs in 38 tasks. Timing: seeds {0, 1}, B ∈ {1, 4}, 300 steps × 3 rounds → 8 tasks (one per (n, seed)), 256 timing rows (17 timed methods × B ∈ {1, 4} at n ≤ 16, 15 at n ≥ 32).

**Run.** Fresh build (old partial results deleted), `memguard pool --workers 3 --max-gb 40` for accuracy (2 min 17 s wall, 302 s CPU, 38/38 ok), the same for timing (40 s wall, 8/8 ok), analysis ≈ 10 s. Peak RSS per task: 191 MB (accuracy), 276 MB (timing). **Load caveat:** GitHub-Actions runners on this machine pushed the load average to 100–180 during the timing pool; the probe flagged 8 % of rounds (worst 1.57 × the best dgemm). Mini timings are pipeline checks, not paper numbers.

**Results at n = 64** (B = 1 clean cost, median MSSRE):

| method | µs/step | MSSRE |
|---|---|---|
| GNN (h = 0.007) | 24.2 | 5.4e-1 |
| explicit-inverse ZNN | 28.7 | 7.7e-5 |
| LCCZNN (12), linear φ | 45.6 | 1.9e-1 |
| Newton tracking | 51.7 | 2.2e-2 |
| Davidenko | 65.0 | 3.6e2 |
| ZNN-Euler | 69.3 | 7.7e-5 |
| H-GNN-ZNN (h = 0.007) | 71.0 | 3.2e-1 |
| PC-gradient | 79.9 | 1.0e-3 |
| ZNN-Uhlig 3_4 | 84.0 | 8.8e-12 (floor) |
| PC-Newton | 91.1 | 2.3e-5 |
| FA-GNN-ZNN (h = 0.007) | 122.4 | 2.9e-1 |

- Pareto front (protocol arm) at every n: gnn, znn_explicit_inverse, uhlig_3_4; with the tuned arm: gnn_tuned, znn_explicit_inverse_tuned, uhlig_3_4.
- Contrasts at n = 64: explicit-inverse / ZNN-Euler cost 0.41 (95 % CI 0.39–0.42), MSSRE ratio 1.00; PC-Newton / ZNN-Euler 1.32× cost for 0.30× MSSRE at h = 0.3, but 1.35× cost at identical MSSRE against ZNN-Euler at h = 1; Uhlig 3_4 1.21× cost, MSSRE at the round-off floor. A/A controls: 0.97 [0.94, 1.00] and 1.01 [0.95, 1.03].
- Overhead share (core, median): 100 % / 70 % / 55 % / 31 % at n = 4 / 16 / 32 / 64. All scaling fits are 2-point and pre-asymptotic (local slopes 0.5–0.9 over 32 → 64).
- Kendall τ_b time vs costs.py FLOPs 0.76–0.86 (core); B = 1 vs B = 4 0.93–1.00. costs.py cross-check: executed / costs.py code FLOPs = 0.977 median (0.92–1.00) at n ≥ 32.
- Censoring: gnn_matched, hgnn_znn_matched, fa_gnn_znn_matched and lcc_znn (exp-sig) are censored at every n (4/4); all drawn in the strip at proxy costs; no failures.
- Extras: CDTZNN 61× ZNN cost at n = 16 (MSSRE 5.7e-10), Broyden 6.3× (3.6e-2), hybrids ≈ 0.3 MSSRE (no feed-forward, Eq. (10)/(11)), LCCZNN-linear 0.66× ZNN cost at MSSRE 0.19 (the scalar-energy design's discrete floor; spec §6).

Figures checked visually (`figures/mini/fig_E7_*.png`): one legend per figure in the fixed group order, ≤ 8 series per panel, no dual axes, labelled log axes with units, the diverged strip with an ∞ pseudo-tick and no numeric tick labels inside it.

## 5. Full-scale cost estimate and commands

**Task sets** (built and counted in a scratch folder; nothing launched):

| | accuracy | timing |
|---|---|---|
| specs / tasks | 342 specs (1368 trajectories) / 114 tasks (one (n, method) × 3 seeds per chunk) | 576 specs / 18 tasks (one per (n, seed)) |
| work | 342 × (1 recorded + 5 lean) × 6000 steps, B = 4 | per spec 200 warm-up + 5 × 2000 timed + 300 profile steps |
| core time (measured per-step loop costs, single process, 1 thread) | ≈ 57 core-min (n = 128: 23, n = 16: 13 incl. CDTZNN, n = 64: 8) | ≈ 27 core-min (n = 128 task ≈ 3.6 min, n = 16 ≈ 2.1 min) |
| with ×1.5 load margin | ≈ 1.5 core-h | ≈ 0.7 core-h |
| longest task | `cdtznn_mahuang2025` n = 16 chunk ≈ 10 min | n = 128 task ≈ 3.6 min |
| recommended `--workers` | 24 | **1** |
| expected wall | ≈ 12–15 min (bounded by the CDTZNN chunk) | ≈ 30–40 min |
| peak RSS per task | ≤ 0.3 GB (mini 191 MB); 24 workers < 8 GB | ≤ 0.4 GB (mini 276 MB) |

Total ≈ 1.4 core-h measured, ≈ 2.2 core-h with margin; far below the 200 GB cap.

**Run the timing pool alone, with one worker.** On this M3 Ultra (4 P-clusters of 6 cores, one matrix unit per cluster), concurrent single-thread processes slow a 128 × 128 dgemm by +8 / +40 / +80 / +118 % at 2 / 4 / 8 / 12 processes, LU + solve only by +0.5 / +3 / +8 / +12 %, with CPU/wall = 1.00 throughout. Seven concurrent dgemm loops moved the E7 headline ratio explicit-inverse / ZNN-Euler at n = 64 from 0.42 to 0.48 (explicit inverse +16–19 %, ZNN-Euler +2–3 %); at n = 128, where GEMMs dominate, the bias of an 8-worker timing pool would be larger (≈ +25 % estimated). The accuracy pool may share the machine (its runner wall is then contended too — it is only a rank sanity check). For timing: no other pool, CI runners idle (they pushed the load to 180 during the mini). macOS offers no core pinning, so one worker is the only clean setting.

**Commands** (from the repository root `znn-bench/`; build into a fresh folder because run ids do not see zoo default changes, zoo finding 4):

```bash
# build (both task sets; extras registered at build time are included)
python experiments/configs/E7.py --scale full --out results/full/E7
# accuracy
python tools/memguard.py pool --tasks results/full/E7/accuracy/tasks.jsonl --workers 24 --max-gb 200
# timing: AFTER the accuracy pool, ALONE on the machine (no other pool, CI runners idle), ONE worker
python tools/memguard.py pool --tasks results/full/E7/timing/tasks.jsonl --workers 1 --max-gb 200
# analyze
python experiments/analyze_E7.py --results results/full/E7 --figdir figures --tabdir results/tables
```

**QA after the timing pool** (SUMMARY.md): contention probe < 10 % of rounds above 1.15 × best (else delete the done parquet of `machine_probe.worst_task` / flagged tasks under `results/full/E7/timing/results/` and re-run the pool), CPU/wall ≈ 1.00, A/A ratios with CIs containing 1, settle line empty at n = 128.

**Mini:** the same with `--scale mini --out results/mini/E7`, `--workers 3 --max-gb 40` for both pools, `--results results/mini/E7 --figdir figures/mini --tabdir results/tables/mini`.

## 6. Zoo / core findings (nothing under znnbench/ or tests/ was edited)

| # | finding | status 2026-10-18 |
|---|---|---|
| 1 | `problems/matrix.py`: Ȧ(t) is generated lazily (`_memo("dA")`) on the first `dEdt` call, i.e. inside `disc.step`, so `wall_dynamics` includes the synthetic generator (440 vs 176 µs/step, ZNN-Euler n = 128). Suggest warming it in `data(t)` or timing it separately. | open (E7 workaround `PrefetchDerivative`) |
| 2 | `simulate.f` always evaluates `ctx.E` and `ctx.dEdt`, even for models that ignore them (`gnn`, `newton_flow`, hybrids without feed-forward: `uses_dEdt = False`; `davidenko` never reads E). Suggest skipping dE/dt when `uses_dEdt` is False. | open (E7 runs them with `info='none'`; costs.py has `skip_unused_dEdt`) |
| 3 | `runner.run_spec` keeps only `wall` of the lean timing runs (data generator + two residuals included) and drops `wall_dynamics`. Suggest a `wall_dyn` column. | open |
| 4 | Run/task ids do not change when a zoo factory default changes (`SPEC_VERSION = 1`; e.g. CDTZNN's `M0` default changed identity → newton_schulz during the session). Build full runs into a fresh folder or bump `SPEC_VERSION`. | open |
| 5 | `lcc_znn` default activation exp-sig (13) overflows at the first step from U[−2, 2] (ν₀ ≈ 1e2–1e5 ⇒ exp(ν₀) = inf). A property of the published activation (unit-dependent); suggest documenting an admissible ν₀ range or an overflow guard. | open (E7 adds `lcc_znn_linear`) |
| 6 | `broyden_dtznn_zhang2012` with `M0="newton_schulz"` reached MSSRE 8e9 at n = 16 without being flagged (|x| < 1e10). E7 uses the default `M0="identity"`. | not re-checked (not used) |
| 7 | **new:** `docs/spec/structured_extra.md` still describes `inverse_free_znn` as registered (§1 table, §2 "implemented as…", §4 tests table), while `models/structured_extra.py` and the tests de-register it (`NOT_REGISTERED`). | open (doc owner) |
| 8 | **new, cross-cutting:** Accelerate's dgemm runs on the per-cluster matrix unit; concurrent single-thread pool workers contend for it with CPU/wall = 1.00. Every experiment that reports wall-clock from a parallel pool (the runner's `timing_repeats` under 24 workers, E9 if it times) is biased toward LU-heavy methods. | report to integrator |

## 7. Review log (2026-10-18)

Reviewer: methodology/statistics review of E7 (this session). Severity: HIGH = could change a headline conclusion; MEDIUM = changes a secondary number or the interpretation; LOW = presentation/bookkeeping.

| id | severity | issue | change | effect on numbers |
|---|---|---|---|---|
| R1 | HIGH | Timing contention invisible to the single-thread check: concurrent processes share the per-cluster matrix (AMX) unit that Accelerate's dgemm uses (+8/+40/+80/+118 % dgemm at 2/4/8/12 processes; LU +0.5/+3/+8/+12 %; CPU/wall 1.00). The plan ran the timing pool with 8 workers. | Per-round machine probe in every timing row (`machine_probe`, FORMAT `/3`), SUMMARY contention line, scaling panel (c) ceiling; full-scale timing with **1 worker, alone**; config prints the new command. | Headline explicit-inverse / ZNN-Euler ratio at n = 64: 0.42 alone vs 0.48 under 7 concurrent dgemm loops; ≈ +25 % expected at n = 128 with 8 workers. |
| R2 | HIGH | Censoring: diverged cells (gnn_matched, matched hybrids, exp-sig LCCZNN) silently vanished from every Pareto figure, and the MSSRE Kendall ranking dropped +∞ (`isfinite` filter). | "Diverged" strip (∞ pseudo-tick, half-filled entity marker, k/K, ‡ = proxy cost from the operation-identical sibling); `censored`, `cost_plot`, `n_timing_div` columns; censoring section in SUMMARY/summary.json; Kendall keeps +∞ as tied worst. | 16 censored cells now visible in the mini; τ_b "MSSRE vs n_max (all)" now includes lcc_znn (+∞). |
| R3 | MEDIUM-HIGH | Coverage: `inverse_free_znn` is de-registered and `lcc_znn` (default exp-sig) diverges at step 1, so PLAN's "low-complexity / inverse-free ZNN" had no finite Pareto point. | `MODEL_VARIANTS` + `lcc_znn_linear` (model entries {key, kwargs, label}; exclusion rules match labels; analysis labels from `model_label`; costs.py called with the kwargs), labelled "LCCZNN (12), linear φ" (never Chen et al.). | New point: 0.66× ZNN-Euler cost at n = 64, MSSRE 0.19 (≈ 2500× ZNN's): the O(mn) scalar-energy design is cheap but far off the front. +8 mini / +18 full accuracy specs. |
| R4 | MEDIUM | "Wall per step" at small n measures the interpreter, not the method; not quantified. | Overhead floor (cost at n = 4) and `overhead_share` per (n, method) in stats, `E7_scaling.tex`, SUMMARY section and interpretation. | Core median overhead share 100/70/55/31 % at n = 4/16/32/64 (mini); ≈ 12 % at n = 128 (micro-sample). Headline contrasts are at n_max only. |
| R5 | MEDIUM | Asymptotic-range tolerance 0.35 (checklist: 0.15); 2-point fits not flagged. | `SLOPE_TOL = 0.15`, `two_point_fit` flag in tables/SUMMARY. | All mini fits are 2-point / pre-asymptotic (as before; now labelled). |
| R6 | LOW | Pre-existing LaTeX bug: `_tex_desc` escaped `_` before substituting `sigma_max^2`, leaving `_{\max}` outside math → `E7_headline.tex` did not compile. | Substitute math pieces first. | Tables compile (pdflatex checked). |
| R7 | LOW | Kendall p-values not labelled. | Marked descriptive in SUMMARY and summary.json. | none |
| R8 | LOW | Notes out of date (timing layout, tuned arm, bootstrap, extras list with `inverse_free_znn`, 8 timing workers, task counts). | Notes rewritten (§1–§6), Confounds section added. | Full counts now 342/114 accuracy, 576/18 timing. |
| R9 | check | Single-thread env: verified (tasks.jsonl env + `setdefault` before numpy; dgemm bench CPU/wall = 1.00; mini rows 0.996). Warm-up excluded (untimed warm-up per spec; GC off); `pc_gradient_causal`'s Lipschitz estimate is inside each run (≈ 1 % at 2000 steps). | none needed | — |
| R10 | check | costs.py cross-check (it is under verification elsewhere; read-only here): for all 13 E7 methods at n ∈ {16, 64}, the per-step call counts (E, dEdt, solve_J, JT, LU factorisations) from `wrappers.profile_step_cost` + a counted factor cache equal costs.py's `calls`/`factorizations` exactly; costs.py totals = leading order (2n³ per call + 2/3 n³ per LU) × 1.002–1.036 at n = 64 (the O(n²) vector work), Broyden/CDTZNN dominated by their own n_x² / n_x³ work (6.3e5 and 1.5e8 at n = 16, as the spec states). | none needed | — |
| R11 | check | Fairness / conditioning / memory: κ fixed across n ([6.7, 15]); gain confounds handled by the tuned arm and stated (§3); `simulate` stores only (N_rec, B) residuals, so memory at n = 128, B = 4 is trivial (peak RSS ≤ 0.3 GB). | §3 Confounds | — |
| R12 | check | Pseudo-replication: the bootstrap is paired and seed-clustered (checklist 5); unit dependence (checklist 1) only for exp-sig LCCZNN and the FA fuzzy factor, qualified in §3 and SUMMARY. | none needed | — |
