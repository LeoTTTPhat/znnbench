# R: robotics case study (PUMA 560 and Panda), notes

The protocol entry is "R Robotics case study" in `experiments/PLAN.md`. This file records the exact protocol, the rationale, the deviations from PLAN.md, the results of the mini run, the full-scale cost estimate and the commands.

## Files

| file | role |
|---|---|
| `experiments/configs/R.py` | `get_configs()` returns 3 runner configs, `R_off`, `R_on` and `R_units`. Run as a script, it also writes the joint-trace tasks and a combined `tasks_all.jsonl`. |
| `experiments/tasks/R_common.py` | Shared constants: grid, model entries, noise level, joint limits. Also the `JointStats` discretizer wrapper (runner key `experiments.tasks.R_common:JointStats`) and the `ScaledError` problem wrapper of the error-unit arm (runner key `experiments.tasks.R_common:ScaledError`). |
| `experiments/tasks/R_joint_traces.py` | Custom memguard task. It writes per-step joint-velocity traces and ‖E‖ traces for one robot × path, as an atomic npz done file. |
| `experiments/analyze_R.py` | Builds the figures, LaTeX tables, `summary.json` and `SUMMARY.md`. |

**Scale switch.** Set the environment variable `ZNNBENCH_SCALE=mini|full` (`R_SCALE` is also accepted; the default is `full`), or pass `--scale mini|full` when `configs/R.py` is run as a script. The standard runner CLI also works (`ZNNBENCH_SCALE=full python -m znnbench.runner build experiments/configs/R.py --out results/R` builds R_off, R_on and R_units), but it does not write the trace tasks or `tasks_all.jsonl`.

## Problem and its limitation

The keys are `robot_puma560` and `robot_panda` (`docs/spec/problems_robot.md`). The PUMA uses the Corke standard-DH model with a 0.2 m tool. The Panda uses the FER modified-DH model with the end-effector at the flange.

The task is **position-only, unconstrained, velocity-level pseudoinverse control**: θ̇ = J⁺(ṙ_d − γΦ(f(θ) − r_d)), with J the 3×n position Jacobian. It has:
* no orientation task;
* no joint position, velocity or acceleration bounds (no QP);
* no dynamics and no actuator saturation;
* no repetitive-motion term.

So the joint speeds and accelerations reported here are what the controller **commands**. `JointStats` measures them against the actuator limits but never enforces them.

On both arms the last column of the position Jacobian vanishes analytically (problems_robot spec, Sec. 0). The effective redundancy is therefore n − 4, and the last joint receives a round-off-level velocity (|θ̇_n| / ‖θ̇‖∞ ≈ 2e-16, measured).

Paths, both closed at T = 10 s:
* `circle`: radius 0.1 m, plane tilted 30° about X, smooth phase 2π sin²(πt/2T), zero speed at t = 0 and t = T.
* `lissajous`: Yang et al. 2025, eq. (30); A = B = 0.06 m, a = 3, b = 2, δ = π/2, linear phase.

Both paths are anchored at the nominal start p0 = f(θ0).

## Exact protocol

| axis | values |
|---|---|
| robots × paths | {PUMA 560, Panda} × {circle, Lissajous}, T = 10 s (path period = T) |
| control period τ | 1e-2, 3e-3, 1e-3 s (100 Hz, 333 Hz, 1 kHz) |
| gain | **fixed-γ** mode, γ ∈ {10, 100}, batched (`batch_gains`, B·G = 8 trajectories per `simulate` call) |
| model entries | `znn_lin`: `znn`, linear, Euler. `znn_sbp`: `znn`, `sign_bi_power` r = 0.5, Euler. `znn_sbp_ie`: the same activation with `implicit_euler_err` (resolvent of the error dynamics). `sptc_nt_znn`: t_c = 1 s, p = 0.5, noise_bound Δ = 0.05, so ζ = 5Δ = 0.25 m/s and ξ = ζ/γ; Euler. `ieznn`: λ = γ²/4 (default); Euler. |
| noise | none, and constant ν = 0.05 m/s per Cartesian axis (design-formula space, J θ̇ = … + ν) |
| starts, `R_off` | θ(0) = θ0 + U[−0.05, 0.05] rad per joint (`x0_spread = 0.05`), seeds {0, 1, 2} × B = 4, giving 12 trajectories per cell |
| starts, `R_on` | θ(0) = θ0 (E(0) = 0). Deterministic: seed 0, B = 1, one trajectory per cell. |
| error-unit arm, `R_units` | The `R_off` starts (same seeds and B, so every trajectory pairs with an `R_off` trajectory), noise-free, both γ, all τ, with the task error expressed in **mm** instead of m (`ScaledError`, scale 1000: E_s = 1000 E, J_s = 1000 J, recorded residual still in m). Entries: `znn_lin` (invariance control), `znn_sbp`, `znn_sbp_ie`, `sptc_nt_znn` with Δ converted to 50 mm/s (the same physical sign term). |
| recording | `record_every = 1`. ‖E‖ traces at 1001 points (R_off). Steady window = last 25 % (7.5 to 10 s). |
| runner metrics | `mssre`, `mean_ssre`, `t_eps` at 1 mm and 0.1 mm, `chatter`, `chatter_vec`, `rate_alt`, `peak_rate`, `peak_rate_ss`, `drift`, `final_residual`, `max_residual`, `peak_xdot` |
| `JointStats` metrics | `qd_peak` (= `peak_rate`, checked), `qd_util`/`qd_util_ss` (peak speed / per-joint limit), `qd_viol_frac` (fraction of steps over the speed limit), `qdd_peak`/`qdd_peak_ss` (acceleration implied by consecutive commands; the jump from rest at t = 0 is excluded), `qdd_util` (Panda only), `alt_active` (sign-alternation rate of the joint increments, counted on the moving joints only), `qlim_viol` (joint-position-limit violation, rad) |

Model entries are paired with their discretization through `exclude` rules on the model × disc product. Every registry key is filtered by `znnbench.available()` at build time. If `sptc_nt_znn` is missing, the predefined-time entry falls back to `znn` + `pvaf1` (c2 = 0).

**Joint-speed limits.**
* Panda (FER), from the Franka "Robot and interface specifications" page (`frankarobotics.github.io/docs/robot_specifications.html`, re-read 2026-10-18):
  * q̇_max = 2.175 rad/s (J1–J4) and 2.61 rad/s (J5–J7);
  * q̈_max = 15, 7.5, 10, 12.5, 15, 20, 20 rad/s².
* PUMA 560: **ASSUMPTION**, a nominal 2 rad/s for every joint. No primary per-joint velocity limit was found:
  * Corke's MATLAB `mdl_puma560.m` and Python `Puma560.py` define none (the Python file was re-read);
  * the ARTE toolbox file `robots/UNIMATE/puma560/parameters.html` sets `robot.velmax = [1 1 1 1 1 1]` with the comment "not available", and lists an end-effector limit of 0.5 m/s "from datasheet".

  PUMA utilisation numbers are therefore indicative only.

**Statistics** (`analyze_R.py`):
* Medians and IQR over the 12 trajectories of a cell.
* Bootstrap 95 % CIs, 2000 resamples, seeded:
  * of the median MSSRE and median peak speed per cell;
  * of **paired** ratios of medians (pairs = same seed and x0) for the headline comparisons: implicit/explicit SBP, and IEZNN/ZNN under noise.
* Kendall τ_b of the 5-model rankings: by MSSRE between 100 Hz and 1 kHz, between PUMA and Panda, between the MSSRE ranking and the peak-speed ranking, and between the rankings with E in m and E in mm. With n = 5 models the smallest attainable two-sided p is 0.017 and 22+ comparisons share data: p-values are reported as descriptive only, with no multiplicity correction and no significance claims.
* Empirical order of MSSRE in τ at fixed γ:
  * consecutive local slopes;
  * an LS fit over all points;
  * an "asymptotic" fit, extended from the smallest τ for as long as consecutive local slopes agree within **0.15** (the tolerance of the order test in `tests/test_problems_robot.py`). Points at or below 1e-12 m are excluded.

  Both fits are reported, plus a flag. With only 3 rates the order is indicative.
* Censoring: a diverged trajectory has MSSRE = +inf, `peak_rate` = +inf and NaN wrapper statistics. The analysis sets the "larger is worse" wrapper statistics of diverged rows to +inf (`qd_viol_frac` to 1) and keeps +inf as a worst-case value in medians, bootstrap CIs (sentinel, mapped back to inf), paired ratios (inf numerator gives inf, inf denominator gives 0) and Kendall rankings. Nothing diverged is ever dropped. `alt_active` of a diverged row stays NaN (undefined). `n_diverged` / `n_failed` are reported per cell; `t_eps` = inf ("never reached and stayed") enters medians as inf.

## Rationale and deviations from PLAN.md

1. **Two start conditions.** The repetitive-motion drift ‖θ(T) − θ(0)‖ is only meaningful when the arm starts on the path. From an off-path start it mostly measures the initial correction: about 0.34 rad on the Panda circle for every model, as measured in development. The drift is therefore taken from `R_on`. `R_on` is deterministic, because constant noise draws no random numbers, so it runs one trajectory instead of 3 × 4 identical ones (deviation from 3 × 4). All transient, steady-state and actuation statistics come from `R_off`, with 3 seeds × B = 4. The robot problems have no random instance parameter, so the seed only changes the start draw (and the noise streams).
2. **Fixed γ, not fixed h.** A robot controller is designed with a gain γ (Jin et al. 2017 use k0 = γ = 5) and then deployed at a control rate. The discrete error therefore falls like τ for linear ZNN (one order lost, as in PLAN.md's fixed-γ hypothesis). γ = 100 is added as a "high-gain design" stress case, because ZNN papers routinely use γ ≥ 100. It costs almost nothing thanks to `batch_gains`.
3. **Noise for all models.** PLAN.md lists only "IEZNN (+ constant noise)". The same noise is also applied to every model, so that IEZNN's rejection is compared with linear ZNN's bias √3·ν/γ (8.66 mm at γ = 10) and with the "noise-tolerant" claims of SPTC-NT-ZNN and the sign-type terms. Level: ν = 0.05 m/s per axis. **ASSUMPTION:** chosen to be of the order of the path speeds (about 0.1 m/s); it is easy to change via `R_common.NOISE_LEVEL`.
4. **Predefined-time model = `sptc_nt_znn`** rather than `pvaf1`. It comes from a robot-control paper (Li, Guo, Ma, Pan, IEEE TIE 2024), is the comparison model in Yang et al. 2025, and has a noise-tolerance claim. Its noise bound must be set in problem units: the zoo default Δ = 0.5 would give a 2.5 m/s sign term on a robot. We set **Δ = 0.05 = ν**, so ζ = 0.25 m/s, and keep the same value in the noise-free runs (the designer does not know whether noise is present).
5. **"Implicit sign-type"** is the sign-bi-power activation (r = 0.5) stepped with `implicit_euler_err`. It is paired one-to-one with the explicit SBP entry.
6. **Chattering metric.** The online `rate_alt` divides by all n joints. On both arms the last joint's increments are round-off (≈1e-16 relative), so `rate_alt` is capped at (n − 1)/n in clean runs. In violent runs it also counts the random sign flips of that round-off: PUMA explicit SBP at γ = 100, 100 Hz gives `rate_alt` = 0.89–0.91 > 5/6. Figures and tables therefore use `alt_active` from `JointStats`, which counts only joints with |Δθ_i| > 1e-6·max_j|Δθ_j| in both steps and is in [0, 1]. Raw `rate_alt` is kept in `summary.json`. (See "Zoo observations".)
7. **Extra realism metrics** (joint acceleration, joint-limit violation, fraction of steps over the speed limit) are not in PLAN.md. They come at no extra cost through the wrapper.
8. **Joint-velocity traces** (`fig_R_jointvel_*`) come from one off-path start, seed 0 and x0 #0, re-simulated by the custom task. The task reproduces the matching sweep rows exactly (`trace_consistency` in `summary.json`: maximum relative difference 0.0 in final ‖E‖ and peak speed over 168 run–gain pairs).
9. **Mini scale:** `R_off` and `R_units` with seed 0 only (B = 4, all τ, both γ; `R_off` with both noises), and `R_on` noise-free only. The trace tasks are the same at both scales. The mini specs are an exact subset of the full specs (228 of 624 run ids, checked).
10. **Error-unit arm `R_units` (added in review; not in PLAN.md).** Power-type activations are not invariant to the unit of E: with E in units of 1/s m, Φ_s(E) = Φ(sE)/s, so for `sign_bi_power` (r = 0.5, Φ = ½(|e|^½ + |e|²)sign e) the |e|^½ term is scaled by s^-½ and the |e|² term by s. The explicit-Euler SBP steady band is the exact 2-cycle |e_i| = (γτ/4)² **in the unit of E** (1.08 mm = √3·(0.1/4)² m at 100 Hz, matching the data), so "explicit SBP is worse than linear ZNN and chatters" is a statement about metres, not about the scheme. Linear ZNN and IEZNN are exactly invariant (verified on linear ZNN to 3.8e-10 relative). The arm re-runs the unit-dependent entries with E in mm, paired with `R_off`. Noise-free only, because the additive design-formula noise passes through `solve_J` and would be read in scaled units. Cost: +144 runs (+0.06 core-hours).

## Confounds and how they are handled

* **Equal nominal gain, not equal effective gain.** All models share γ (fixed-γ deployment). Near the steady state the SBP term has effective gain γ/(2√|e|) (in the unit of E), SPTC-NT-ZNN has γ/(t_c − t) before t_c, IEZNN adds λ = γ²/4 integral action. Rankings are therefore rankings of "the same γ knob", not of matched controllers; matched-gain / matched-peak protocols are E4's job. Stated in SUMMARY.
* **Unit dependence** of the power-type entries: quantified by `R_units` (above). Headline implicit-vs-explicit numbers are labelled "error in metres".
* **Unequal information.** SPTC-NT-ZNN is given the true noise bound (Δ = ν). This favours it under noise; stated.
* **Start-offset-driven peaks.** The transient peak joint speed of every linear-in-E model is |J⁺(ṙ_d − γE(0))|∞ at the first step, independent of τ and proportional to the arbitrary 0.05-rad start offset. "Every model exceeds the limits at γ = 100" is conditional on that offset; steady-window peaks (`peak_rate_ss`) are reported separately.
* **Transients in the steady window.** Window 7.5–10 s. Linear ZNN at γ = 10 decays as e^{-10t} (e^{-75} at 7.5 s); IEZNN's double pole at −γ/2 gives (1 + 5t)e^{-5t} ≈ 2e-15 at 7.5 s; SPTC t_c = 1 s. No leakage.
* **SPTC near t_c.** The Euler error map has factor 1 − γ/m at m steps before t_c, so h_eff > 2 during the last γ/2 sampling steps for every τ (4 steps at γ = 10, 49 at γ = 100). A faster rate therefore does not remove the blow-up (γ = 100: peak 3e4 rad/s at 100 Hz, 2–4e5 rad/s at 1 kHz). This explains the non-monotone peak speed in τ.
* **Seeds.** The robot problems have no random instance parameter, so the 3 seeds × B = 4 are 12 independent starts; the same seed and x0 index give the same start for every entry (pairing).

## Outputs

* **Figures** (`--figdir`):
  * `fig_R_traces_{circle,lissajous}`: ‖E(t)‖ in mm, median + IQR. Rows are robots, columns are rates; γ = 10, noise-free.
  * `fig_R_traces_noise_{path}`: the same under constant noise.
  * `fig_R_bars_g{10,100}`: grouped bars by model × rate. Top: peak commanded joint speed (log scale), with hollow markers for the steady-window peak and dashed/dotted limit lines. Bottom: chattering (`alt_active`).
  * `fig_R_mssre_rate`: MSSRE (mm) vs control rate. Rows: noise-free and constant noise (with the continuous-time bias line).
  * `fig_R_units`: MSSRE (mm) vs control rate at γ = 10 with E in m (top, = `R_off`) and E in mm (bottom, `R_units`).
  * `fig_R_jointvel_{robot}_{path}`: ‖θ̇(t)‖∞ for each model. Rows: γ = 10 and γ = 100. The three rates are shown as a sequential blue ramp with distinct line styles.
* **Style.** Bars carry a per-entity hatch as secondary encoding (added in review). Model colours come from `keys.style_in_group` with a runtime group `r_model`. Its slots are chosen so that each entity keeps its global colour: ZNN/linear blue, implicit_euler_err orange, IEZNN green, sign_bi_power yellow, SPTC-NT-ZNN (= pvaf1 slot) purple.
* **Tables** (`--tabdir`):
  * `R_tracking.tex`: MSSRE at 3 rates, order (LS / asymptotic), t_1mm;
  * `R_actuation.tex`: peak and steady peak speed, chattering, fraction of steps over the limit, steady peak acceleration;
  * `R_gamma100.tex`;
  * `R_noise.tex`;
  * `R_drift.tex` (on-path drift);
  * `R_units.tex` (error-unit sensitivity, both γ).
* **Summary:** `<results>/summary.json` (all cells with median, IQR and CIs; headline comparisons; orders; Kendall; drift; bookkeeping) and `<results>/SUMMARY.md`.

## Mini run (2026-10-18)

**Execution (re-run from a clean folder after the review fixes).** Build, then `memguard pool --workers 3 --max-gb 40`: 23 tasks (10 R_off, 5 R_on, 4 R_units, 4 traces), 0 failed, 153 s wall, 392 s CPU. The analysis takes 17 s and writes 12 figures and 6 tables. (Before the review: 19 tasks, 104–130 s.) The R_off / R_on / trace task ids were unchanged by the review, so their numbers are identical to the preparer's run.
* Statuses: R_off 960/960 `ok`, R_on 120/120 `ok`, R_units 384/384 `ok`; no divergence, error or timeout.
* `qd_peak` (wrapper) equals `peak_rate` (online statistic) exactly.
* Medians are over 4 trajectories (one seed) and should be treated as preliminary.

**Headline numbers** (γ = 10, noise-free, off-path, **error E in metres**; MSSRE in mm at 100 Hz → 1 kHz). Every entry has nearly the same median on both arms (within about 12 %). The SBP and SPTC rows depend on the unit of E (see the error-unit arm below); the linear-ZNN and IEZNN rows do not.

| model | PUMA circle | Panda circle | Lissajous (both) | peak speed (rad/s) | chattering `alt_active` 100 Hz → 1 kHz |
|---|---|---|---|---|---|
| ZNN linear | 0.024 → 0.0024 | 0.027 → 0.0027 | 0.11 → 0.011 | 0.31–0.44 (transient) | ≈0.003 → 0.0003 |
| ZNN-SBP explicit | 1.08 → 0.011 | 1.09 → 0.011 | 1.09 → 0.011 | 1.5–1.6 | 0.89–1.0 → 0.09–0.58 |
| ZNN-SBP implicit | 0.0023 → 2.3e-5 | 0.0026 → 2.5e-5 | 0.012 → 1.2e-4 | 1.1–1.6 | ≈0 |
| SPTC-NT-ZNN | 8.8 → 0.34 | 8.9 → 0.34 | 8.8–9.0 → 0.34 | 3.5–4.8 at 100 Hz | 0.97–1.0 → 0.88–1.0 |
| IEZNN | 0.014 → 0.0014 | 0.015 → 0.0015 | 0.073 → 0.0073 | 0.31–0.44 | ≈0 |

* **Orders.** At fixed γ, the MSSRE order in τ is 1.00 for ZNN linear and for IEZNN and 2.00 for both SBP variants (local slopes equal, all 3 points asymptotic). SPTC-NT-ZNN: LS 1.42, local slopes 1.53 (large τ) → 1.30 (small τ), flagged pre-asymptotic; asymptotic estimate 1.30 from the two smallest τ. Its band is a mixture of the (γτ)²-type band of its |E|^½ term and the O(τ) band ζτ/2 of its sign term, so its true asymptotic order is 1. (The preparer's tolerance of 0.3 reported 1.42 as "asymptotic".) The explicit-SBP band is the 2-cycle (γτ/4)² (exact order 2); the implicit SBP behaves like a deadbeat (|E| ≲ h) correction.
* **Implicit vs explicit SBP, error in metres** (paired ratio of MSSRE medians): 0.0021–0.0023 on the circle and 0.0107 on the Lissajous, at every rate. The CIs are tight because the steady state hardly depends on the start. Chattering drops from 0.89–1.0 to ≈0.
* **Error-unit arm (E in mm, paired with R_off, γ = 10, 100 Hz).**
  * Control: linear ZNN reproduces R_off to 3.8e-10 relative (MSSRE) and 3e-13 (peak speed) over 24 cells.
  * Explicit SBP: 1.08 mm → 0.0020–0.046 mm (ratio mm/m 0.0018–0.042, CIs tight), chattering 0.89–1.0 → 0.004–0.019, but transient peak speed 1.5–1.6 → 2.1–5.1 rad/s (the |e|² branch is 1000× stronger).
  * Implicit SBP: 0.0023–0.012 mm → 0.0040–0.057 mm (ratio 1.7–4.9; 1.7–5.1 over all rates). In mm units the implicit/explicit ratio is **1.2–2.1, i.e. explicit is the more accurate scheme**, the opposite of the metres result (0.002–0.011).
  * The best of the 5 models by MSSRE is implicit SBP in metres and explicit SBP in millimetres in **12/12** (robot, path, rate) cells at γ = 10; Kendall τ_b between the two rankings 0.4–0.6.
  * γ = 100: explicit SBP becomes violent in mm (MSSRE 1.2–1.8 m, peak 2.7–4.7e6 rad/s, vs 130–240 mm in m), while implicit SBP is essentially unit-invariant (ratio 1.01–1.04) and bounded. SPTC-NT-ZNN improves in mm (γ = 10: 8.8 → 2.4 mm; γ = 100: 1.2–1.8 m → 8.1–8.4 mm) but still chatters (`alt_active` ≈ 1) and keeps its 2–4e4 rad/s pre-t_c spike.
  * Reading: the robust property of the implicit (resolvent) scheme is **boundedness across units and gains**, not accuracy at a given unit. Claims of the form "explicit sign-type ZNN chatters, implicit does not" must state the unit/scale of E (or normalise the error) to be meaningful.
* **SPTC-NT-ZNN, γ = 10.**
  * The post-t_c sign term chatters at the sampling rate.
  * At 100 Hz, 90 % of the control steps exceed the joint-speed limit.
  * The steady-window peak joint acceleration is 660–760 rad/s², against Panda limits of 7.5–20 rad/s².
  * Explicit SBP reaches 96–98 rad/s².
* **γ = 100.**
  * Linear ZNN and IEZNN command 2.9 (Panda) / 4.4 (PUMA) rad/s in the transient: every trajectory is over the limit.
  * Explicit SBP commands 170–470 rad/s at 100 Hz, with MSSRE 130–240 mm and joint-limit violations of 1.3–5.0 rad.
  * SPTC-NT-ZNN reaches 3e4 to 4e5 rad/s. γ/(t_c − t) exceeds the Euler limit h_eff = 2 in the last γτ/2 s before t_c. Its MSSRE is 1.2–1.8 m at 100 Hz and joints are driven 520–970 rad past their limits (angles wind up).
  * Implicit SBP stays bounded (2.8–13.4 rad/s, still over the limits). Its peak grows with the rate, because the resolvent rate is capped by |E|/τ only at low rates.
* **Constant noise, 0.05 m/s.**
  * ZNN linear: 8.66–8.72 mm, i.e. the continuous-time bias √3ν/γ = 8.66 mm.
  * IEZNN: MSSRE identical to its noise-free value (ratio IEZNN/ZNN 1.6e-4 to 8.7e-3, CIs tight). The constant disturbance is rejected exactly in discrete time too.
  * Explicit SBP: 0.17 mm, which is (2ν/γ)²·√3.
  * Implicit SBP: 0.26 mm at 1 kHz. The resolvent rate adds a ν·τ offset.
  * SPTC-NT-ZNN: 0.45–0.55 mm at 1 kHz (chattering band).
* **Drift** (on-path, closed paths, γ = 10):
  * Panda circle 0.33 rad for every model, matching the problems_robot spec table. PUMA circle 0.0073 rad.
  * Lissajous: ≤ 1.2e-3 rad for the non-chattering models; explicit SBP 3.1e-3 to 3.7e-3 rad at 100 Hz.
  * SPTC-NT-ZNN at 100 Hz increases the drift to 0.76 rad (Panda circle) and 0.41 rad (Panda Lissajous), because chattering scrambles the null-space motion.
* **Kendall τ_b.**
  * The MSSRE ranking of the 5 models is identical on PUMA and Panda (τ_b = 1 for every path and rate).
  * Between 100 Hz and 1 kHz it is stable: τ_b = 1 on the circle, 0.8 on the Lissajous.
  * Accuracy and peak-speed rankings agree only weakly (τ_b 0.1–0.6). The most accurate model, implicit SBP, has the 3rd-highest peak speed.

**Figures inspected** (PNG). All 12 render (review: `fig_R_units` added; bars now carry entity hatches; limit-line legend labels carry units). They follow the mandated style: fixed entity colours with marker and line-style encoding, ≤ 5 series per panel, a legend on every multi-series figure, no dual axes, Elsevier DOUBLE width, and log axes with plain tick labels. Fixed during inspection:
* the legend–axes gap;
* `rate_alt` renormalisation values above 1;
* mathtext `max_i` labels;
* crowded decade ticks;
* an off-by-direction bug in the asymptotic-range order fit, caught before the final run.

## Full-scale cost estimate

| part | specs / tasks | trajectories | `simulate` steps | measured wall (mini × scale factor) |
|---|---|---|---|---|
| R_off | 360 specs / 30 chunks (balanced, 12 specs each) | 2880 | 1,719,960 (B·G = 8) | 3 × 152 s = 455 s |
| R_on | 120 specs / 10 chunks | 240 | 573,320 (B·G = 2) | 2 × 71 s = 143 s |
| R_units | 144 specs / 12 chunks | 1152 | 687,984 (B·G = 8) | 3 × 66 s = 199 s |
| traces | 4 tasks | 168 runs × 2 gains | 401,324 (B·G = 2) | 89 s |
| **total** | **624 runs, 56 memguard tasks** | 4272 + traces | **3,382,588** | **≈ 890 s** |

The full build was dry-run into a scratch folder: 360 / 120 / 144 specs, 30 / 10 / 12 / 4 tasks, and the 228 mini run ids are a subset of the 624 full ids.

**Measured wall per step** (clean mini re-run, 3 workers, on a machine shared with other workflows):
* explicit schemes: 164 µs (B·G = 8) and 150 µs (B·G = 2);
* `implicit_euler_err`: 667 / 650 µs. The resolvent is an 80-step bisection.

**Totals.**
* Simulation time is about 890 s, plus about 2 s per task for interpreter start-up and parquet/npz writing (56 tasks): about 0.28 core-hours; **0.45 core-hours with 50 % contingency** (the shared machine may be up to 2× slower).
* The longest single task is about 25–30 s (a trace task, or a balanced chunk with `implicit_euler_err` at 1 kHz).

**Resources.**
* Recommended `--workers 16`. With 56 tasks the wall time is about 2 min; 8 workers takes about 3–4 min.
* Peak RSS per task: 125 MB measured (`ru_maxrss` of the chunk processes, same for R_units); the trace task uses 77 MB. With 16 workers that is about 2 GB in total, far below the 200 GB cap.
* Disk: about 25 MB at mini scale, about 70 MB expected at full scale.

**Exact commands** (from `znn-bench/`):

```bash
# build (writes results/R/{R_off,R_on,R_units,traces}/tasks.jsonl and results/R/tasks_all.jsonl)
python experiments/configs/R.py --scale full --out results/R
# launch under the memory guard (resumable)
python tools/memguard.py pool --tasks results/R/tasks_all.jsonl --workers 16 --max-gb 200
# progress
python -m znnbench.runner status results/R/R_off
python -m znnbench.runner status results/R/R_units
# analyze
python experiments/analyze_R.py --results results/R --figdir figures --tabdir results/tables
```

Mini (as run):

```bash
python experiments/configs/R.py --scale mini --out results/mini/R
python tools/memguard.py pool --tasks results/mini/R/tasks_all.jsonl --workers 3 --max-gb 40
python experiments/analyze_R.py --results results/mini/R --figdir figures/mini --tabdir results/tables/mini
```

## Zoo observations (nothing under `znnbench/` was edited)

1. **`simulate` online `rate_alt` has no magnitude threshold** (`znnbench/simulate.py`, online statistics). The code is `alt_cnt += (np.sign(dx) * np.sign(prev_dx) < 0).sum(axis=1)` and `alt_tot += dx.shape[1]`. Components whose increments are pure round-off are counted in the numerator and the denominator:
   * the robot's last joint has an analytically zero Jacobian column; its increments are about 2e-16·‖Δθ‖∞ and exactly zero in only 0.5–3 % of configurations;
   * the result is that `rate_alt` is capped at (n − 1)/n in clean runs and can exceed it in violent runs (measured 0.89–0.91 > 5/6 on the PUMA, explicit SBP, γ = 100, 100 Hz);
   * the values are not comparable across arms with different n.

   Suggested fix: count only components with |Δx_i| > ε·max_j|Δx_j| in both steps, and normalise by the number of such components (as `JointStats.alt_active` does). This is not a crash and does not affect any other metric.
2. **`sptc_nt_znn` default `noise_bound` = 0.5 is in problem units.** On the robot problems it means a ζ = 2.5 m/s sign term. This is not a code bug: the spec flags Δ as problem-dependent. Configs must set it per problem; R uses 0.05.
3. **Robot problem, info.** Because the last Jacobian column is zero, the PUMA has 2 and the Panda 3 effective redundant DOF (already documented in the spec). Null-space-related metrics (drift) are therefore driven by fewer joints than nominal.
4. **Power-type activations have no length scale (design observation, not a bug).** `sign_bi_power` (and the other power/sign families) act on the raw error in problem units, so their behaviour depends on the unit of E (see `R_units`). A declared, optional normalisation kwarg (e.g. Φ_s(e) = Φ(s·e)/s, default s = 1) in `znnbench/activations.py` would let configs state the scale instead of inheriting it from the problem's unit. R implements the same effect as a problem wrapper (`R_common.ScaledError`).

## Limitations

* The task is position-only, unconstrained pseudoinverse control (see above). Real controllers would saturate or reject the commands reported for SPTC-NT-ZNN, explicit SBP and γ = 100. libfranka rejects commands that violate its velocity, acceleration or jerk limits.
* The PUMA velocity limit is a nominal assumption.
* The joint acceleration is the finite difference of consecutive commands. It excludes the initial jump from rest.
* The noise is a constant design-formula disturbance only. No measurement noise or delay is modelled, and no time-derivative estimation (that is E6 territory).
* With 3 rates, the orders are indicative.
* Bootstrap CIs reflect variability over start configurations only. The dynamics are otherwise deterministic, so the CIs are narrow.
* Only two error units (m, mm) are tested for the power-type entries; the unit is a continuous design parameter, so `R_units` demonstrates the dependence rather than mapping it.

## Review log (methodology review, 2026-10-18)

Found and fixed in the R files (nothing under `znnbench/` or `tests/` was touched):

1. **Major: unit-dependence confound of the power-type entries.** The headline "explicit SBP chatters and is worse than linear ZNN; implicit SBP is ~400× more accurate" held only with E in metres. Added the paired `R_units` arm (`ScaledError` wrapper, config, analysis, `fig_R_units`, `R_units.tex`, SUMMARY section); with E in mm the accuracy ordering of explicit vs implicit SBP reverses in 12/12 cells at γ = 10, while explicit SBP blows up at γ = 100. Interpretation, notes and the headline labels now qualify the metres result.
2. **Major: censoring in the statistics.** `boot_median_ci`, `boot_ratio_ci` and `kendall_b` dropped non-finite values, i.e. diverged trajectories (MSSRE = inf) would have silently left CIs and rankings; diverged rows also had NaN wrapper statistics that medians dropped. Fixed: +inf kept as worst case everywhere, wrapper "larger is worse" statistics of diverged rows set to inf, per-cell `n_diverged` / `n_failed` and a censoring line in SUMMARY. (No divergence occurs in the mini run, so no number changed.)
3. **Minor: asymptotic-range criterion too loose.** Tolerance 0.3 → 0.15; SPTC-NT-ZNN is now flagged pre-asymptotic (local slopes 1.53 → 1.30; true asymptotic order 1), its asymptotic estimate is 1.30 instead of 1.42.
4. **Minor: Kendall p-values** with n = 5 and 22+ overlapping comparisons are now labelled descriptive, with no multiplicity correction.
5. **Minor: SPTC mechanism statement.** The blow-up near t_c occurs during the last γ/2 sampling steps for every τ (not "γτ/2 seconds at γ = 100"); interpretation now says so and reports the 1 kHz peak too.
6. **Minor: confounds not stated.** Equal nominal γ ≠ equal effective gain, the SPTC information advantage, and the start-offset dependence of transient peaks are now stated (Confounds section and SUMMARY).
7. **Minor: figure style.** Bars had colour as their only encoding; added per-entity hatches (secondary encoding). Limit-line legend labels now carry units.

Checked and found sound: protocol fidelity to PLAN.md (robots, paths, T, rates, model list, metrics, figures); steady window (7.5–10 s, no transient leakage); step-size convention (fixed γ, h = γτ, Euler c0 = 1); pairing by seed and x0 (same start draw for every entry; batched == unbatched); aggregation keys (entry = model label + discretization); medians/IQR; seeded 2000-resample bootstrap; trace-task reproduction of the sweep rows (0 difference); wrapper `qd_peak` = online `peak_rate`; memguard task format and atomic done files; cost and memory estimates (re-measured).

## Matched-effort arm (2026-10-26)

Added on 26 Oct 2026 to compare the robot accuracy gain at matched control effort. At γ = 10 implicit SBP (`sign_bi_power` r = 0.5 stepped by `implicit_euler_err`) is 10–100 times more accurate than linear ZNN with E in m, but it commands 3–4 times the peak joint speed. The arm asks whether it is still more accurate when both controllers use the same control effort. Every number below is taken from `results/R_matched/SUMMARY.md`.

**Files.** All of them are new. No existing R file, config, result or package file was changed.

| file | role |
|---|---|
| `experiments/configs/R_matched.py` | `protocol()` and the task list: 24 memguard tasks, one per robot × path × rate × error unit |
| `experiments/tasks/R_matched_common.py` | `EffortStats` wrapper: a `JointStats` subclass that only adds measurements (`qd_peak_late`, `qd_rms`, `qd_rms_late`, `qd_first`, `qd_peak_early`, `k_qd_peak`, `t_qd_peak`). It also holds the batched evaluation (the 12 R_off starts × G gains in one `simulate` call) and the matched-gain search. |
| `experiments/tasks/R_matched_search.py` | Custom task. It runs the γ = 10 reference runs, direction A for linear ZNN and IEZNN and direction B for SBP, for both matched effort measures. The done file is a JSON search log, written after the per-trajectory parquet. |
| `experiments/analyze_R_matched.py` | Sanity reproduction and comparisons. Writes `SUMMARY.md`, `summary.json`, `R_matched_table.tex` and `fig_R_matched`. |
| `results/R_matched/` | `tasks.jsonl`, `protocol.json`, `results/*.json` (search logs), `results/*.parquet` (43,008 trajectory rows), `SUMMARY.md`, `summary.json`, `R_matched_table.tex`, `pool.out` |
| `figures/fig_R_matched.pdf/png` | Accuracy–effort curves: median MSSRE against median peak joint speed, for 100 Hz and 1 kHz and the 4 robot × path cells. Drawn at the 390 pt text width. |

**Protocol.**
* Cells: {PUMA 560, Panda} × {circle, Lissajous} × {100, 333, 1000 Hz} × E in {m, mm} (`ScaledError` ×1000), noise-free.
* Starts: the 12 R_off off-path starts (seeds 0–2 × B = 4), so every trajectory pairs with an R_off (or R_units) trajectory.
* Linear ZNN and IEZNN are linear in E. They are therefore simulated in m for both unit cells.
* Effort measures:
  * main: the median over the 12 trajectories of the peak commanded joint speed max_{k,i} |θ̇_i| (`qd_peak`, the R_off measure);
  * secondary: the RMS joint speed over the run, sqrt(mean_k ‖θ̇_k‖₂²) (`qd_rms`);
  * reported only: the peak over the last 75 % of the steps (t ≥ 2.5 s, `qd_peak_late`).
* **Direction A (main):** raise the gain of linear ZNN, and of IEZNN with λ = γ²/4, from 10 until its median effort equals that of SBP at γ = 10 in the same cell.
* **Direction B (check):** lower the gain of SBP from 10 until its median effort equals that of linear ZNN at γ = 10.
* The matched gain is the first crossing of the target along the search direction, starting from γ = 10.
* Search on log γ:
  * stage 1: a 16-gain grid, [10, min(150, 1/τ)] for A and [0.25, 10] for B (range extensions were available but never needed);
  * two refinements with 8 interior gains per measure;
  * at most 3 secant steps on (log γ, log effort), until |effort/target − 1| ≤ 0.2 %.
* A match is accepted at the brief's tolerance of 2 %.
* MSSRE is compared as the ratio of medians SBP / comparator, with a paired bootstrap 95 % interval (2000 resamples, pairs = same start). The interval is descriptive only, because the paths are deterministic.

**Deviations from the lead's design, with reasons.**
1. *Batched starts.* All 12 starts and all gains of a stage share one `simulate` call (up to 192 trajectories) instead of the runner's one spec per seed. This is about 10 times cheaper per gain. The runs are noise-free, so the seed enters only through the start draw. The γ = 10 runs reproduce the R_off and R_units rows **exactly**: maximum relative difference 0 over 14 metrics, 72 task × entry sets.
2. *Search method.* A batched bracketing search plus secant steps on log γ replaces plain bisection. The tolerance is the same and fewer sequential calls are needed. All 144 searches matched; the largest |effort/target − 1| is 1.2e-4 and the largest final bracket is 0.3 %.
3. *No search for the steady-state measure.* The late-window peak is flat in γ: it changes by at most 0.16 % for linear ZNN between γ = 10 and 150. It therefore cannot select a gain and is reported instead. The RMS speed, which does depend on the gain, was added as the secondary matched measure.

**Results.**
* **The extra effort of SBP is the start transient.**
  * Its peak is the very first command in 92–100 % of the trajectories of a cell.
  * 63–80 % (m) and 64–91 % (mm) of the peak lies above the late-window level.
  * From on-path starts (R_on), SBP's peak and linear ZNN's peak differ by at most 0.15 %.
  * After 2.5 s every controller at every gain commands the same joint speeds (late-window peaks within 0.48 %). The steady-state effort is the path's feed-forward and cannot be traded for accuracy.
* **Direction A, E in m.**
  * Linear ZNN needs γ* = 30.8–44.2.
  * SBP stays more accurate in all 12 cells: 2.85–3.27 times (100 Hz), 8.09–9.04 times (333 Hz) and 22.9–25.5 times (1 kHz). At the nominal gain the factors are 9.80–10.4, 32.6–34.9 and 97.8–105.
  * IEZNN at the same effort has the same γ*, because its first command is identical. Its factor is 0.422–0.709 at 100 Hz (IEZNN is more accurate in 4/4 cells), 0.964–1.70 at 333 Hz and 2.56–4.50 at 1 kHz.
* **Direction A, E in mm.**
  * γ* = 32.1–94.1.
  * Factors 0.558–1.55 (100 Hz, SBP better in 2/4), 0.971–3.27 (333 Hz, 3/4) and 2.12–7.75 (1 kHz, 4/4).
  * IEZNN is more accurate than SBP in all 12 cells (factor 0.102–0.541).
* **Direction B.**
  * E in m: SBP's γ* = 2.25–2.66. SBP's MSSRE hardly depends on its gain (exponent −0.045 to −0.007), so it keeps 9.25–10.3 (100 Hz), 30.6–34.3 (333 Hz) and 91.7–103 (1 kHz) times linear ZNN's accuracy.
  * E in mm: γ* = 0.721–1.70. Factors 0.148–0.257 (100 Hz), 0.202–0.564 (333 Hz) and 0.270–1.55 (1 kHz, SBP better in 2/4).
* **Mechanism.** Writing MSSRE ∝ γ^p at fixed τ:
  * linear ZNN has p = −1.03 to −0.987;
  * IEZNN has p = −2.12 to −1.90;
  * SBP has p ≈ 0 in m and p = −1.76 to −0.985 in mm.
  
  Raising a linear design's gain buys accuracy in proportion (IEZNN: quadratically), whereas SBP in m already sits at its floor.
* **RMS-matched (secondary).** Same picture. Direction A gives 2.36–2.67 (m, 100 Hz), 16.4–19.0 (m, 1 kHz), 0.613–1.87 (mm, 100 Hz) and 3.78–10.7 (mm, 1 kHz). Direction B gives 9.16–10.2, 87.4–102, 0.181–0.517 and 0.376–4.68.
* **Joint-speed limits.**
  * With E in m, every matched run is within the limits (largest peak/limit 0.988).
  * With E in mm at 333 Hz and 1 kHz, SBP at γ = 10 itself exceeds a limit in 25–75 % of the trajectories (peak/limit up to 4.47). The peak-matched comparators exceed one in 8–83 %.
  * The direction-B runs are within the limits everywhere (largest peak/limit 0.534).

**Caveats.**
* *Direction B and the floor.* Direction B's target (linear ZNN's peak at γ = 10) lies only 0.4–24 % above the path's feed-forward floor. SBP's γ* is therefore set by where its start transient falls to the path speed. In mm this makes the B result sensitive to the target level; in m it does not, because p ≈ 0.
* *Two effort levels.* Directions A and B compare at different effort levels: SBP's peak versus linear ZNN's peak. In mm the ranking depends on that level: linear ZNN wins at low effort, SBP at high effort and fast rates.
* *Scope.* Effort means the commanded peak (or RMS) speed only. Joint accelerations and jerk are not matched. The problem stays position-only and unconstrained, the PUMA speed limit is an assumption, and the intervals are descriptive.
* *Rounding of the nominal factors.* Computed from unrounded medians, the nominal factors are 9.80–10.4 (100 Hz) and 97.8–105 (1 kHz). Ratios of the 3-digit medians of the robot table give "9.7–10.4" and "97–105"; quote the ratios of the unrounded medians, 9.8–10.4 and 98–105.

**Execution (2026-10-26).**
* Pool: `memguard pool --workers 8 --max-gb 60`, 24 tasks, 0 failed.
* Wall time 139 s (12:40:53–12:43:12) on a shared machine (1-min load average 8.6 on 32 cores at launch).
* CPU 943 s user + 12 s system, measured by `/usr/bin/time` on the pool. This is about 0.27 core-hours; the sum of the task CPU times is 929 s.
* The longest task took 84 s (1 kHz) and the maximum peak RSS was 170 MB per task.
* The memguard log shows only the pool start and done lines: no kill, no limit event.
* Analysis: 4 s.

```bash
# from znn-bench/
python experiments/configs/R_matched.py --out results/R_matched
python tools/memguard.py pool --tasks results/R_matched/tasks.jsonl --workers 8 --max-gb 60
python experiments/analyze_R_matched.py --results results/R_matched --rbase results/R --figdir figures
```
