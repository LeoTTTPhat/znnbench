# Spec: experiment runner (`znnbench/runner.py`, `experiments/run_task.py`)

Area: **runner**. Owner files: `znnbench/runner.py`, `experiments/run_task.py`,
`experiments/README.md`, `tests/test_runner.py`, this file.

The runner implements no neural-dynamics model, so it has no source paper of its
own. It does fix the **conventions** every sweep relies on: the step-size modes,
the batching of gains, how ids are derived, which metrics are recorded and how
failures are recorded. Each convention is stated below with its source, or
flagged **ASSUMPTION** where it is our own choice.

---------------------------------------------------------------------------
## 1. Step-size conventions

Sampling instants are $t_k = k\tau$, for $k = 0, \dots, N$ with $N = \operatorname{round}(T/\tau)$
(this is `simulate`). Each gain entry of a sweep is either

| mode          | given | derived                                 |
|---------------|-------|-----------------------------------------|
| `fixed_h`     | $h$   | $\gamma = h/\tau$                       |
| `fixed_gamma` | $\gamma$ | $h = \gamma\tau$                     |

$$\gamma = \frac{h}{\tau}\ \ (\texttt{fixed\_h}),\qquad h=\gamma\tau\ \ (\texttt{fixed\_gamma}).$$

* **fixed-h** is the discrete-time ZNN (DTZNN) convention: the step size $h$ is held fixed while
  $\tau\to 0$. For linear ZNN with a look-ahead formula of truncation order $q$, the residual is then
  $O(\tau^{q+1})$ (see `tests/test_core.py::test_lookahead_order`).
* **fixed-γ** is the continuous-time convention: $\gamma$ is held fixed, so $h\to 0$ and one order is
  lost (`tests/test_core.py::test_fixed_gamma_euler_is_first_order`).

Sources:
* Y. Zhang, L. Jin, D. Guo, Y. Yin, Y. Chou, "Taylor-type 1-step-ahead numerical differentiation rule
  for first-order derivative approximation and ZNN discretization", *J. Comput. Appl. Math.* 273 (2015)
  29–40, doi:[10.1016/j.cam.2014.05.027](https://doi.org/10.1016/j.cam.2014.05.027). This paper sets out
  the DTZNN framework in which $h=\tau\gamma$ is the step size. (I checked the bibliographic record
  through Crossref but did not see the full text.)
* Y. Zhang, W. Ma, B. Cai, "From Zhang neural network to Newton iteration for matrix inversion",
  *IEEE Trans. Circuits Syst. I* 56(7) (2009) 1405–1415,
  doi:[10.1109/TCSI.2008.2007065](https://doi.org/10.1109/TCSI.2008.2007065). This paper builds the
  discrete-time ZNN and studies how its behaviour depends on the step size (record checked through
  Crossref; I read the abstract only).
* M. Sun, M. Tian, Y. Wang, "Multi-step discrete-time Zhang neural networks with application to
  time-varying nonlinear optimization", Optimization Online preprint 6960 (2018). I read the full text.
  It defines **formula-normalised** step sizes: $h = 2\tau\gamma$ for its special two-step ZeaD formula
  and $h = \tau\gamma/(1+a_1)$ for its general two-step formula. Published version:
  *Discrete Dyn. Nat. Soc.* 2019, 4745759, doi:[10.1155/2019/4745759](https://doi.org/10.1155/2019/4745759).
  **ASSUMPTION:** the journal paper is the same work under a different title; I have not checked
  this.

* Supporting the raw-product convention for the Euler-type DTZNN: the review "A Review on Zeroing
  Neural Networks", arXiv:2507.00387 (2025), §4.2.1, writes the forward-difference discrete ZNN as
  $x_{k+1} = x_k - A_k^{\dagger}[(A_k-A_{k-1})x_k - (b_k-b_{k-1}) + \eta\gamma(A_kx_k-b_k)]$ with
  $\eta$ the sampling gap. The product (sampling gap × gain) multiplies the error, as in our
  $h=\gamma\tau$ (checked by the verifier on the arXiv HTML; the symbol there is $\eta$, not $\tau$).

**ASSUMPTION (convention).** The runner's $h$ is always the raw product $\gamma\tau$, matching
`docs/DESIGN.md` and `tests/test_core.py`, whatever the formula. Some papers normalise $h$ by the
formula's leading coefficient $c_{-1}$ (Sun–Tian–Wang). To compare with such a paper's stability
interval, the analysis must convert $h_\text{paper} = \gamma\tau / c_{-1}$ (or apply that paper's own
scaling). The runner records `gamma`, `h`, `tau` and `disc_describe` (which holds the coefficients)
for exactly this purpose.

---------------------------------------------------------------------------
## 2. Config format

A config can be a Python dict, a `.yaml`/`.yml`/`.json` file, or a `.py` file that defines `CONFIG`,
`CONFIGS` or `get_configs()`. In YAML, exponent floats without a dot (`1e-3`) are read as floats. Plain
PyYAML follows YAML 1.1 and would read them as strings.

| key | type / default | meaning |
|---|---|---|
| `name` | str, `"sweep"` | config name; used in task ids and the `config` column |
| `problems`, `models`, `discretizations` (alias `discs`) | list of `key` or `{key, kwargs, label}` | registry components |
| `noises` | list, `[null]` | `null`/`"none"` = no noise, or `{key, kwargs}` (registry `NOISES`) |
| `taus` | list of float | sampling periods τ |
| `hs` / `gammas` | lists of float | gain entries: `fixed_h` values h and/or `fixed_gamma` values γ |
| `mode` | `fixed_h` / `fixed_gamma` / `both` / omitted | restrict to one list (an error if the other list is given) |
| `infos` | list, `["analytic"]` | `"analytic"`/`"bd1"`/`"bd2"`/`"none"` or `{info, data_noise}` (`simulate` info regime + measurement noise σ) |
| `wrappers` | list of alternatives, `[[]]` | each alternative is `null`/`[]` (none), one wrapper, or a chain `[w1, w2]`; wrapper = `key` or `{key, target, kwargs}`; `{"$sweep": [...]}` in wrapper kwargs gives one alternative per value |
| `seeds` | list of int, `[0]` | `simulate` seed (initial states + noise stream) |
| `B` | int, 1 | initial states per run (batch) |
| `T` | float > 0 or null | horizon; null = the problem's `T` |
| `record_every` | int, 1 | `simulate` recording stride |
| `metrics` | `{eps:[1e-3,1e-6], frac:0.25, extra:[]}` | arguments to `metrics.summarize` plus extra metrics (§4) |
| `traces` | null / `{points: P}` / `{every: k}` | store down-sampled residual/solution-error traces (npz) |
| `max_wall` | float or null | `simulate(max_wall=...)` in seconds → status `timeout` |
| `div_threshold`, `record_solution_error` | 1e10, true | passed to `simulate` |
| `batch_gains` | false | merge the gain values that share everything else into one batch of B·G trajectories |
| `exclude` | list of rules | drop combinations (§3) |
| `skip_unsupported_implicit` | true | record implicit discretization + model without `supports_implicit` as a skip |
| `timing_repeats` | 0 | number R of extra *lean* timing runs (§4); 0 = time the metric run itself |
| `chunk_size`, `balance` | 16, false | defaults for `build` (`chunk_size` ≥ 1) |

Inside component kwargs:
* `{"$sweep": [v1, v2, ...]}` expands into a grid. Several swept kwargs give a Cartesian product
  over the sorted kwarg names.
* The strings `"$seed"` and `"$tau"` are replaced by the run seed and τ. These replacements, and
  `$sweep`, also apply to wrapper kwargs (a swept wrapper chain expands into the Cartesian product
  over all swept kwargs of the chain). Use `"$seed"` to draw a new problem instance per seed.
* `label` may use `{kwarg}` placeholders. The default label is `key(k1=v1,k2=v2)`.

---------------------------------------------------------------------------
## 3. Expansion, ids and exclusion

`expand(config)` takes the Cartesian product, in this order:
problems × models × discretizations × taus × gains × infos × noises × wrappers × seeds.

**Exclusion.** An `exclude` rule is a dict, and a combination is dropped when *all* the rule's
fields match. The fields are:
* `problem`/`model`/`disc`/`noise`: a key or label string, a list of them, or
  `{key, label, kwargs: subset}`.
* `wrappers`: a wrapper key, or `none`.
* `tau`, `info`, `data_noise`, `seed`, `gain_mode`, `h`, `gamma`: a value or a list of values. `h`
  and `gamma` are the derived values from §1. Numbers are compared with relative tolerance 1e-9.

Rules apply to single-gain combinations *before* `batch_gains` merges the gains.

**Id.**
$$\mathrm{id} = \mathrm{SHA1}\big(\mathrm{canonical\_json}(\mathrm{payload})\big)[0{:}16].$$
* The payload holds every field that determines the numbers: the component keys and kwargs (labels
  stripped), the wrappers, τ, the gain mode and values, the info regime, data_noise, seed, B, T,
  record_every, the metrics config, div_threshold, record_solution_error,
  skip_unsupported_implicit and `spec_version`.
* The payload leaves out fields that only affect output: `config` name, labels, `traces`,
  `max_wall` and `timing_repeats`. So the same run gets the same id in every config, and
  `aggregate` can de-duplicate the same run across builds.
* The **task id** (and therefore the memguard done file) is a different hash: it covers the
  *complete* specs of the chunk, output-only fields included (§5). Turning traces on, raising
  `max_wall` after timeouts, changing `timing_repeats` or a label therefore re-runs exactly the
  affected chunks. (Before verification the task id hashed only the run ids, so such edits were
  silently ignored: the chunks counted as done and, e.g., no traces were ever written.)
* `canonical_json` writes plain types with sorted keys and no whitespace. Floats use Python's
  shortest round-trip repr.
* **ASSUMPTION:** integral floats are written as ints, so `kappa: 10` and `kappa: 10.0` hash to the
  same id. A component that treats `10` and `10.0` differently would therefore collide.
* Duplicate ids are dropped, and the first occurrence is kept.
* `tests/test_runner.py::test_run_id_golden_value` pins one id. Bump `SPEC_VERSION` deliberately
  whenever the semantics change.

**Batched gains (`batch_gains: true`).**
* G gain values share one `simulate` call of $B\cdot G$ trajectories. Trajectory $i = gB + b$ starts
  from initial state $b$ (the same B states for every gain) and uses gain $\gamma_g$.
* The initial states are drawn exactly as `simulate` draws them,
  `problem.x0(default_rng(seed), B)`, so with deterministic noise and `data_noise = 0` each
  trajectory equals the unbatched run (tested).
* **ASSUMPTION / limitation:** with stochastic noise or `data_noise > 0`, a batched run and an
  unbatched run use different noise draws: they are statistically equivalent but not bitwise
  identical, because `simulate` draws x0 and noise from one stream (see CORE_CHANGE_REQUESTS).

---------------------------------------------------------------------------
## 4. Execution and recorded quantities

`run_spec(spec)` builds the components in this order:
1. `make_problem`, `make_model`, `make_discretization`, `make_noise`. Problems are cached within a
   chunk (`PROBLEM_CACHE_SIZE = 4` entries, LRU) unless a problem wrapper is present, because problem
   wrappers may be stateful. Seeds are the innermost expansion axis, so with `"seed": "$seed"`
   consecutive specs cycle through `len(seeds)` instances; 4 entries cover the PLAN's 3 seeds (with 2
   entries the cache missed every time).
2. The wrappers, in chain order.
3. Then it checks compatibility, simulates and summarises.

**Statuses (per trajectory).**
* `ok`
* `diverged`: `simulate` flagged the trajectory. MSSRE = ∞ and `diverged_at` is recorded.
* `skipped`: incompatible combination. Two cases:
  * `model.requires ⊄ problem.capabilities()`, or
  * an implicit discretizer with a model that has `supports_implicit = False`, which would otherwise
    silently run explicitly under an implicit label. **ASSUMPTION:** recorded as a skip by default.
* `timeout`: `simulate(max_wall=…)` raised `TimeoutError`.
* `error`: any other exception. The message and the traceback tail go into `message`.

A failed or skipped run still produces B·G rows, with NaN metrics. Nothing in a spec can crash the
chunk.

**Metrics.** These come from `metrics.summarize(sim, eps, frac)`, the core implementation, with
steady window $W = \{t_k \ge t_N - \text{frac}\,(t_N - t_0)\}$:
$$\mathrm{MSSRE} = \max_{t_k\in W}\|E(x_k,t_k)\|_2,\qquad
\mathrm{mean\_ssre} = \operatorname{mean}_{t_k\in W}\|E(x_k,t_k)\|_2,$$
$$t_\varepsilon = \min\{t_k : \|E(x_j,t_j)\|_2<\varepsilon\ \forall j\ge k\},\qquad
\mathrm{chatter} = \frac{\operatorname{mean}_W |r_{k+1}-r_k|}{\operatorname{mean}_W r_k},$$
plus `ss_sol_err` $=\max_W\|x_k - x^*(t_k)\|_2$ when $x^*$ is known.
* These are computed on the *recorded* samples. With `record_every > 1`, chattering and $t_\varepsilon$
  use the subsampled trace.
* Column names follow `summarize`: `t_eps_0.001`, `t_eps_1e-06`, and so on.

Optional `metrics.extra` (runner-defined):
* `final_residual` $=\|E(x_N,t_N)\|$
* `initial_residual` $=\|E(x_0,0)\|$
* `max_residual` $=\max_k\|E(x_k,t_k)\|$ (transient peak)
* `final_sol_err`
* `peak_xdot` $=\max\|\dot x\|_2$ over **every** evaluation of `model.rhs`. Stage evaluations of
  non-causal schemes count. The design-formula noise that `simulate` adds afterwards is not included
  (`PeakXdotProbe`). This is needed for the "matched peak ‖ẋ‖" protocol (E4-P3) and for joint-velocity
  peaks (R).
* any `"module:function"` returning a (B·G,) array or a dict of arrays.

Any wrapped problem, discretizer or noise object that exposes `extra_metrics(sim) -> dict` also
contributes columns.

**Timing.**
* Every spec first does one fully recorded **metric run**, which gives all metrics, traces and
  `wall_recorded`. With `timing_repeats = R ≥ 1`, R extra **lean runs** follow. They use the same
  arguments, except `record_every = N` (only $t_0$ and $t_N$ are recorded), no solution error and
  no peak-ẋ probe, and stateful noises are `reset()` first.
  * The dynamics are identical because recording never feeds back into the state and draws no
    random numbers in `simulate`. A test asserts that the final residuals are bitwise equal, with
    Gaussian noise, `bd2` and data noise.
  * The lean runs therefore time the algorithm without the per-step residual and $x^*(t)$
    evaluations. At n = 32 those evaluations add about 35–40 % (measured: 0.29 vs 0.21 ms/step for
    ZNN-Euler on `matrix_inversion`), and they would compress E7 cost ratios.
* $\text{wall} = \operatorname{median}_r(\text{wall}_r)$ and $\text{wall\_min} = \min_r$ are taken
  over the lean runs (`timing_mode = "lean"`, `timing_runs = R`). With R = 0 both equal the metric
  run's wall (`timing_mode = "recorded"`, `timing_runs = 1`).
* $\text{wall\_per\_step} = \text{wall}/N_\text{run}$.
* $N_\text{run} = N$, unless every trajectory diverged, in which case $N_\text{run} =
  \operatorname{round}(\max_b \text{diverged\_at}_b/\tau)$.
* `wall` is `simulate`'s own timer.
* `setup_wall` covers component construction.
* `peak_rss_mb` is the process high-water mark (`ru_maxrss`).
* `run_task.py` and the task `env` set `OMP/OPENBLAS/MKL/VECLIB/NUMEXPR_NUM_THREADS = 1`. The pool
  supplies the parallelism, and E7 timings are single-thread by construction.

**Traces.**
* Indices are `unique(round(linspace(0, N_rec-1, P)))` (`points`) or every k-th recorded sample
  (`every`). The last sample is always kept.
* They are stored in `<chunk>.npz` under the keys `<run_id>__t` (float64),
  `<run_id>__residual` and `<run_id>__sol_err` (float32, shape (P, B·G)), and `<run_id>__gamma`.
* Read them back with `load_traces(npz, run_id)`.

**Output columns** (one row per trajectory):
* Config columns: `config, run_id, spec_version, {problem,model,disc,noise}_{key,label,kw_<k>}`,
  `wrappers` (canonical JSON), `wrapper_keys`, `tau, gain_mode, n_gains, info, data_noise, seed, B, T,
  record_every, div_threshold, metrics_eps, metrics_frac`.
* Trajectory columns: `traj, x0_index, gain_index, gain_value, gamma, h`.
* Run columns: `status, message, wall, wall_min, wall_recorded, timing_mode, timing_runs, setup_wall,
  steps, steps_run, wall_per_step, n_diverged, peak_rss_mb`.
* Component description columns: `problem_{name,family,n_x,m,describe}`,
  `model_{name,family,uses_dEdt,supports_implicit,requires,describe}`,
  `disc_{name,causal,expected_order,implicit,describe}`.
* Metric columns.
* Bookkeeping: `traces_file`, `chunk`, and `spec_json` (the exact spec; `aggregate` drops it by
  default).
* Kwarg values that are lists or dicts are stored as canonical JSON text. Mixed-type object columns
  are converted to JSON text, so every chunk's parquet can be written.

---------------------------------------------------------------------------
## 5. Tasks, atomicity, aggregation

* `chunk(specs, size)` makes consecutive slices, so problem instances stay together for the cache.
  `balance=True` instead deals the specs, sorted by $\text{cost}=N(1+BG/16)\cdot(1+R)$, in snake order.
  **ASSUMPTION:** this is a rough cost model; Python overhead per step dominates for small batches.
* `write_tasks(name, chunks, out_dir)` writes:
  * `chunks/<id>.json` (format `znnbench-chunk/1`, with the specs, result path and traces path);
  * `tasks.jsonl`, with one memguard task per line:
    `{"id": "<name>-<i:05d>-<sha1(canonical complete specs)[:10]>", "cmd": [<venv python>, "experiments/run_task.py",
    "--chunk", <abs chunk>], "done": <abs results/<id>.parquet>, "cwd": <znn-bench root>, "env": {…=1}}`.
  * `cwd` is required: memguard's default cwd is the parent project folder, and `cmd` holds a relative
    script path.
  * A changed chunk gets a new id and a new done file. "Changed" means that any field of any spec
    changed, output-only fields included (§3).
* `execute_chunk(file)` runs all specs and then writes the npz traces, if any. The parquet is written
  last, through a temporary file and `os.replace`, so **the done file exists only when the chunk is
  complete**. If the parquet already exists, the chunk is skipped (`force=True` re-runs it).
  An exception or `KeyboardInterrupt` leaves no done file and no temporary file (tested). A SIGTERM or
  SIGKILL from memguard leaves no done file; at worst it leaves a hidden `.<name>.tmp-*` file, which
  `aggregate` ignores.
* `aggregate(dir)` concatenates the chunk parquets, oldest first. Which files are read is decided
  by `result_files(dir, current_only)`:
  * By default (`current_only=None`), when any `tasks.jsonl` exists under `dir`, only the done files
    listed there are read. Results of earlier builds of a changed config (removed specs, chunks
    re-run under new task ids) and other parquet files, such as an `all.parquet` written into the
    folder, are ignored. A moved folder is handled by looking up `results/<name>` next to each
    `tasks.jsonl`.
  * With `current_only=False`, or when there is no `tasks.jsonl`, every `*.parquet` under `dir` is
    read recursively (CLI: `aggregate --all`).
  * Rows with the same `(config, run_id, traj)` are de-duplicated, and the newest file wins.
  * The CLI `aggregate --out x.parquet` sanitises column types (mixed-type columns from different
    chunks → JSON text) and writes atomically.
  * Boolean columns with missing values become pandas `boolean`.
  * `columns=` restricts the columns that are loaded.

---------------------------------------------------------------------------
## 6. Tests (`tests/test_runner.py`, 20 tests, about 12 s, well under 1 GB)

* `test_canonical_json_rules`, `test_run_id_golden_value`: canonicalisation, the pinned golden id,
  and a JSON round trip.
* `test_expand_counts_unique_and_stable`: the product count (384 specs); ids are unique and
  invariant to dict key order, `10` vs `10.0`, labels, traces and max_wall; ids change with T, B,
  record_every and metrics.
* `test_gain_modes`: γ = h/τ and h = γτ, plus validation errors (missing gains, negative h,
  unknown keys, unknown registry keys, T ≤ 0, negative timing_repeats, chunk_size 0,
  div_threshold 0).
* `test_seed_substitution_sweeps_and_exclude`: `$seed`, `$sweep`, and exclusion on components,
  info and derived h.
* `test_yaml_and_py_configs`: YAML `1e-2` is read as a float; YAML, dict and `.py` configs give the
  same ids.
* `test_chunk_and_write_tasks_format`: chunk sizes, balanced costs, and tasks readable by
  `tools/memguard.py:load_tasks` with valid cmd/cwd/done/env; task ids are deterministic. Editing
  `traces`, `max_wall`, `timing_repeats` or a label keeps every run id but changes the task id of
  exactly the chunks whose specs changed.
* `test_end_to_end_in_process`: 2 tasks; done files and no temporary files; statuses
  (skipped / diverged / ok); **metrics equal a direct `simulate` + `summarize` call (rtol 1e-12)**;
  `peak_xdot ≥ ‖ẋ(0)‖`; traces; re-running is idempotent.
* `test_metrics_match_hand_written_euler` (added in verification): **independent reference**. A
  hand-written linear-ZNN forward-Euler loop, which uses neither `simulate` nor `LookAhead`, gives
  MSSRE, mean SSRE, chatter, `ss_sol_err`, both $t_\varepsilon$, `peak_xdot` (exactly
  $\max_k\|\dot x_k\|$) and the four extra residual metrics. The runner reproduces all of them at
  rtol 1e-9, in both gain modes, for 2 τ × 3 initial states.
* `test_skip_incompatible_and_wrappers`: a capability skip via a test wrapper; an import-path wrapper
  (gain ×2 at h = 0.3 ≡ h = 0.6); `$sweep` in a two-wrapper chain expands to the 2 × 3 product.
* `test_real_capability_skip`: `znn_explicit_inverse` (sibling area, `requires={'matrix_inverse'}`)
  is skipped on `linear_system` and runs on `matrix_inversion`. Skips itself if those keys are absent.
* `test_sibling_wrappers_module_resolution`: `QuantizedDiscretizer` from `znnbench/wrappers.py` is
  resolved by name with the target inferred; `$tau` substitution. Skips itself if absent.
* `test_errors_timeouts_recorded_and_chunk_survives`: a bad kwarg gives `error`, `max_wall=1e-9`
  gives `timeout`, and a good spec in the same chunk is still `ok`.
* `test_done_file_only_after_complete_chunk`: an interrupted chunk leaves no done or temporary file;
  a re-run completes.
* `test_batch_gains_equals_unbatched`: identical metrics with constant noise (deterministic).
* `test_aggregate_dedupe_columns_and_empty`: de-duplication, `columns=`, a chunk holding only
  skipped runs concatenates cleanly, an empty directory gives an empty frame, and `current_only`
  ignores parquet files that are not listed.
* `test_rebuild_reruns_changed_chunks_and_aggregate_ignores_stale` (regression, added in
  verification): build → run → rebuild with a model removed and traces on. The new chunk is
  pending, stale chunks and an `all.parquet` inside the folder are ignored, the re-run writes the
  traces, and the unchanged run keeps its run id and bitwise-identical MSSRE.
* `test_timing_repeats_and_record_every`: lean timing runs leave the id, MSSRE and `peak_xdot`
  unchanged. It checks `timing_mode`, `timing_runs`, `wall_min ≤ wall` and
  `wall_per_step = wall/N`, and verifies the lean premise directly (final residual of a lean
  `simulate` is bitwise equal to a recorded one under Gaussian noise + `bd2` + data noise).
* `test_command_line_build_local_aggregate`: `python -m znnbench.runner build/local/aggregate` and
  `experiments/run_task.py --build/--chunk`.
* `test_memguard_pool_end_to_end`: `tools/memguard.py pool --workers 2 --max-gb 50` on 2 tasks
  (including a recorded skip). The results equal in-process execution, and a second pool run is a
  no-op ("2 already done"). Skips itself if less than 60 GB of RAM is available, because memguard
  refuses to launch below its 48 GB reserve.

## 7. Open points / not verified
* For the h = τγ notation I read the in-text definition only in the Sun–Tian–Wang preprint. For the
  JCAM 2015 and TCAS-I 2009 papers I checked only the bibliographic records and abstracts.
* With stochastic noise, batched-gain runs are not bitwise reproducible against unbatched runs (§3).
* The runner cannot bisect for stability limits (E2). It supplies dense h grids (`batch_gains`),
  and the bisection belongs in an analysis script.
* Problem instances are cached per chunk (4 entries), and the chunking keeps expansion order. With
  more than 4 seeds per problem the cache misses again. This costs performance only: problems are
  deterministic in their kwargs.
* `peak_rss_mb` is the process high-water mark, so within a chunk it never decreases from one spec
  to the next. It is not a per-spec measurement.

---------------------------------------------------------------------------
## 8. Verification (adversarial review of the runner area)

What was checked, and against what:

1. **Step-size conventions vs. sources.**
   * Sun–Tian–Wang preprint (Optimization Online 6960). I extracted the full text with `pdftotext`.
     Eq. (27) uses $h = 2\tau\gamma$; Eq. (28) uses $h = \tau\gamma/(1+a_1)$ with
     $a_1\in(-1/2,\infty)$. Its two-step formulas have $c_{-1}=1/2$ and $c_{-1}=1+a_1$, so the
     §1 conversion $h_\text{paper}=\gamma\tau/c_{-1}$ is **confirmed** for both models.
   * arXiv:2507.00387 §4.2.1 (Euler-type DTZNN, sampling gap × gain multiplying the error)
     **supports** the raw-product convention $h=\gamma\tau$ for $c_{-1}=1$ formulas.
   * The JCAM 2015 and TCAS-I 2009 full texts were not accessible (paywalled). Only their
     bibliographic records and abstracts were seen, and the abstracts do not state $h=\tau\gamma$.
     Status: **not verified from the primary text**, as already flagged in §7.
2. **Metrics vs. an independent implementation.** A scratch script, and then
   `test_metrics_match_hand_written_euler`, re-implemented linear ZNN + forward Euler by hand
   ($x_{k+1}=x_k+\tau A^{-1}(-\gamma E_k-\partial_tE_k)$) and re-implemented the metric formulas.
   The runner matches at rtol 1e-9 for MSSRE, mean SSRE, chatter, `ss_sol_err`, $t_\varepsilon$
   (both ε), `peak_xdot`, and initial / final / max residual and final solution error, in both
   `fixed_h` (γ = h/τ) and `fixed_gamma` (h = γτ) modes.
3. **Batching.** With `batch_gains` and RK4 / Euler, Gaussian noise, `record_every = 7` (which does
   not divide N = 73) and `traces.every = 3`, the run produces B·G rows. Statuses are per
   trajectory: h = 3 diverges, the others are ok. Trace shapes are (5, B·G). `steps_run` equals 33
   for an all-diverged Euler run at h = 3, τ = 1e-2 (divergence at t = 0.33). Batched runs on
   `matrix_inversion`, `sylvester` and `tvqp_eq` (sibling problems) complete with status ok.
4. **API contract.** No core file was edited. The runner only calls `simulate`, `summarize` and
   the `make_*` factories, and reads `capabilities()`, `requires`, `supports_implicit`, `implicit`
   and `causal`. The memguard task format was checked against `tools/memguard.py` (`load_tasks`,
   `run_pool`: env merged over `os.environ`, cwd default = parent folder). Task env values are now
   coerced to `str`, which `Popen` requires.

Bugs found and fixed during verification:

* **Task ids ignored output-only fields.** The task id hashed only the run ids. Enabling
  `traces`, raising `max_wall` after timeouts, changing `timing_repeats` or a label therefore left
  every finished chunk "done": no traces, no retry, stale timings and labels. Fixed: the task id
  hashes the complete specs. Regression tests are in `test_chunk_and_write_tasks_format` and
  `test_rebuild_reruns_changed_chunks_and_aggregate_ignores_stale`.
* **`aggregate` mixed in stale results.** It read every parquet under the folder. That included
  results of specs removed from the config and the `all.parquet` that the README itself told users
  to write into the folder (double rows with `dedupe=False`). Fixed: by default `aggregate` reads
  only the done files listed in `tasks.jsonl` (`current_only`; `--all` restores the old behaviour).
* **Wall-clock included recording overhead.** That overhead is about 35–40 % at n = 32 and
  includes an $x^*(t)$ solve per step, which biases E7 cost ratios. The `timing_repeats` runs also
  ran with the peak-ẋ probe attached. Fixed: `timing_repeats` now means R extra lean runs (§4).
  Component `extra_metrics` are read right after the metric run, before any timing run.
* **`$sweep` in wrapper kwargs was passed through literally**, as a dict kwarg. It now expands like
  component kwargs.
* **Missing validation:** `T ≤ 0`, `timing_repeats < 0`, `chunk_size < 1` and
  `div_threshold ≤ 0` are now rejected when the config is normalised.
* **Problem cache thrashed** under `"seed": "$seed"` with 3 seeds (2-entry LRU). It now has 4
  entries.
* The CLI `aggregate --out *.parquet` now sanitises mixed-type columns before writing.
