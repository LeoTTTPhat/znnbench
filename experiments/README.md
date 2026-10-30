# Running ZNN-Bench sweeps

Every sweep goes through four stages:
1. a config is expanded into run specs;
2. the specs are grouped into chunks, and a `tasks.jsonl` is written;
3. `tools/memguard.py pool` runs the chunks under the 200 GB memory cap;
4. the per-chunk parquet results are aggregated.

The runner is `znnbench/runner.py`. Its full reference, covering config keys, ids, statuses, metrics
and output columns, is in [`docs/spec/runner.md`](../docs/spec/runner.md). The experiment protocol is
in [`PLAN.md`](PLAN.md).

All commands below are run from the `znn-bench/` folder, using the project environment
(`python`, see `requirements.txt`).

## 1. Write a config

You can write a config in YAML, in JSON, or as a `.py` file that defines `CONFIG`, `CONFIGS` or
`get_configs()`. For example, `experiments/configs/E1_smoke.yaml`:

```yaml
name: E1_smoke
problems:
  - key: linear_system
    kwargs: {n: 4, kappa: 10, seed: "$seed", T: 6.0}   # "$seed" -> one problem instance per seed
models: [znn]
discretizations: [euler, taylor_zhang, rk4]
taus: [1e-1, 3e-2, 1e-2, 3e-3, 1e-3]
hs: [0.1, 0.3]            # fixed_h:     gamma = h / tau
gammas: [10.0]            # fixed_gamma: h = gamma * tau   (omit one list, or set mode:)
infos: [analytic, {info: bd1, data_noise: 1e-6}]
noises: [null, {key: constant, kwargs: {level: 0.1}}]
wrappers: [null]          # alternatives; e.g. [null, {key: QuantizedDiscretizer, kwargs: {bits: 16, frac_bits: 10}}]
seeds: [0, 1, 2]
B: 4                      # initial states per run
record_every: 1
metrics: {eps: [1e-3, 1e-6], frac: 0.25, extra: [final_residual, peak_xdot]}
traces: {points: 256}     # optional down-sampled traces (npz), off when omitted
max_wall: 1800            # seconds per simulate() call -> status "timeout"
timing_repeats: 0         # E7: e.g. 5 extra "lean" timing runs (no recording overhead); 0 = time the metric run
exclude:
  - {disc: rk4, info: bd1}            # drop combinations (all fields must match)
batch_gains: false        # true: all h (or gamma) values of a run share one batch of B*G trajectories
chunk_size: 20
```

Useful features:
* `{"$sweep": [...]}` inside kwargs expands into a grid, for example `n: {$sweep: [4, 16, 64]}`.
* `"$seed"` and `"$tau"` are replaced by the run seed and the sampling period τ.
* Both also work in wrapper kwargs, e.g.
  `wrappers: [null, {key: QuantizedDiscretizer, kwargs: {bits: 16, frac_bits: {$sweep: [8, 10, 12]}}}]`
  gives one alternative per value.

## 2. Build the tasks

```bash
python -m znnbench.runner build experiments/configs/E1_smoke.yaml --out results/E1_smoke --chunk-size 20
# equivalent: python experiments/run_task.py --build experiments/configs/E1_smoke.yaml --out results/E1_smoke
```

This writes the following files:
* `results/E1_smoke/config.json`: a copy of the config.
* `results/E1_smoke/specs.jsonl`: every run spec, each with its stable id.
* `results/E1_smoke/chunks/<task>.json`: one file per chunk.
* `results/E1_smoke/tasks.jsonl`: one memguard task per chunk, in this form:
  `{"id", "cmd": [python, experiments/run_task.py, --chunk, <chunk>], "done": results/<task>.parquet, "cwd": znn-bench, "env": {single BLAS thread}}`.

Keep one output folder per config. Add `--balance` to spread expensive specs (small τ) evenly across
the chunks.

## 3. Launch under the memory guard

```bash
python tools/memguard.py pool --tasks results/E1_smoke/tasks.jsonl --workers 24 --max-gb 200
```

* **Resumable.** A task whose `done` parquet exists is skipped. The parquet is written atomically
  (temporary file + rename) only after every spec in the chunk has finished. After a kill or crash,
  re-run the same command and only the missing chunks run.
* **Changed configs.** Re-running `build` into the same folder after editing the config gives
  new task ids to exactly the chunks whose specs changed. This includes output-only options such
  as `traces`, `max_wall`, `timing_repeats` and labels. The next pool launch re-runs only those
  chunks; the run ids, and therefore the numbers, of unchanged runs stay the same. Old result files
  are kept on disk but ignored by `aggregate` (see below).
* **Robust.** A failing spec does not fail its chunk. It is recorded with status `error`, `timeout`
  or `skipped`; an incompatible model and problem pair (`model.requires ⊄ problem.capabilities()`)
  is recorded as `skipped`.
* **Logs.** Per-task stdout goes to `logs/tasks/<task>.log`, and pool events go to
  `logs/memguard.log`.
* **Progress.** Run `python -m znnbench.runner status results/E1_smoke`.
* **Small sweeps without the pool** (only for a handful of cheap specs):
  `python -m znnbench.runner local results/E1_smoke`.
* Each worker uses one BLAS thread, so set `--workers` to roughly the number of physical cores you
  want to use (32 on the M3 Ultra).

## 4. Aggregate

```python
from znnbench.runner import aggregate, load_traces
df = aggregate("results/E1_smoke")      # one row per trajectory, config columns flattened
# reads only the done files listed in results/E1_smoke/tasks.jsonl (current build); pass
# current_only=False to read every *.parquet under the folder instead
ok = df[df.status == "ok"]
ok.groupby(["disc_key", "gain_mode", "tau"]).mssre.median()
tr = load_traces("results/E1_smoke/results/" + ok.traces_file.iloc[0], ok.run_id.iloc[0])
```

Or from the command line:

```bash
python -m znnbench.runner aggregate results/E1_smoke --out results/E1_smoke/all.parquet
```

Writing `all.parquet` into the sweep folder is safe: it is not listed in `tasks.jsonl`, so later
aggregations ignore it (unless `--all` / `current_only=False` is used).

`status` takes one of these values: `ok`, `diverged` (MSSRE = inf, and `diverged_at` is set),
`skipped` (the reason is in `message`), `timeout`, or `error` (the traceback is in `message`).

## Memory and time guidance

* Each run allocates two arrays of `(T/τ/record_every + 1) × B·G` floats, for the residual and
  solution-error traces. For example, T = 10 s, τ = 1e-4 and B·G = 64 at `record_every: 1` take
  about 100 MB. For small τ, set `record_every: 10` or more: `simulate` advances every step but
  records only every k-th. Chattering and t_ε are then computed on the recorded samples.
* Parquet results are small: about 1 KB per trajectory row.
* The cost of one spec scales roughly with T/τ × (1 + B·G/16) × (1 + `timing_repeats`). Python
  overhead per step dominates, so `batch_gains: true` is the cheapest way to scan many h or γ values,
  for example for stability maps.
* **Timing (E7).** With `timing_repeats: 0`, `wall` is the wall time of the metric run. That run
  computes the residual and ‖x − x*‖ at every recorded step, which costs about 35 % extra at n = 32
  (measured). With `timing_repeats: R ≥ 1`, R extra *lean* runs record only t_0 and t_N and skip
  x* and the peak-ẋ probe. `wall` and `wall_min` are then the median and minimum over the lean runs
  (`timing_mode = "lean"`), and `wall_recorded` keeps the metric run's time. Use `timing_repeats: 5`
  for the cost–accuracy experiment.
* Never launch large sweeps from tests or in-process. Always use the pool.
