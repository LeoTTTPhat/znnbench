"""Tests of the experiment runner (znnbench/runner.py, experiments/run_task.py).

All sweeps here are tiny (n = 4, T <= 1 s, tau = 1e-2, B <= 3): the whole file
runs in a few seconds plus one memguard pool round trip (~10 s) and uses far
less than 1 GB of RAM.
"""
from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from znnbench.simulate import initial_state
import pandas as pd
import pytest

from znnbench import make_discretization, make_model, make_noise, make_problem, simulate
from znnbench import runner as R
from znnbench.core import Context
from znnbench.metrics import summarize

ROOT = Path(__file__).resolve().parents[1]
MEMGUARD = ROOT / "tools" / "memguard.py"


def base_config(**over):
    cfg = {
        "name": "t_runner",
        "problems": [{"key": "linear_system", "kwargs": {"n": 4, "kappa": 10, "seed": 1, "T": 1.0}}],
        "models": ["znn", "gnn"],
        "discretizations": ["euler", "implicit_euler_err"],
        "taus": [1e-2],
        "hs": [0.3],
        "seeds": [0],
        "B": 2,
    }
    cfg.update(over)
    return cfg


# -----------------------------------------------------------------------------
# canonical JSON / ids
# -----------------------------------------------------------------------------
def test_canonical_json_rules():
    assert R.canonical_json({"b": 1.0, "a": [np.float64(0.5), (1, 2)], "c": np.int64(3)}) == \
        '{"a":[0.5,[1,2]],"b":1,"c":3}'
    assert R.canonical_json({"x": True, "y": None}) == '{"x":true,"y":null}'
    assert R.stable_hash({"a": 1, "b": 2}) == R.stable_hash({"b": 2.0, "a": 1.0})
    with pytest.raises(TypeError):
        R.canonical_json({"f": object()})


def test_run_id_golden_value():
    """Guards hash stability across sessions/versions: if this changes, every
    existing result's id changes (bump SPEC_VERSION deliberately instead)."""
    cfg = {"name": "golden", "problems": [{"key": "linear_system", "kwargs": {"n": 4, "kappa": 10, "seed": 1}}],
           "models": ["znn"], "discretizations": ["euler"], "taus": [0.01], "hs": [0.3], "seeds": [0],
           "B": 1}
    (s,) = R.expand(cfg)
    assert s["id"] == GOLDEN_ID, s["id"]
    assert s["id"] == R.run_id(json.loads(json.dumps(s)))      # survives a JSON round trip


GOLDEN_ID = "7beb8140af9f16e6"


def test_expand_counts_unique_and_stable():
    cfg = base_config(
        problems=[{"key": "linear_system", "kwargs": {"n": {"$sweep": [2, 4]}, "kappa": 10, "T": 1.0}}],
        taus=[1e-2, 3e-3], hs=[0.1, 0.3], gammas=[10.0],
        infos=["analytic", {"info": "bd1", "data_noise": 1e-6}],
        noises=[None, {"key": "constant", "kwargs": {"level": 0.1}}],
        seeds=[0, 1])
    specs = R.expand(cfg)
    assert len(specs) == 2 * 2 * 2 * 2 * 3 * 2 * 2 * 2
    ids = [s["id"] for s in specs]
    assert len(set(ids)) == len(ids)
    assert ids == [s["id"] for s in R.expand(copy.deepcopy(cfg))]
    # key order of the config dict and of kwargs does not matter
    rev = {k: cfg[k] for k in reversed(list(cfg))}
    rev["problems"] = [{"kwargs": {"T": 1.0, "kappa": 10, "n": {"$sweep": [2, 4]}}, "key": "linear_system"}]
    assert sorted(ids) == sorted(s["id"] for s in R.expand(rev))
    # 10 vs 10.0, labels, traces and max_wall do not change ids ...
    same = copy.deepcopy(cfg)
    same["problems"][0]["kwargs"]["kappa"] = 10.0
    same["problems"][0]["label"] = "ls n={n}"
    same.update(traces={"points": 16}, max_wall=100.0)
    assert ids == [s["id"] for s in R.expand(same)]
    assert R.expand(same)[0]["problem"]["label"] == "ls n=2"
    # ... but anything that changes the numbers does
    for key, val in [("T", 0.5), ("B", 3), ("record_every", 2), ("metrics", {"frac": 0.5})]:
        other = R.expand({**cfg, key: val})
        assert not set(ids) & {s["id"] for s in other}, key


def test_gain_modes():
    s = R.expand(base_config(models=["znn"], discretizations=["euler"], taus=[1e-2, 1e-3],
                             hs=[0.5], gammas=[20.0]))
    got = sorted((sp["tau"], sp["gain"]["mode"], R._gains_of(sp)[1][0], R._gains_of(sp)[2][0]) for sp in s)
    exp = sorted([(1e-2, "fixed_h", 50.0, 0.5), (1e-3, "fixed_h", 500.0, 0.5),
                  (1e-2, "fixed_gamma", 20.0, 0.2), (1e-3, "fixed_gamma", 20.0, 0.02)])
    for g, e in zip(got, exp):
        assert g[:2] == e[:2] and np.isclose(g[2], e[2]) and np.isclose(g[3], e[3])
    with pytest.raises(ValueError):
        R.expand(base_config(mode="fixed_gamma"))                 # hs given, gammas missing
    with pytest.raises(ValueError):
        R.expand(base_config(hs=None))                            # no gains at all
    with pytest.raises(ValueError):
        R.expand(base_config(hs=[-1.0]))
    with pytest.raises(ValueError):
        R.expand(base_config(taus_typo=[1e-2]))                   # unknown key
    with pytest.raises(ValueError):
        R.expand(base_config(models=["no_such_model"]))           # registry validation
    for bad in ({"T": 0.0}, {"T": -1.0}, {"timing_repeats": -1}, {"chunk_size": 0}, {"div_threshold": 0}):
        with pytest.raises(ValueError):
            R.expand(base_config(**bad))


def test_seed_substitution_sweeps_and_exclude():
    cfg = base_config(problems=[{"key": "linear_system", "kwargs": {"n": 4, "seed": "$seed", "T": 1.0}}],
                      seeds=[3, 4], infos=["analytic", "bd1", "bd2"],
                      exclude=[{"model": "gnn", "info": ["bd1", "bd2"]},
                               {"disc": "implicit_euler_err", "model": {"key": "gnn"}}])
    specs = R.expand(cfg)
    assert all(s["problem"]["kwargs"]["seed"] == s["seed"] for s in specs)
    combos = {(s["model"]["key"], s["disc"]["key"], s["info"]) for s in specs}
    assert ("gnn", "euler", "analytic") in combos
    assert not any(m == "gnn" and (i != "analytic" or d != "euler") for m, d, i in combos)
    assert len(specs) == 2 * (2 * 3 + 1)
    # exclude on derived gain values
    s2 = R.expand(base_config(models=["znn"], discretizations=["euler"], taus=[1e-2, 1e-3],
                              hs=None, gammas=[10.0, 1000.0], exclude=[{"h": 1.0}]))
    assert {round(R._gains_of(s)[2][0], 12) for s in s2} == {0.1, 0.01, 10.0}


def test_yaml_and_py_configs(tmp_path):
    text = """
name: t_runner
problems:
  - key: linear_system
    kwargs: {n: 4, kappa: 10, seed: 1, T: 1.0}
models: [znn, gnn]
discretizations: [euler, implicit_euler_err]
taus: [1e-2]
hs: [0.3]
seeds: [0]
B: 2
"""
    y = R.yaml_loads(text)
    assert isinstance(y["taus"][0], float)                       # PyYAML alone gives the string '1e-2'
    ids = [s["id"] for s in R.expand(base_config())]
    f = tmp_path / "c.yaml"
    f.write_text(text)
    (cy,) = R.load_configs(f)
    assert [s["id"] for s in R.expand(cy)] == ids
    p = tmp_path / "cfg_py.py"
    p.write_text("CONFIGS = [" + repr(base_config()) + ", " + repr(base_config(name='other')) + "]\n")
    c1, c2 = R.load_configs(p)
    assert [s["id"] for s in R.expand(c1)] == ids and c2["name"] == "other"


def _load_memguard():
    spec = importlib.util.spec_from_file_location("memguard_for_tests", MEMGUARD)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_chunk_and_write_tasks_format(tmp_path):
    specs = R.expand(base_config(taus=[1e-2, 3e-3, 1e-3], seeds=[0, 1, 2]))
    assert len(specs) == 2 * 2 * 3 * 3
    for balance in (False, True):
        chunks = R.chunk(specs, 5, balance=balance)
        assert all(1 <= len(c) <= 5 for c in chunks) and len(chunks) == 8
        assert sorted(s["id"] for c in chunks for s in c) == sorted(s["id"] for s in specs)
    costs = [sum(R.cost_estimate(s) for s in c) for c in R.chunk(specs, 5, balance=True)]
    naive = [sum(R.cost_estimate(s) for s in c) for c in R.chunk(specs, 5)]
    assert max(costs) - min(costs) <= max(naive) - min(naive)
    tf = R.write_tasks("t runner/x", R.chunk(specs, 5), tmp_path)
    tasks = _load_memguard().load_tasks(tf) if MEMGUARD.exists() else R.load_tasks(tf)
    assert len(tasks) == 8
    for t in tasks:
        assert set(t) >= {"id", "cmd", "done"}
        assert t["id"].startswith("t_runner_x-") and "/" not in t["id"]
        assert t["cmd"][1:3] == ["experiments/run_task.py", "--chunk"]
        assert Path(t["cmd"][0]).name.startswith("python")
        assert Path(t["cwd"]) == ROOT and (Path(t["cwd"]) / t["cmd"][1]).exists()
        ch = json.loads(Path(t["cmd"][3]).read_text())
        assert ch["result"] == t["done"] and t["done"].endswith(".parquet")
        assert not Path(t["done"]).exists()
        assert t["env"]["OMP_NUM_THREADS"] == "1"
    assert [t["id"] for t in R.load_tasks(R.write_tasks("t runner/x", R.chunk(specs, 5), tmp_path))] == \
        [t["id"] for t in tasks]
    # output-only options keep the run ids but change the task ids (so the chunks re-run)
    for over in ({"traces": {"points": 8}}, {"max_wall": 50.0}, {"timing_repeats": 2},
                 {"models": [{"key": "znn", "label": "ZNN"}, "gnn"]}):
        sp2 = R.expand(base_config(taus=[1e-2, 3e-3, 1e-3], seeds=[0, 1, 2], **over))
        assert [s["id"] for s in sp2] == [s["id"] for s in specs], over
        t2 = R.load_tasks(R.write_tasks("t runner/x", R.chunk(sp2, 5), tmp_path / "o"))
        for c_old, c_new, a, b in zip(R.chunk(specs, 5), R.chunk(sp2, 5), tasks, t2):
            changed = any(R.canonical_json(x) != R.canonical_json(y) for x, y in zip(c_old, c_new))
            assert (a["id"] != b["id"]) == changed, over
            assert changed or "models" in over                 # only the label edit leaves gnn chunks alone


# -----------------------------------------------------------------------------
# execution
# -----------------------------------------------------------------------------
def test_end_to_end_in_process(tmp_path):
    cfg = base_config(hs=[0.3, 3.0], metrics={"extra": ["final_residual", "peak_xdot"]},
                      traces={"points": 11})
    tf = R.build(cfg, tmp_path, chunk_size=4)
    tasks = R.load_tasks(tf)
    assert len(tasks) == 2
    paths = R.run_local(tmp_path, verbose=False)
    assert all(Path(t["done"]).exists() for t in tasks)
    assert not [f for f in (tmp_path / "results").iterdir() if f.name.startswith(".")]   # no tmp left
    df = R.aggregate(tmp_path)
    assert len(df) == 8 * 2 and df["run_id"].nunique() == 8
    assert "spec_json" not in df.columns and "spec_json" in pd.read_parquet(paths[0]).columns
    # statuses: GNN + implicit stepping is skipped; h = 3 explicit Euler diverges
    sk = df[(df.model_key == "gnn") & (df.disc_key == "implicit_euler_err")]
    assert (sk.status == "skipped").all() and sk.mssre.isna().all()
    assert sk.message.str.contains("implicit").all()
    dv = df[(df.model_key == "znn") & (df.disc_key == "euler") & np.isclose(df.h, 3.0)]
    assert (dv.status == "diverged").all() and dv.diverged.all() and np.isinf(dv.mssre).all()
    ok = df[(df.model_key == "znn") & (df.disc_key == "euler") & np.isclose(df.h, 0.3)].sort_values("traj")
    assert (ok.status == "ok").all() and (~ok.diverged).all()
    assert np.allclose(ok.gamma, 30.0) and (ok.wall_per_step > 0).all()
    # metrics equal a direct simulate() call with the same parameters
    P = make_problem("linear_system", n=4, kappa=10, seed=1, T=1.0)
    sim = simulate(P, make_model("znn"), make_discretization("euler"), tau=1e-2, B=2, gamma=30.0, seed=0)
    ref = summarize(sim)
    for col in ("mssre", "mean_ssre", "chatter", "ss_sol_err", "t_eps_0.001"):
        assert np.allclose(ok[col].to_numpy(), ref[col], rtol=1e-12, equal_nan=True), col
    assert np.allclose(ok.final_residual, sim["residual"][-1], rtol=1e-12)
    # peak ||x_dot|| is at least the initial ||x_dot(0)||
    x0 = initial_state(P, make_model("znn"), 0, 2)   # simulate's own draw
    d = P.data(0.0)
    ctx = Context(problem=P, t=0.0, tau=1e-2, k=0, x=x0, z=np.zeros((2, 0)), d=d, E=P.E(x0, d),
                  dEdt=P.dEdt(x0, 0.0), gamma=np.full((2, 1), 30.0))
    xd0 = np.linalg.norm(make_model("znn").rhs(ctx)[0], axis=1)
    assert (ok.peak_xdot.to_numpy() >= xd0 * (1 - 1e-12)).all()
    # traces
    row = ok.iloc[0]
    tr = R.load_traces(tmp_path / "results" / row.traces_file, row.run_id)
    assert tr["residual"].shape == (11, 2) and tr["t"][0] == 0.0 and np.isclose(tr["t"][-1], 1.0)
    assert np.allclose(tr["residual"][-1], sim["residual"][-1], rtol=1e-6)
    # idempotent: done chunks are not re-run
    m0 = [Path(t["done"]).stat().st_mtime_ns for t in tasks]
    R.run_local(tmp_path, verbose=False)
    assert m0 == [Path(t["done"]).stat().st_mtime_ns for t in tasks]
    assert R.status(tmp_path) == {"tasks": 2, "done": 2, "pending": 0}


def _hand_euler_znn(P, tau, gamma, x0, T):
    """Linear ZNN + forward Euler written out by hand (no simulate, no LookAhead):
    x_{k+1} = x_k + tau A^{-1}(-gamma (A x_k - b) - (A_dot x_k - b_dot))."""
    N = int(round(T / tau))
    x, res, err, peak = x0.copy(), [], [], np.zeros(len(x0))
    for k in range(N + 1):
        t = k * tau
        A, b = P.A(t), P.b(t)
        E = x @ A.T - b
        res.append(np.linalg.norm(E, axis=1))
        err.append(np.linalg.norm(x - np.linalg.solve(A, b), axis=1))
        if k == N:
            break
        xdot = np.linalg.solve(A, (-gamma * E - (x @ P.A.dot(t).T - P.b.dot(t))).T).T
        peak = np.maximum(peak, np.linalg.norm(xdot, axis=1))
        x = x + tau * xdot
    return np.arange(N + 1) * tau, np.array(res), np.array(err), peak


def test_metrics_match_hand_written_euler():
    """Independent reference: every recorded metric (and the gain mapping of both
    modes) equals a hand-written ZNN-Euler loop evaluated with the formulas of
    metrics.py written out again here."""
    cfg = base_config(problems=[{"key": "linear_system", "kwargs": {"n": 4, "kappa": 10, "seed": 3, "T": 1.0}}],
                      models=["znn"], discretizations=["euler"], taus=[1e-2, 5e-3], hs=[0.2], gammas=[7.0],
                      B=3, seeds=[11], metrics={"extra": ["peak_xdot", "final_residual", "max_residual",
                                                          "initial_residual", "final_sol_err"]})
    P = make_problem("linear_system", n=4, kappa=10, seed=3, T=1.0)
    x0 = initial_state(P, make_model("znn"), 11, 3)   # simulate's own draw
    specs = R.expand(cfg)
    assert len(specs) == 4
    for s in specs:
        df = pd.DataFrame(R.run_spec(s)[0])
        gamma = 0.2 / s["tau"] if s["gain"]["mode"] == "fixed_h" else 7.0
        assert np.allclose(df.gamma, gamma) and np.allclose(df.h, gamma * s["tau"])
        t, res, err, peak = _hand_euler_znn(P, s["tau"], gamma, x0, 1.0)
        w = t >= t[-1] - 0.25 * (t[-1] - t[0])
        exp = {"mssre": res[w].max(0), "mean_ssre": res[w].mean(0), "ss_sol_err": err[w].max(0),
               "peak_xdot": peak, "final_residual": res[-1], "initial_residual": res[0],
               "max_residual": res.max(0), "final_sol_err": err[-1],
               "chatter": np.abs(np.diff(res[w], axis=0)).mean(0) / res[w].mean(0)}
        for eps in (1e-3, 1e-6):
            last_bad = [np.where(res[:, b] >= eps)[0] for b in range(3)]
            exp[f"t_eps_{eps:g}"] = [t[0] if len(lb) == 0 else (t[lb[-1] + 1] if lb[-1] < len(t) - 1 else np.inf)
                                     for lb in last_bad]
        for col, v in exp.items():
            assert np.allclose(df[col].to_numpy(float), v, rtol=1e-9), (s["gain"], col, df[col].tolist(), v)
        assert (df.status == "ok").all()


class _NeedsCapability:
    """Test-only model proxy that declares a capability no problem provides."""

    def __init__(self, inner, cap):
        self._inner, self.requires = inner, {cap}

    def __getattr__(self, name):
        if name == "_inner":
            raise AttributeError(name)
        return getattr(self._inner, name)


def _scale_gain_wrapper(model, factor=1.0):
    """Test-only wrapper resolved by import path: multiplies the design gain."""
    class Scaled:
        def __getattr__(self, name):
            return getattr(model, name)

        def rhs(self, ctx):
            ctx.gamma = ctx.gamma * factor
            return model.rhs(ctx)
    return Scaled()


if "_test_needs_cap" not in R.WRAPPERS:
    R.register_wrapper("_test_needs_cap", target="model")(
        lambda m, cap="_no_such_capability": _NeedsCapability(m, cap))


def test_skip_incompatible_and_wrappers(tmp_path):
    cfg = base_config(models=["znn"], discretizations=["euler"],
                      wrappers=[None, "_test_needs_cap",
                                {"key": f"{__name__}:_scale_gain_wrapper", "target": "model",
                                 "kwargs": {"factor": 2.0}}])
    specs = R.expand(cfg)
    assert len(specs) == 3
    rows = [r for s in specs for r in R.run_spec(s)[0]]
    df = pd.DataFrame(rows)
    plain = df[df.wrapper_keys == "none"]
    skip = df[df.wrapper_keys == "_test_needs_cap"]
    scaled = df[df.wrapper_keys.str.contains("_scale_gain_wrapper")]
    assert (plain.status == "ok").all()
    assert (skip.status == "skipped").all() and skip.message.str.contains("_no_such_capability").all()
    # gain x2 at h = 0.3 behaves exactly like h = 0.6
    (s6,) = R.expand(base_config(models=["znn"], discretizations=["euler"], hs=[0.6]))
    ref = pd.DataFrame(R.run_spec(s6)[0])
    assert np.allclose(scaled.mssre.to_numpy(), ref.mssre.to_numpy(), rtol=1e-12)
    with pytest.raises(ValueError):
        R.expand(base_config(wrappers=["_no_such_wrapper"]))
    # "$sweep" in wrapper kwargs expands (instead of passing the dict through)
    sw = R.expand(base_config(models=["znn"], discretizations=["euler"],
                              wrappers=[None, [{"key": "_test_needs_cap", "kwargs": {"cap": {"$sweep": ["a", "b"]}}},
                                               {"key": f"{__name__}:_scale_gain_wrapper", "target": "model",
                                                "kwargs": {"factor": {"$sweep": [1.0, 2.0, 3.0]}}}]]))
    chains = sorted((s["wrappers"][0]["kwargs"]["cap"], s["wrappers"][1]["kwargs"]["factor"])
                    for s in sw if s["wrappers"])
    assert len(sw) == 1 + 6 and chains == [(c, f) for c in "ab" for f in (1, 2, 3)]


def test_real_capability_skip():
    """A registered model with `requires` (sibling area) is skipped, not crashed,
    on a problem lacking the capability and runs where the capability exists."""
    from znnbench.registry import available
    av = available()
    if "znn_explicit_inverse" not in av["models"] or "matrix_inversion" not in av["problems"]:
        pytest.skip("znn_explicit_inverse / matrix_inversion not registered")
    cfg = base_config(models=["znn_explicit_inverse"], discretizations=["euler"], B=1,
                      problems=[{"key": "linear_system", "kwargs": {"n": 4, "T": 0.5}},
                                {"key": "matrix_inversion", "kwargs": {"n": 3, "T": 0.5}}])
    df = pd.DataFrame([r for s in R.expand(cfg) for r in R.run_spec(s)[0]])
    st = dict(zip(df.problem_key, df.status))
    assert st["linear_system"] == "skipped", df.message.tolist()
    assert st["matrix_inversion"] in ("ok", "diverged"), df.message.tolist()
    assert "matrix_inverse" in df[df.problem_key == "linear_system"].message.iloc[0]


def test_sibling_wrappers_module_resolution():
    """Classes of znnbench/wrappers.py (owned by the noise/info area) resolve by
    name, with the target inferred from their base class; "$tau"/"$seed" fill in."""
    wmod = pytest.importorskip("znnbench.wrappers")
    if not hasattr(wmod, "QuantizedDiscretizer"):
        pytest.skip("znnbench.wrappers.QuantizedDiscretizer not available")
    assert R.wrapper_target({"key": "QuantizedDiscretizer"}) == "disc"
    cfg = base_config(models=["znn"], discretizations=["euler"], seeds=[2],
                      wrappers=[None, {"key": "QuantizedDiscretizer", "kwargs": {"bits": 12, "frac_bits": 6}}],
                      problems=[{"key": "linear_system", "kwargs": {"n": 4, "seed": "$seed", "T": 1.0}}])
    specs = R.expand(cfg)
    assert [s["problem"]["kwargs"]["seed"] for s in specs] == [2, 2]
    df = pd.DataFrame([r for s in specs for r in R.run_spec(s)[0]])
    assert set(df.status) == {"ok"}, df.message.tolist()
    q = df[df.wrapper_keys == "QuantizedDiscretizer"]
    assert q.disc_name.str.contains("+q12", regex=False).all()
    # a coarse 6-fraction-bit state grid floors the residual above the float64 run
    assert (q.mssre.to_numpy() > df[df.wrapper_keys == "none"].mssre.to_numpy()).all()
    (st,) = R.expand(base_config(models=["znn"], discretizations=["euler"],
                                 wrappers=[{"key": "_test_needs_cap", "kwargs": {"cap": "$tau"}}]))
    assert st["wrappers"][0]["kwargs"]["cap"] == 0.01


def test_errors_timeouts_recorded_and_chunk_survives(tmp_path):
    good = R.expand(base_config(models=["znn"], discretizations=["euler"]))
    bad = R.expand(base_config(models=["znn"], discretizations=["euler"],
                               problems=[{"key": "linear_system", "kwargs": {"n": 4, "bogus": 1}}]))
    slow = R.expand(base_config(models=["znn"], discretizations=["euler"], max_wall=1e-9, seeds=[5]))
    tf = R.write_tasks("t_err", [bad + slow + good], tmp_path)
    (task,) = R.load_tasks(tf)
    R.execute_chunk(task["cmd"][3], verbose=False)
    df = R.aggregate(tmp_path)
    e = df[df.run_id == bad[0]["id"]]
    t = df[df.run_id == slow[0]["id"]]
    g = df[df.run_id == good[0]["id"]]
    assert len(e) == len(t) == len(g) == 2
    assert (e.status == "error").all() and e.message.str.contains("TypeError").all()
    assert (t.status == "timeout").all() and t.message.str.contains("TimeoutError").all()
    assert (g.status == "ok").all() and np.isfinite(g.mssre).all()


def test_done_file_only_after_complete_chunk(tmp_path, monkeypatch):
    specs = R.expand(base_config(models=["znn"], discretizations=["euler"], seeds=[0, 1, 2]))
    (task,) = R.load_tasks(R.write_tasks("t_atomic", [specs], tmp_path))
    real, calls = R.run_spec, []

    def flaky(spec, **kw):
        calls.append(spec["id"])
        if len(calls) == 2:
            raise KeyboardInterrupt("simulated kill")
        return real(spec, **kw)

    monkeypatch.setattr(R, "run_spec", flaky)
    with pytest.raises(KeyboardInterrupt):
        R.execute_chunk(task["cmd"][3], verbose=False)
    assert not Path(task["done"]).exists()
    assert not list((tmp_path / "results").iterdir())
    monkeypatch.setattr(R, "run_spec", real)
    R.execute_chunk(task["cmd"][3], verbose=False)
    assert Path(task["done"]).exists() and len(pd.read_parquet(task["done"])) == 6


def test_batch_gains_equals_unbatched():
    cfg = base_config(models=["znn"], discretizations=["euler", "taylor_zhang"], hs=[0.1, 0.3, 0.9], B=3,
                      noises=[{"key": "constant", "kwargs": {"level": 0.05}}])
    sep = pd.DataFrame([r for s in R.expand(cfg) for r in R.run_spec(s)[0]])
    bat_specs = R.expand({**cfg, "batch_gains": True})
    assert len(bat_specs) == 2 and all(len(s["gain"]["values"]) == 3 for s in bat_specs)
    bat = pd.DataFrame([r for s in bat_specs for r in R.run_spec(s)[0]])
    assert len(bat) == len(sep) == 2 * 3 * 3
    key = ["disc_key", "h", "x0_index"]
    a = sep.sort_values(key).reset_index(drop=True)
    b = bat.sort_values(key).reset_index(drop=True)
    assert np.allclose(a.h, b.h) and np.allclose(a.gamma, b.gamma)
    for col in ("mssre", "mean_ssre", "ss_sol_err", "t_eps_0.001"):
        assert np.allclose(a[col], b[col], rtol=1e-10, equal_nan=True), col


def test_aggregate_dedupe_columns_and_empty(tmp_path):
    assert R.aggregate(tmp_path).empty
    specs = R.expand(base_config(models=["znn"], discretizations=["euler"]))
    (task,) = R.load_tasks(R.write_tasks("t_agg", [specs], tmp_path))
    p = R.execute_chunk(task["cmd"][3], verbose=False)
    (tmp_path / "results" / "copy.parquet").write_bytes(Path(p).read_bytes())
    assert len(R.aggregate(tmp_path, dedupe=False, current_only=False)) == 4
    assert len(R.aggregate(tmp_path, dedupe=False)) == 2          # only the done files of tasks.jsonl
    df = R.aggregate(tmp_path, current_only=False)
    assert len(df) == 2
    # a chunk holding only skipped runs (no metric columns) concatenates cleanly
    (sk,) = R.expand(base_config(models=["gnn"], discretizations=["implicit_euler_err"]))
    (t2,) = R.load_tasks(R.write_tasks("t_agg_skip", [[sk]], tmp_path))
    R.execute_chunk(t2["cmd"][3], verbose=False)
    both = R.aggregate(tmp_path, current_only=False)
    assert len(both) == 4 and str(both.diverged.dtype) == "boolean"
    assert both.diverged.isna().sum() == 2 and (~both.diverged.dropna()).all()
    sub = R.aggregate(tmp_path, columns=["tau", "mssre", "problem_kw_n"], current_only=False)
    assert list(sub.columns) == ["tau", "mssre", "problem_kw_n"] and len(sub) == 4
    assert (sub.problem_kw_n == 4).all()


def test_rebuild_reruns_changed_chunks_and_aggregate_ignores_stale(tmp_path):
    """Regression: enabling traces after a finished run must produce traces (the
    old task ids hashed only run ids, so the chunks counted as done), and results
    of an earlier build (a removed model) or an all.parquet written into the
    folder must not leak into aggregate()."""
    cfg = base_config(discretizations=["euler"])                    # znn + gnn
    R.build(cfg, tmp_path, chunk_size=1)
    R.run_local(tmp_path, verbose=False)
    first = R.aggregate(tmp_path)
    assert set(first.model_key) == {"znn", "gnn"} and "traces_file" not in first.columns
    first.to_parquet(tmp_path / "all.parquet")                       # user output inside the folder
    R.build({**cfg, "models": ["znn"], "traces": {"points": 5}}, tmp_path, chunk_size=1)
    assert R.status(tmp_path) == {"tasks": 1, "done": 0, "pending": 1}
    assert R.aggregate(tmp_path).empty                               # stale chunks are not current
    # ... but still on disk: 2 old chunks + all.parquet
    assert len(R.aggregate(tmp_path, current_only=False, dedupe=False)) == 2 * len(first)
    R.run_local(tmp_path, verbose=False)
    df = R.aggregate(tmp_path)
    assert set(df.model_key) == {"znn"} and len(df) == 2
    tr = R.load_traces(tmp_path / "results" / df.traces_file.iloc[0], df.run_id.iloc[0])
    assert tr["residual"].shape == (5, 2)
    # the run id is unchanged, so the numbers equal the first build's
    old = first[first.model_key == "znn"].sort_values("traj")
    assert list(old.run_id) == list(df.sort_values("traj").run_id)
    assert np.allclose(old.mssre.to_numpy(), df.sort_values("traj").mssre.to_numpy(), rtol=0)


# -----------------------------------------------------------------------------
# through tools/memguard.py pool (subprocesses)
# -----------------------------------------------------------------------------
@pytest.mark.skipif(not MEMGUARD.exists(), reason="tools/memguard.py not found")
def test_memguard_pool_end_to_end(tmp_path):
    import psutil
    if psutil.virtual_memory().available < 60 * 2 ** 30:
        pytest.skip("memguard pool refuses to launch with < 48 GB available")
    cfg = base_config(name="pytest_runner_pool")
    tf = R.build(cfg, tmp_path / "pool", chunk_size=2)
    tasks = R.load_tasks(tf)
    assert len(tasks) == 2
    logs = [ROOT / "logs" / "tasks" / f"{t['id']}.log" for t in tasks]
    try:
        proc = subprocess.run([sys.executable, str(MEMGUARD), "pool", "--tasks", str(tf), "--workers", "2",
                               "--max-gb", "50"], capture_output=True, text=True, timeout=300,
                              cwd=str(ROOT))
        detail = proc.stdout[-2000:] + proc.stderr[-2000:] + "".join(
            lg.read_text()[-2000:] for lg in logs if lg.exists())
        assert proc.returncode == 0, detail
        assert all(Path(t["done"]).exists() for t in tasks), detail
    finally:
        for lg in logs:
            if lg.exists():
                lg.unlink()
    pool = R.aggregate(tmp_path / "pool").sort_values(["run_id", "traj"]).reset_index(drop=True)
    assert len(pool) == 4 * 2
    assert "skipped" in set(pool.status) and set(pool.status) <= {"ok", "skipped", "diverged"}
    assert (pool[(pool.model_key == "gnn") & (pool.disc_key == "implicit_euler_err")].status == "skipped").all()
    # identical to in-process execution of the same config
    R.build(cfg, tmp_path / "local", chunk_size=2)
    R.run_local(tmp_path / "local", verbose=False)
    loc = R.aggregate(tmp_path / "local").sort_values(["run_id", "traj"]).reset_index(drop=True)
    assert list(loc.run_id) == list(pool.run_id)
    assert np.allclose(loc.mssre.to_numpy(float), pool.mssre.to_numpy(float), rtol=1e-12, equal_nan=True)
    # running the pool again is a no-op (all done files exist)
    again = subprocess.run([sys.executable, str(MEMGUARD), "pool", "--tasks", str(tf), "--workers", "2",
                            "--max-gb", "50"], capture_output=True, text=True, timeout=120, cwd=str(ROOT))
    assert again.returncode == 0 and "2 already done" in again.stdout


def test_timing_repeats_and_record_every():
    extra = {"extra": ["peak_xdot"]}
    (s,) = R.expand(base_config(models=["znn"], discretizations=["euler"], timing_repeats=3, record_every=5,
                                metrics=extra))
    (s1,) = R.expand(base_config(models=["znn"], discretizations=["euler"], record_every=5, metrics=extra))
    assert s["id"] == s1["id"]                       # timing runs do not change the numbers
    rows = pd.DataFrame(R.run_spec(s)[0])
    ref = pd.DataFrame(R.run_spec(s1)[0])
    assert np.allclose(rows.mssre, ref.mssre, rtol=0) and np.allclose(rows.peak_xdot, ref.peak_xdot, rtol=0)
    assert (rows.steps == 100).all() and (rows.record_every == 5).all()
    # 3 lean timing runs (no recording overhead) vs. the recorded metric run
    assert (rows.timing_mode == "lean").all() and (rows.timing_runs == 3).all()
    assert (rows.wall_min <= rows.wall).all() and (rows.wall_recorded > 0).all()
    assert np.allclose(rows.wall_per_step, rows.wall / 100)
    assert (ref.timing_mode == "recorded").all() and (ref.timing_runs == 1).all()
    assert np.allclose(ref.wall, ref.wall_recorded)
    # the lean-run premise: recording neither feeds back nor draws random numbers,
    # so a lean run (record t_0, t_N only) ends in the same state, even with
    # stochastic design-formula noise and data noise
    P = make_problem("linear_system", n=4, kappa=10, seed=1, T=1.0)
    kw = dict(tau=1e-2, B=2, gamma=30.0, seed=4, noise=make_noise("gaussian", std=0.1), info="bd2",
              data_noise=1e-3)
    full = simulate(P, make_model("znn"), make_discretization("taylor_zhang"), **kw)
    lean = simulate(P, make_model("znn"), make_discretization("taylor_zhang"), record_every=100,
                    record_solution_error=False, **kw)
    assert lean["residual"].shape == (2, 2)
    assert np.array_equal(lean["residual"][-1], full["residual"][-1])


def test_command_line_build_local_aggregate(tmp_path):
    cfg = tmp_path / "cli_cfg.yaml"
    cfg.write_text("name: cli\nproblems: [{key: linear_system, kwargs: {n: 4, T: 0.5}}]\n"
                   "models: [znn]\ndiscretizations: [euler, taylor_zhang]\ntaus: [1e-2]\nhs: [0.3]\nB: 1\n")
    py = [sys.executable]
    r1 = subprocess.run(py + ["-m", "znnbench.runner", "build", str(cfg), "--out", str(tmp_path / "a"),
                              "--chunk-size", "1"], capture_output=True, text=True, cwd=str(ROOT), timeout=120)
    r2 = subprocess.run(py + ["experiments/run_task.py", "--build", str(cfg), "--out", str(tmp_path / "b"),
                              "--chunk-size", "1"], capture_output=True, text=True, cwd=str(ROOT), timeout=120)
    assert r1.returncode == 0 and r2.returncode == 0, r1.stderr + r2.stderr
    ta, tb = R.load_tasks(tmp_path / "a" / "tasks.jsonl"), R.load_tasks(tmp_path / "b" / "tasks.jsonl")
    assert [t["id"] for t in ta] == [t["id"] for t in tb] and len(ta) == 2
    r3 = subprocess.run(py + ["experiments/run_task.py", "--chunk", ta[0]["cmd"][3]], capture_output=True,
                        text=True, cwd=str(ROOT), timeout=120)
    assert r3.returncode == 0 and Path(ta[0]["done"]).exists(), r3.stdout + r3.stderr
    r4 = subprocess.run(py + ["-m", "znnbench.runner", "local", str(tmp_path / "a")], capture_output=True,
                        text=True, cwd=str(ROOT), timeout=120)
    assert r4.returncode == 0 and "already done" in r4.stdout, r4.stderr
    out = tmp_path / "all.parquet"
    r5 = subprocess.run(py + ["-m", "znnbench.runner", "aggregate", str(tmp_path / "a"), "--out", str(out)],
                        capture_output=True, text=True, cwd=str(ROOT), timeout=120)
    assert r5.returncode == 0, r5.stderr
    df = pd.read_parquet(out)
    assert len(df) == 2 and set(df.disc_key) == {"euler", "taylor_zhang"} and (df.status == "ok").all()
