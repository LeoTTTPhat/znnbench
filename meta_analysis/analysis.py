#!/usr/bin/env python3
"""Aggregate the coded ZNN reporting meta-analysis (reproducible; no network access).

Inputs : coded_<k>.csv (primary codes + reserves), coded_<k>_double.csv (independent second codes),
         adjudication.csv (optional: id,item,final,reason for disagreements in the double-coded subsample),
         sample.csv, assignments.json, frame/pool.json (stratum sizes).
Outputs: coded_all.csv, reliability.csv, disagreements.csv, summary.json, summary.md and LaTeX tables in
         ../results/tables/meta_*.tex .

Analysis set: papers whose full text was read (full_text_accessed = yes), i.e. the sampled papers plus
the reserves that replaced inaccessible ones.  Shares are reported with Wilson 95% intervals; counts
with median and IQR.  Stratum-weighted shares (weights = eligible papers per year / coded papers per
year) are reported as a sensitivity check.  Inter-rater reliability on the double-coded subsample:
percent agreement and Cohen's kappa per item (counts: agreement within +-1 and exact).
"""
from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
TAB = HERE.parent / "results" / "tables"
ITEMS = ["R1_n_problems", "R2_max_dim", "R3_sim_method", "R4_step_reported", "R5_discrete_analysis",
         "R6_gain_matched", "R7_n_baselines", "R8_baselines_rerun", "R9_noise_types",
         "R10_time_bound_checked", "R11_code_available", "R12_physical_experiment"]
COUNTS = {"R1_n_problems", "R2_max_dim", "R7_n_baselines"}
R3_CODES = ["ode45/adaptive-ODE", "fixed-step-euler", "zead/higher-order-formula", "simulink", "hardware",
            "not-stated"]


# ----------------------------------------------------------------------------- normalisation
def norm_bin(v):
    v = (v or "").strip().lower()
    if v.startswith("yes") or v in ("y", "true", "1"):
        return "yes"
    if v.startswith("no") or v in ("n", "false", "0"):
        return "no"
    if v.startswith("na") or v in ("n/a", "not applicable"):
        return "NA"
    return "unclear"


def norm_r3(v):
    s = (v or "").strip().lower()
    if not s:
        return "unclear"
    if "ode" in s or "adaptive" in s or "runge" in s and "fixed" not in s:
        return "ode45/adaptive-ODE"
    if "zead" in s or "higher" in s or "taylor" in s or "look-ahead" in s or "instant" in s:
        return "zead/higher-order-formula"
    if "euler" in s:
        return "fixed-step-euler"
    if "simulink" in s:
        return "simulink"
    if "hardware" in s or "fpga" in s or "circuit" in s:
        return "hardware"
    if "not" in s or "unclear" in s:
        return "not-stated"
    return "other"


def norm_r9(v):
    s = (v or "").strip().lower()
    if not s or s.startswith("unclear"):
        return "unclear", []
    if s.startswith("none") or s == "no":
        return "none", []
    kinds = []
    for k, pat in [("constant", r"constant"), ("linear", r"linear|ramp"), ("random", r"random|gaussian|white|bounded"),
                   ("harmonic", r"harmonic|sin|periodic"), ("polynomial", r"quadratic|polynomial|cubic"),
                   ("other", r"impuls|colou?red|measurement|time-varying")]:
        if re.search(pat, s):
            kinds.append(k)
    if s.startswith("multiple") or len(kinds) > 1:
        return "multiple", kinds
    return (kinds[0] if kinds else "other"), kinds


def norm_count(v):
    m = re.search(r"\d+", v or "")
    return int(m.group()) if m else None


def normalise(row):
    out = dict(row)
    for it in ITEMS:
        v = row.get(it, "")
        if it in COUNTS:
            out[it] = norm_count(v)
        elif it == "R3_sim_method":
            out[it] = norm_r3(v)
        elif it == "R9_noise_types":
            out[it], out["R9_kinds"] = norm_r9(v)
        else:
            out[it] = norm_bin(v)
    out["full_text_accessed"] = norm_bin(row.get("full_text_accessed"))
    out["year"] = int(row["year"])
    return out


# ----------------------------------------------------------------------------- statistics
def wilson(k, n, z=1.959964):
    if n == 0:
        return (math.nan, math.nan, math.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (p, max(0.0, c - h), min(1.0, c + h))


def newcombe(k1, n1, k2, n2):
    """Difference p2 - p1 with Newcombe's hybrid score interval."""
    p1, l1, u1 = wilson(k1, n1)
    p2, l2, u2 = wilson(k2, n2)
    d = p2 - p1
    lo = d - math.sqrt((p2 - l2) ** 2 + (u1 - p1) ** 2)
    hi = d + math.sqrt((u2 - p2) ** 2 + (p1 - l1) ** 2)
    return d, lo, hi


def quantiles(xs):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None

    def q(f):
        i = f * (len(xs) - 1)
        lo, hi = math.floor(i), math.ceil(i)
        return xs[lo] + (xs[hi] - xs[lo]) * (i - lo)
    return {"median": q(0.5), "q1": q(0.25), "q3": q(0.75), "n": len(xs), "min": xs[0], "max": xs[-1]}


def cohen_kappa(a, b):
    n = len(a)
    if n == 0:
        return math.nan
    cats = sorted(set(a) | set(b), key=str)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[c] * cb[c] for c in cats) / (n * n)
    return (po - pe) / (1 - pe) if pe < 1 else (1.0 if po == 1 else math.nan)


# ----------------------------------------------------------------------------- load
def load_coded():
    prim, dbl = [], []
    for k in list(range(1, 5)) + ["lead"]:
        p = HERE / f"coded_{k}.csv"
        if p.exists():
            prim += [normalise(r) for r in csv.DictReader(open(p))]
        d = HERE / f"coded_{k}_double.csv"
        if d.exists():
            dbl += [normalise(r) for r in csv.DictReader(open(d))]
    return prim, dbl


def strata_sizes():
    A = json.load(open(HERE / "assignments.json"))
    return {int(y): n for y, n in A["eligible_per_year"].items()}


def apply_adjudication(rows, dbl):
    """Final value per item: primary code, unless adjudication.csv overrides it."""
    adj = {}
    p = HERE / "adjudication.csv"
    if p.exists():
        for r in csv.DictReader(open(p)):
            adj[(r["id"], r["item"])] = r["final"]
    by_id = {r["id"]: r for r in rows}
    for (pid, item), val in adj.items():
        if pid in by_id:
            tmp = normalise({**{it: "" for it in ITEMS}, item: val, "year": by_id[pid]["year"],
                             "full_text_accessed": "yes"})
            by_id[pid][item] = tmp[item]
            if item == "R9_noise_types":
                by_id[pid]["R9_kinds"] = tmp["R9_kinds"]
    return rows, adj


# ----------------------------------------------------------------------------- main analysis
def share_table(rows, item, cats=None, denom_filter=None):
    sub = [r for r in rows if (denom_filter is None or denom_filter(r))]
    vals = [r[item] for r in sub]
    cats = cats or sorted(set(vals), key=str)
    n = len(vals)
    return {c: {"k": vals.count(c), "n": n, "p": wilson(vals.count(c), n)} for c in cats}


def weighted_share(rows, pred, sizes):
    by_y = defaultdict(list)
    for r in rows:
        by_y[r["year"]].append(1.0 if pred(r) else 0.0)
    N = sum(sizes[y] for y in by_y)
    est = sum(sizes[y] / N * (sum(v) / len(v)) for y, v in by_y.items())
    var = sum((sizes[y] / N) ** 2 * (sum(v) / len(v)) * (1 - sum(v) / len(v)) / max(1, len(v) - 1)
              * max(0.0, 1 - len(v) / sizes[y]) for y, v in by_y.items())
    return est, math.sqrt(var)


def claims_time_bound(r):
    s = (r.get("model_class", "") + " " + r.get("notes", "")).lower()
    return r["R10_time_bound_checked"] != "NA" or bool(re.search(r"finite[- ]time|fixed[- ]time|predefined|prescribed", s))


def main():
    prim, dbl = load_coded()
    raw = {r["id"]: dict(r) for r in prim}          # codes before adjudication (for reliability)
    prim, adj = apply_adjudication(prim, dbl)
    sizes = strata_sizes()
    excl = {}
    if (HERE / "exclusions.csv").exists():
        excl = {r["id"]: r["reason"] for r in csv.DictReader(open(HERE / "exclusions.csv"))}
    r6h = {}
    if (HERE / "r6_harmonized.csv").exists():
        r6h = {r["id"]: r["R6_class"] for r in csv.DictReader(open(HERE / "r6_harmonized.csv"))}
    for r in prim:
        r["R6_class"] = r6h.get(r["id"], "")
    analysis = [r for r in prim if r["full_text_accessed"] == "yes" and r["id"] not in excl]
    abstract_only = [r["id"] for r in prim if r["full_text_accessed"] != "yes"]

    # ---------------- reliability
    rel, dis = [], []
    pairs = [(raw[d["id"]], d) for d in dbl if d["id"] in raw]
    for it in ITEMS:
        a = [p[it] for p, _ in pairs]
        b = [q[it] for _, q in pairs]
        if not pairs:
            break
        if it in COUNTS:
            ok = [(x is not None and y is not None) for x, y in zip(a, b)]
            ex = sum(x == y for x, y, o in zip(a, b, ok) if o)
            w1 = sum(abs(x - y) <= 1 for x, y, o in zip(a, b, ok) if o)
            rel.append({"item": it, "n": sum(ok), "exact": ex / max(1, sum(ok)), "within1": w1 / max(1, sum(ok)),
                        "kappa": ""})
        else:
            po = sum(x == y for x, y in zip(a, b)) / len(a)
            rel.append({"item": it, "n": len(a), "exact": po, "within1": "", "kappa": cohen_kappa(a, b)})
        for (p, q) in pairs:
            if p[it] != q[it] and not (it in COUNTS and p[it] is not None and q[it] is not None
                                        and abs(p[it] - q[it]) <= 1):
                dis.append({"id": p["id"], "item": it, "primary": p[it], "second": q[it],
                            "ev_primary": p.get("ev_" + it.split("_")[0], ""),
                            "ev_second": q.get("ev_" + it.split("_")[0], ""),
                            "notes_primary": p.get("notes", "")[:300], "notes_second": q.get("notes", "")[:300],
                            "adjudicated": adj.get((p["id"], it), "")})

    # ---------------- shares
    S = {}
    n = len(analysis)
    S["n_analysis"] = n
    S["n_abstract_only"] = len(abstract_only)
    S["excluded_ineligible"] = excl
    S["abstract_only_ids"] = abstract_only
    S["n_reserves_used"] = sum(1 for r in analysis if r["id"].startswith("R"))
    S["per_year"] = dict(sorted(Counter(r["year"] for r in analysis).items()))
    S["R3"] = share_table(analysis, "R3_sim_method", R3_CODES + ["other", "unclear"])
    for it in ["R4_step_reported", "R5_discrete_analysis", "R11_code_available", "R12_physical_experiment"]:
        S[it] = share_table(analysis, it, ["yes", "no", "unclear"])
    comp = lambda r: (r["R7_n_baselines"] or 0) > 0  # noqa: E731
    S["R6_gain_matched|compares"] = share_table(analysis, "R6_gain_matched", ["yes", "no", "NA", "unclear"], comp)
    gain_cmp = [r for r in analysis if r["R6_class"] and r["R6_class"] != "NA"]
    cls = ["matched", "nominal", "nominal_claim", "different", "not_reported"]
    S["R6_harmonized|gain_baselines"] = {c: {"k": sum(r["R6_class"] == c for r in gain_cmp), "n": len(gain_cmp),
                                             "p": wilson(sum(r["R6_class"] == c for r in gain_cmp), len(gain_cmp))}
                                         for c in cls}
    k_any = sum(r["R6_class"] in ("nominal", "nominal_claim") for r in gain_cmp)
    S["R6_harmonized|gain_baselines"]["nominal_any"] = {"k": k_any, "n": len(gain_cmp), "p": wilson(k_any, len(gain_cmp))}
    S["R6_harmonized_missing"] = [r["id"] for r in analysis if comp(r) and not r["R6_class"]]
    S["R8_baselines_rerun|compares"] = share_table(analysis, "R8_baselines_rerun", ["yes", "no", "NA", "unclear"], comp)
    S["R9"] = share_table(analysis, "R9_noise_types",
                          ["none", "constant", "linear", "random", "harmonic", "polynomial", "multiple", "other", "unclear"])
    kinds = Counter(k for r in analysis for k in r.get("R9_kinds", []))
    S["R9_kinds_any"] = {k: {"k": c, "n": n, "p": wilson(c, n)} for k, c in kinds.items()}
    S["R10|time_claim"] = share_table(analysis, "R10_time_bound_checked", ["yes", "no", "NA", "unclear"],
                                      lambda r: r["R10_time_bound_checked"] != "NA")
    for it in COUNTS:
        S[it] = quantiles([r[it] for r in analysis])

    # ---------------- derived indicators
    disc = lambda r: r["R3_sim_method"] in ("fixed-step-euler", "zead/higher-order-formula", "hardware")  # noqa: E731
    S["step_reported|ode45"] = share_table(analysis, "R4_step_reported", ["yes", "no", "unclear"],
                                           lambda r: r["R3_sim_method"] == "ode45/adaptive-ODE")
    S["step_reported|discrete"] = share_table(analysis, "R4_step_reported", ["yes", "no", "unclear"], disc)
    S["discrete_analysis|discrete"] = share_table(analysis, "R5_discrete_analysis", ["yes", "no", "unclear"], disc)
    full = lambda r: (r["R4_step_reported"] == "yes" and r["R3_sim_method"] != "not-stated"  # noqa: E731
                      and r["R11_code_available"] == "yes")
    k_full = sum(full(r) for r in analysis)
    S["reproducible_minimum"] = {"k": k_full, "n": n, "p": wilson(k_full, n),
                                 "definition": "solver/scheme stated AND step or tolerance stated AND code released"}

    # ---------------- trends
    early = [r for r in analysis if r["year"] <= 2022]
    late = [r for r in analysis if r["year"] >= 2023]
    T = {}
    for name, pred in [("R4 step reported", lambda r: r["R4_step_reported"] == "yes"),
                       ("R3 solver/scheme stated", lambda r: r["R3_sim_method"] != "not-stated"),
                       ("R3 discrete scheme", disc),
                       ("R5 discrete analysis", lambda r: r["R5_discrete_analysis"] == "yes"),
                       ("R11 code released", lambda r: r["R11_code_available"] == "yes"),
                       ("R12 physical experiment", lambda r: r["R12_physical_experiment"] == "yes")]:
        k1, k2 = sum(map(pred, early)), sum(map(pred, late))
        T[name] = {"2020-2022": wilson(k1, len(early)) + (k1, len(early)),
                   "2023-2026": wilson(k2, len(late)) + (k2, len(late)),
                   "diff": newcombe(k1, len(early), k2, len(late))}
        w, se = weighted_share(analysis, pred, sizes)
        T[name]["weighted_all"] = (w, se)
    S["trends"] = T
    S["reliability"] = rel
    S["n_double_pairs"] = len(pairs)
    S["n_disagreements"] = len(dis)
    S["n_adjudicated"] = len(adj)

    # ---------------- write
    cols = ["id", "replaces", "year", "venue", "title", "doi_or_url", "model_class", "problem_class",
            "full_text_accessed"] + ITEMS + ["R6_class", "R9_kinds", "ev_R3", "ev_R4", "ev_R6", "ev_R11", "step_value",
                                              "solver_detail", "notes", "coder"]
    with open(HERE / "coded_all.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in sorted(prim, key=lambda r: r["id"]):
            w.writerow({**r, "R9_kinds": ";".join(r.get("R9_kinds", []))})
    with open(HERE / "reliability.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["item", "n", "exact", "within1", "kappa"])
        w.writeheader()
        w.writerows(rel)
    with open(HERE / "disagreements.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "item", "primary", "second", "ev_primary", "ev_second",
                                          "notes_primary", "notes_second", "adjudicated"])
        w.writeheader()
        w.writerows(dis)
    json.dump(S, open(HERE / "summary.json", "w"), indent=1, default=str)
    write_md(S)
    write_tex(S)
    print(f"analysis set n = {n} (abstract-only {len(abstract_only)}, reserves used {S['n_reserves_used']}); "
          f"double-coded pairs {len(pairs)}, disagreements {len(dis)}")


def pct(t):
    p, lo, hi = t
    return "--" if math.isnan(p) else f"{100 * p:.0f}% [{100 * lo:.0f}, {100 * hi:.0f}]"


def write_md(S):
    L = [f"# Meta-analysis summary (n = {S['n_analysis']} full-text papers)", "",
         f"Abstract-only (excluded): {S['n_abstract_only']}; reserves used: {S['n_reserves_used']}; "
         f"per year: {S['per_year']}", ""]
    L += ["## Shares (Wilson 95% CI)", ""]
    for key in ["R3", "R4_step_reported", "R5_discrete_analysis", "R6_gain_matched|compares",
                "R6_harmonized|gain_baselines",
                "R8_baselines_rerun|compares", "R9", "R10|time_claim", "R11_code_available",
                "R12_physical_experiment", "step_reported|ode45", "step_reported|discrete",
                "discrete_analysis|discrete"]:
        L.append(f"**{key}**: " + "; ".join(f"{c} {v['k']}/{v['n']} {pct(v['p'])}" for c, v in S[key].items() if v["k"]))
    L.append(f"**reproducible minimum** ({S['reproducible_minimum']['definition']}): "
             f"{S['reproducible_minimum']['k']}/{S['reproducible_minimum']['n']} {pct(S['reproducible_minimum']['p'])}")
    L.append(f"**noise kinds tested (any)**: " + "; ".join(f"{k} {v['k']}/{v['n']}" for k, v in S['R9_kinds_any'].items()))
    for it in sorted(COUNTS):
        L.append(f"**{it}**: {S[it]}")
    L += ["", "## Trends (2020-2022 vs 2023-2026; difference with Newcombe 95% CI)", ""]
    for name, t in S["trends"].items():
        a, b, d = t["2020-2022"], t["2023-2026"], t["diff"]
        L.append(f"- {name}: {pct(a[:3])} ({a[3]}/{a[4]}) -> {pct(b[:3])} ({b[3]}/{b[4]}); "
                 f"diff {100 * d[0]:+.0f} pts [{100 * d[1]:+.0f}, {100 * d[2]:+.0f}]; "
                 f"stratum-weighted overall {100 * t['weighted_all'][0]:.0f}% (se {100 * t['weighted_all'][1]:.0f})")
    L += ["", f"## Reliability (double-coded pairs: {S['n_double_pairs']}; disagreements {S['n_disagreements']}, "
              f"adjudicated {S['n_adjudicated']})", ""]
    for r in S["reliability"]:
        kap = "" if r["kappa"] == "" else f", kappa {r['kappa']:.2f}"
        w1 = "" if r["within1"] == "" else f", within +-1 {100 * r['within1']:.0f}%"
        L.append(f"- {r['item']}: n {r['n']}, exact {100 * r['exact']:.0f}%{w1}{kap}")
    (HERE / "summary.md").write_text("\n".join(L) + "\n")


def write_tex(S):
    TAB.mkdir(parents=True, exist_ok=True)

    def row(label, d, cat):
        v = d[cat]
        p, lo, hi = v["p"]
        return f"{label} & {v['k']}/{v['n']} & {100 * p:.0f} & [{100 * lo:.0f}, {100 * hi:.0f}] \\\\"
    rows = [
        r"\begin{tabular}{@{}lrrr@{}}", r"\toprule", r"Item (denominator) & $k/n$ & \% & 95\,\% CI \\", r"\midrule",
        row("R3 simulator: adaptive ODE solver", S["R3"], "ode45/adaptive-ODE"),
        row("R3 simulator: fixed-step Euler-type (or RK) scheme", S["R3"], "fixed-step-euler"),
        row("R3 simulator: higher-order formula", S["R3"], "zead/higher-order-formula"),
        row("R3 simulator: not stated", S["R3"], "not-stated"),
        row("R4 step size or tolerance reported", S["R4_step_reported"], "yes"),
        row("R5 discrete-time model analysed", S["R5_discrete_analysis"], "yes"),
        row("R6 effective gain matched or justified (gain-carrying baselines)", S["R6_harmonized|gain_baselines"], "matched"),
        row("R6 equal nominal gains across different models", S["R6_harmonized|gain_baselines"], "nominal_any"),
        row("R6 \\quad of which the paper calls the comparison fair", S["R6_harmonized|gain_baselines"], "nominal_claim"),
        row("R6 baseline gains not reported", S["R6_harmonized|gain_baselines"], "not_reported"),
        row("R8 baselines re-simulated (papers with baselines)", S["R8_baselines_rerun|compares"], "yes"),
        row("R9 no noise tested", S["R9"], "none"),
        row("R10 time bound checked (time-bound claims)", S["R10|time_claim"], "yes"),
        row("R11 code or data released", S["R11_code_available"], "yes"),
        row("R12 physical experiment", S["R12_physical_experiment"], "yes"),
        r"\bottomrule", r"\end{tabular}"]
    (TAB / "meta_shares.tex").write_text("\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
