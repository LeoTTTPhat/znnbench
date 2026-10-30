#!/usr/bin/env python3
"""Fetch a sampled paper's full text to fulltext/<id>.txt and print the rubric-relevant passages.

Usage:
  fetch_text.py M001            # try fulltext_url, then alt_urls, then https://doi.org/<doi>
  fetch_text.py M001 --url URL  # try a specific URL first (e.g. an arXiv or repository copy)
  fetch_text.py M001 --from-file page.txt   # register text saved from a browser (get_page_text)
  fetch_text.py M001 --show     # only print the passages of an already fetched text

PDF -> pdftotext -layout; HTML -> macOS textutil (falls back to a tag stripper).  A text counts as a
full text if it has > 15000 characters and mentions a conclusion and references.  The passage filter
prints numbered lines that match the rubric keywords (step size, solver, gains, baselines, noise,
code, hardware, dimensions), with one line of context, capped at --max-lines.

No personal identifier is sent: the User-Agent is a generic desktop browser string.
"""
from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "fulltext"
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/129.0 Safari/537.36")
KW = re.compile(
    r"step[- ]?size|sampling (?:gap|period|interval|time|rate)|\bτ\b|\btau\b|\bh\s*=|ode45|ode15s|ode23|"
    r"ode solver|reltol|abstol|tolerance|simulink|matlab|python|euler|runge|kutta|zead|taylor|discreti[sz]|"
    r"\bγ\s*=|gamma\s*=|\bλ\s*=|design parameter|gain|convergence time|settling time|predefined|fixed[- ]time|"
    r"finite[- ]time|noise|github|gitee|zenodo|code is|codes are|source code|available at|data availab|"
    r"hardware|fpga|circuit|physical|experiment|robot|manipulator|dimension|\bn\s*=|×|compared with|"
    r"comparison|baseline|existing model|\bgnn\b|\bznn\b", re.I)


def row(pid):
    for name in ("sample.csv", "reserves.csv"):
        for r in csv.DictReader(open(HERE / name)):
            if r["id"] == pid:
                return r
    sys.exit(f"unknown id {pid}")


def curl(url, dst):
    r = subprocess.run(["curl", "-s", "-L", "-m", "60", "-A", UA, "-H", "Accept-Language: en", "-o", str(dst),
                        "-w", "%{http_code}|%{content_type}", url], capture_output=True, text=True, timeout=90)
    code, _, ctype = (r.stdout or "000|").partition("|")
    return code, ctype


def to_text(raw: Path) -> str:
    head = raw.read_bytes()[:5]
    if head == b"%PDF-":
        r = subprocess.run(["pdftotext", "-layout", str(raw), "-"], capture_output=True, text=True, timeout=120)
        return r.stdout
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tf:
        out = Path(tf.name)
    html = raw.with_suffix(".html")
    raw.replace(html)
    r = subprocess.run(["textutil", "-convert", "txt", "-output", str(out), str(html)], capture_output=True,
                       timeout=120)
    if r.returncode == 0 and out.exists():
        return out.read_text(errors="ignore")
    t = html.read_text(errors="ignore")
    t = re.sub(r"(?is)<(script|style).*?</\1>", " ", t)
    return re.sub(r"\s+\n", "\n", re.sub(r"<[^>]+>", " ", t))


def looks_full(t: str) -> bool:
    low = t.lower()
    return len(t) > 15000 and "conclusion" in low and ("references" in low or "bibliography" in low) \
        and not re.search(r"just a moment|verify you are human|enable javascript and cookies", low)


def show(pid, max_lines):
    p = OUT / f"{pid}.txt"
    lines = p.read_text(errors="ignore").splitlines()
    # stop at the reference list (last occurrence of a references heading)
    end = len(lines)
    for i in range(len(lines) - 1, int(len(lines) * 0.5), -1):
        if re.match(r"\s*(references|bibliography)\s*$", lines[i], re.I):
            end = i
            break
    keep = set()
    for i, ln in enumerate(lines[:end]):
        if KW.search(ln):
            keep.update({i - 1, i, i + 1})
    shown = 0
    print(f"== {pid}: {p} ({len(lines)} lines, references start ~line {end})")
    for i in sorted(k for k in keep if 0 <= k < end):
        s = lines[i].strip()
        if s:
            print(f"{i + 1:5d}: {s[:300]}")
            shown += 1
            if shown >= max_lines:
                print(f"... capped at {max_lines} lines; grep {p} for more")
                break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("id")
    ap.add_argument("--url")
    ap.add_argument("--from-file")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--max-lines", type=int, default=160)
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    dst = OUT / f"{a.id}.txt"
    if a.show:
        return show(a.id, a.max_lines)
    if a.from_file:
        t = Path(a.from_file).read_text(errors="ignore")
        dst.write_text(t)
        print(f"saved {len(t)} chars; full={looks_full(t)}")
        return show(a.id, a.max_lines)
    r = row(a.id)
    urls = ([a.url] if a.url else []) + [r["fulltext_url"]] + r.get("alt_urls", "").split() + \
           ["https://doi.org/" + r["doi"]]
    for u in dict.fromkeys(x for x in urls if x):
        raw = OUT / f"{a.id}.raw"
        try:
            code, ctype = curl(u, raw)
        except Exception as e:  # noqa: BLE001
            print(f"  {u}: {e}")
            continue
        if not raw.exists() or raw.stat().st_size < 2000:
            print(f"  {u}: HTTP {code}, too small")
            continue
        t = to_text(raw)
        for f in (OUT / f"{a.id}.raw", OUT / f"{a.id}.html"):
            f.unlink(missing_ok=True)
        ok = looks_full(t)
        print(f"  {u}: HTTP {code} {ctype[:30]} -> {len(t)} chars, full={ok}")
        if ok:
            dst.write_text(t)
            return show(a.id, a.max_lines)
    print("NO FULL TEXT via curl: try WebFetch or the browser (save page text with --from-file)")
    sys.exit(2)


if __name__ == "__main__":
    main()
