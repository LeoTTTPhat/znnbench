#!/bin/bash
# E4 full-scale pipeline (two stages: calibration -> sweep built from the calibration -> analysis).
# Resumable: re-run the script; finished tasks are skipped via their done files.
cd "$(dirname "$0")/.."            # repository root
PY=${PYTHON:-python}
W=${1:-8}
mkdir -p logs
LOG=logs/sweep_E4.log
note() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
note "E4 calibration build"
$PY experiments/tasks/E4_calibrate.py build --scale full >> "$LOG" 2>&1 || { note "E4 calibration build FAILED"; exit 1; }
$PY tools/memguard.py pool --tasks results/full/E4/calib/tasks.jsonl --workers "$W" --max-gb 200 >> "$LOG" 2>&1
note "E4 calibration pool rc=$?"
$PY experiments/tasks/E4_calibrate.py sweep --scale full >> "$LOG" 2>&1 || { note "E4 sweep build FAILED"; exit 1; }
note "E4 sweep build done"
$PY tools/memguard.py pool --tasks results/full/E4/sweep/all_tasks.jsonl --workers "$W" --max-gb 200 >> "$LOG" 2>&1
note "E4 sweep pool rc=$?"
$PY experiments/analyze_E4.py --results results/full/E4 --figdir figures --tabdir results/tables >> logs/analyze_E4.log 2>&1
note "E4 analysis rc=$?"
