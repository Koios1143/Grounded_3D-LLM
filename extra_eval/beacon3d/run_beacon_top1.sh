#!/bin/bash
# Re-run Beacon3D GROUNDING only (ScanNet, 3RScan, MultiScan) with the top-1 proposal choice
# of extra_eval/infer.py (the most probable matched proposal of the first <ref>, instead of the
# lowest query index above 0.5 used by the earlier runs), then score it with the Beacon3D
# grounding evaluator. QA is not re-run: its judged results (GPT-4o-mini) are reused from
# extra_eval/results/beacon3d_<domain> (score.py --beacon_qa_from), so only grounding and the
# grounding->QA chain numbers change. One GPU.
#   bash extra_eval/beacon3d/run_beacon_top1.sh
# Knobs: OUT (default outputs/beacon3d_top1_<timestamp>; a domain whose log ends with
# INFER_DONE is skipped, so re-running with the same OUT resumes), RESULT_DIR (default
# extra_eval/results/beacon3d_top1), DOMAINS (default "scannet 3rscan multiscan").
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
[ -f env.sh ] && source env.sh
unset CONDA_PREFIX CONDA_DEFAULT_ENV
PY="${G3D_ENV:-.venv}/bin/python"
export PYTHONPATH=.:${PYTHONPATH:-} OMP_NUM_THREADS=3 HYDRA_FULL_ERROR=1
OUT="${OUT:-outputs/beacon3d_top1_$(date +%Y%m%d_%H%M%S)}"
RESULT_DIR="${RESULT_DIR:-extra_eval/results/beacon3d_top1}"
mkdir -p "$OUT"
rc=0
for d in ${DOMAINS:-scannet 3rscan multiscan}; do
    log="$OUT/infer_${d}.log"
    if grep -q INFER_DONE "$log" 2>/dev/null; then echo "$d: done already"; continue; fi
    rm -f "$OUT"/beacon3d_"${d}"_grounding*.jsonl
    EXTRA=()
    [ "$d" != scannet ] && EXTRA+=(--domain "$d" --processed_dir "${G3D_ROOT:?set G3D_ROOT}/pointcloud/$d/scannet200")
    # same settings as the original Beacon3D runs (no prefix cache)
    $PY extra_eval/infer.py --out_dir "$OUT" --benches beacon3d --beacon_tasks grounding \
        --prefix_cache 0 "${EXTRA[@]}" > "$log" 2>&1 || rc=1
    tail -n 2 "$log"
done
[ "$rc" = 0 ] || { echo "INFER_EXIT=$rc out=$OUT"; exit 1; }
$PY extra_eval/score.py --pred_dir "$OUT" --result_dir "$RESULT_DIR" --benches beacon3d \
    --beacon_domains scannet,3rscan,multiscan --beacon_qa_from extra_eval/results \
    --qa_model "${QA_MODEL:-gpt-4o-mini-2024-07-18}" > "$OUT/score.log" 2>&1 || { tail -n 30 "$OUT/score.log"; exit 1; }
# old (lowest index) vs new (top-1) grounding case accuracy
$PY - "$RESULT_DIR" <<'PY'
import json, sys
for d in ["scannet", "3rscan", "multiscan"]:
    o = json.load(open(f"extra_eval/results/beacon3d_{d}/scores.json"))
    n = json.load(open(f"{sys.argv[1]}/beacon3d_{d}/scores.json"))
    for v in ["best", "iou25"]:
        g0, g1 = o[f"metrics_{v}"]["grounding"], n[f"metrics_{v}"]["grounding"]
        print(f"{d:9} {v:5} case {g0['case']:6.2f} -> {g1['case']:6.2f}   object {g0['object']:6.2f} -> {g1['object']:6.2f}")
PY
echo "TOP1_DONE out=$OUT results=$RESULT_DIR"
