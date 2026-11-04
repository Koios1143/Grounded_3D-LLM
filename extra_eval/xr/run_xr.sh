#!/bin/bash
# Grounded 3D-LLM zero-shot inference on XR-QA / XR-SceneCaption val, then scoring (CPU).
#   bash extra_eval/xr/run_xr.sh                       # both placements
#   PLACEMENTS=glb bash extra_eval/xr/run_xr.sh
# Needs preprocess_xr.sh. Knobs: OUT (default outputs/xr_<timestamp>), PLACEMENTS ("rc glb"),
# NGPU, PREFIX_CACHE, MAX_SCENES (as run_infer.sh).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
[ -f env.sh ] && source env.sh
export OUT="${OUT:-outputs/xr_$(date +%Y%m%d_%H%M%S)}"
P="${XR_PC:-${G3D_ROOT:?set G3D_ROOT}/pointcloud/xr}"  # XR_PC: glb-normal cache (env.sh)
rc=0
for pl in ${PLACEMENTS:-rc glb}; do
    DOMAIN=xr_$pl PROCESSED_DIR="$P/xr_$pl/scannet200" XR_ITEMS="$P/xr_val_items.json" \
        bash extra_eval/run_infer.sh || rc=1
done
for pl in ${PLACEMENTS:-rc glb}; do
    for t in qa cap; do
        "${G3D_ENV:-.venv}/bin/python" extra_eval/xr/score_xr.py --preds "$OUT"/xr${t}_${pl}*.jsonl \
            --items "$P/xr_val_items.json" --out "$OUT/xr${t}_${pl}.scores.json" || rc=1
    done
done
echo "XR_EXIT=$rc out=$OUT"
