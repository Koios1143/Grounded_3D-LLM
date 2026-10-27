#!/bin/bash
# Preprocess the ScanNet val scans for the released model;
# links the result to ./data/processed/scannet200 and ./data/rawscannet (read for the
# axis-alignment matrices).
#   bash extra_eval/preprocess_scannet.sh
# CPU only. Each scene needs _vh_clean_2.ply, _vh_clean_2.0.010000.segs.json,
# .aggregation.json and .txt (axisAlignment), in $SCANNET_SCANS/<scene>/. Paths come from env.sh (see env.sh.example): G3D_ENV, G3D_ROOT, SCANNET_SCANS,
# SCANNET_LABEL_TSV.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
[ -f env.sh ] && source env.sh
PY="${G3D_ENV:-.venv}/bin/python"
ROOT="${G3D_ROOT:?set G3D_ROOT}"
$PY extra_eval/preprocess_scannet.py --scans "${SCANNET_SCANS:?}" --label_tsv "${SCANNET_LABEL_TSV:?}" \
    --out_dir "$ROOT/scannet" --workers "${WORKERS:-16}"
mkdir -p data/processed
ln -sfn "$ROOT/scannet/scannet200" data/processed/scannet200
ln -sfn "$(cd "$SCANNET_SCANS/.." && pwd)" data/rawscannet  # read for <scene>.txt (axis alignment)
echo PREPROCESS_DONE
