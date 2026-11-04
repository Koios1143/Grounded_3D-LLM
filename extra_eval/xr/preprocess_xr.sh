#!/bin/bash
# XR-QA / XR-SceneCaption scenes -> processed layout under $XR_PC/xr_{rc,glb}/scannet200
# (+ xr_<split>_items.json). CPU only; regions are cached, so re-runs only assemble scenes.
#   bash extra_eval/xr/preprocess_xr.sh
#   SPLITS="train val" bash extra_eval/xr/preprocess_xr.sh
#   NORMALS=pca XR_PC=$G3D_ROOT/pointcloud/xr bash extra_eval/xr/preprocess_xr.sh   # the first runs
# Needs preprocess_scannet.sh first (ScanNet200 label / colour files).
# Paths come from env.sh (see env.sh.example): G3D_ENV, G3D_ROOT, XR_DIR, XR_SCAN_DATA, XR_PC
# (default $G3D_ROOT/pointcloud/xr), and for NORMALS=glb (default) HM3D_GLB.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
[ -f env.sh ] && source env.sh
PY="${G3D_ENV:-.venv}/bin/python"
NORMALS="${NORMALS:-glb}"
EXTRA=()
[ "$NORMALS" = glb ] && EXTRA=(--glb_dir "${HM3D_GLB:?}")
$PY extra_eval/xr/preprocess_xr.py --xr_dir "${XR_DIR:?}" --scan_data "${XR_SCAN_DATA:?}" \
    --out_dir "${XR_PC:-${G3D_ROOT:?set G3D_ROOT}/pointcloud/xr}" --splits ${SPLITS:-val} --workers "${WORKERS:-16}" \
    --normals "$NORMALS" "${EXTRA[@]}"
echo PREPROCESS_DONE
