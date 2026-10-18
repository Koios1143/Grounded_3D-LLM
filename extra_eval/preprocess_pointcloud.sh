#!/bin/bash
# Beacon3D 3RScan / MultiScan scenes (SceneVerse point clouds) and the ScanNet control
# (scannet_pcseg) -> processed layout under $G3D_ROOT/pointcloud/<domain>/scannet200.
#   bash extra_eval/preprocess_pointcloud.sh                 # all three
#   DOMAINS=3rscan bash extra_eval/preprocess_pointcloud.sh
# CPU only. Needs preprocess_scannet.sh first (ScanNet200 label/colour files, scannet_pcseg).
# Paths come from env.sh (see env.sh.example): G3D_ENV, G3D_ROOT, SCENEVERSE_DIR.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
[ -f env.sh ] && source env.sh
PY="${G3D_ENV:-.venv}/bin/python"
for d in $(echo "${DOMAINS:-3rscan,multiscan,scannet_pcseg}" | tr ',' ' '); do
    $PY extra_eval/preprocess_pointcloud.py --domain "$d" --sceneverse "${SCENEVERSE_DIR:-}" \
        --out_dir "${G3D_ROOT:?set G3D_ROOT}/pointcloud" --workers "${WORKERS:-16}"
done
echo PREPROCESS_DONE
