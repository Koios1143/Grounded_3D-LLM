#!/bin/bash
# ScanRefer / Multi3DRefer of the native test pass over the official val sets (CPU, seconds):
#   bash extra_eval/run_score_official_val.sh
# Writes extra_eval/results/official_val/scanrefer_m3dref.json (asked-only and official-val
# scores). NATIVE_DIR (default outputs/native_nocache) is the infer.py output dir of the native
# test pass; the official files default to deps/official_val/, the unique / multiple lookup to
# deps/scanrefer_official/ (extra_eval/build_scanrefer_lookup.py; see env.sh.example).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
[ -f env.sh ] && source env.sh
D="${OFFICIAL_VAL_DIR:-deps/official_val}"
PY="${G3D_ENV:-.venv}/bin/python"
$PY extra_eval/score_official_val.py --native_dir "${NATIVE_DIR:-outputs/native_nocache}" \
    --scanrefer_val "${SCANREFER_VAL_JSON:-$D/ScanRefer_filtered_val.json}" \
    --m3dref_val "${M3DREF_VAL_JSON:-$D/multi3drefer_val.json}" \
    --scanrefer_lookup "${SCANREFER_LOOKUP_JSON:-deps/scanrefer_official/scanrefer_unique_multiple_lookup.json}" "$@"
