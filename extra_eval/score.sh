#!/bin/bash
# Score an extra_eval/run_infer.sh output dir (CPU only).
#   PREDS_DIR=outputs/extra_eval_<ts> bash extra_eval/score.sh
# OPENAI_API_KEY (env.sh) enables Beacon3D QA GPT scoring (QA_MODEL, default
# gpt-4o-mini-2024-07-18); RESULT_DIR defaults to extra_eval/results; BEACON_DOMAINS limits
# which Beacon3D domains are (re)scored (the overall score reads all three from RESULT_DIR).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
[ -f env.sh ] && source env.sh
PY="${G3D_ENV:-.venv}/bin/python"
export PYTHONPATH=.:${PYTHONPATH:-}
$PY extra_eval/score.py --pred_dir "${PREDS_DIR:?set PREDS_DIR}" \
    --result_dir "${RESULT_DIR:-extra_eval/results}" --benches "${BENCHES:-scanrefer,m3dref,scan2cap,scanqa,sqa3d,real3dqa,beacon3d}" --qa_model "${QA_MODEL:-gpt-4o-mini-2024-07-18}" \
    --beacon_domains "${BEACON_DOMAINS:-scannet,3rscan,multiscan,scannet_pcseg}"
