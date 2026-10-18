#!/bin/bash
# Grounded 3D-LLM inference on SQA3D / Real-3DQA / Beacon3D (ScanNet); one process per GPU,
# each taking every NGPU-th scene. Needs build_env.sh, download.sh, preprocess_scannet.sh.
#   bash extra_eval/run_infer.sh
#   OUT=outputs/debug MAX_SCENES=2 BENCHES=sqa3d bash extra_eval/run_infer.sh
# Paths come from env.sh (see env.sh.example): G3D_ENV, SQA3D_DIR, REAL3DQA_DIR.
#   OUT        output dir           (default outputs/extra_eval_<timestamp>)
#   NGPU       processes / GPUs     (default: all visible)
#   BENCHES    comma list           (default sqa3d,real3dqa,beacon3d)
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
[ -f env.sh ] && source env.sh
unset CONDA_PREFIX CONDA_DEFAULT_ENV
PY="${G3D_ENV:-.venv}/bin/python"
export PYTHONPATH=.:${PYTHONPATH:-} OMP_NUM_THREADS=3 HYDRA_FULL_ERROR=1
NGPU="${NGPU:-$(nvidia-smi -L | wc -l)}"
OUT="${OUT:-outputs/extra_eval_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT"
EXTRA=()
[ -n "${MAX_SCENES:-}" ] && EXTRA+=(--max_scenes "$MAX_SCENES")
pids=()
for ((i = 0; i < NGPU; i++)); do
    CUDA_VISIBLE_DEVICES=$i $PY extra_eval/infer.py --out_dir "$OUT" \
        --benches "${BENCHES:-sqa3d,real3dqa,beacon3d}" --shard "$i/$NGPU" "${EXTRA[@]}" \
        > "$OUT/infer_shard$i.log" 2>&1 &
    pids+=($!)
done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
tail -n 3 "$OUT"/infer_shard*.log
echo "INFER_EXIT=$rc out=$OUT"
