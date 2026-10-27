#!/bin/bash
# Grounded 3D-LLM inference on ScanRefer / Multi3DRefer / Scan2Cap / ScanQA / SQA3D / Real-3DQA /
# Beacon3D (ScanNet); one process per GPU, each taking every NGPU-th scene. Needs build_env.sh,
# download.sh, preprocess_scannet.sh.
#   PREFIX_CACHE=1 bash extra_eval/run_infer.sh
#   OUT=outputs/debug MAX_SCENES=2 BENCHES=sqa3d bash extra_eval/run_infer.sh
# Paths come from env.sh (see env.sh.example): G3D_ENV, SQA3D_DIR, REAL3DQA_DIR.
#   OUT        output dir           (default outputs/extra_eval_<timestamp>)
#   NGPU       processes / GPUs     (default: all visible)
#   BENCHES    comma list           (default scanrefer,m3dref,scan2cap,scanqa,sqa3d,real3dqa,beacon3d)
#   PREFIX_CACHE 1: encode the LLM prefix shared by a scene's questions once and reuse its
#              KV cache (same outputs up to floating-point noise, faster) (default 0)
#   DOMAIN     scannet | scannet_pcseg | 3rscan | multiscan (default scannet); the others run
#              Beacon3D only, on preprocess_pointcloud.sh output in $G3D_ROOT/pointcloud/<domain>
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
DOMAIN="${DOMAIN:-scannet}"
[ "$DOMAIN" != scannet ] && EXTRA+=(--domain "$DOMAIN" --processed_dir "${G3D_ROOT:?set G3D_ROOT}/pointcloud/$DOMAIN/scannet200")
pids=()
for ((i = 0; i < NGPU; i++)); do
    CUDA_VISIBLE_DEVICES=$i $PY extra_eval/infer.py --out_dir "$OUT" \
        --benches "${BENCHES:-scanrefer,m3dref,scan2cap,scanqa,sqa3d,real3dqa,beacon3d}" \
        --prefix_cache "${PREFIX_CACHE:-0}" --shard "$i/$NGPU" "${EXTRA[@]}" \
        > "$OUT/infer_${DOMAIN}_shard$i.log" 2>&1 &
    pids+=($!)
done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
tail -n 3 "$OUT"/infer_${DOMAIN}_shard*.log
echo "INFER_EXIT=$rc out=$OUT"
