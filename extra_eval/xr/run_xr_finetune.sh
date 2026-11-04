#!/bin/bash
# Finetune Grounded 3D-LLM on XR-QA + XR-SceneCaption train (see finetune_xr.py), evaluating
# on val after every epoch, then score every epoch (CPU).
#   PLACEMENT=glb bash extra_eval/xr/run_xr_finetune.sh
# Needs SPLITS="train val" bash extra_eval/xr/preprocess_xr.sh.
# Knobs: PLACEMENT (rc | glb, default glb), OUT (default outputs/xr_ft_<placement>_<timestamp>),
#   EPOCHS (3), LR (2e-4, OneCycle max), BATCH_SIZE (32), NGPU (detector-cache processes).
# Detector features are cached in $XR_PC/cache_<placement> (reused across runs). Resumable: a
# rerun with the same OUT continues after the last finished epoch (finetune_xr.py resume.pt).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
[ -f env.sh ] && source env.sh
PY="${G3D_ENV:-.venv}/bin/python"
export PYTHONPATH=.:${PYTHONPATH:-} OMP_NUM_THREADS=3 HYDRA_FULL_ERROR=1
PL="${PLACEMENT:-glb}"
P="${XR_PC:-${G3D_ROOT:?set G3D_ROOT}/pointcloud/xr}"  # XR_PC: glb-normal cache (env.sh)
NGPU="${NGPU:-$(nvidia-smi -L | wc -l)}"
# under Slurm the job id names OUT, so a requeued job (same id) resumes per epoch (resume.pt)
OUT="${OUT:-outputs/xr_ft_${PL}_${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}}"
mkdir -p "$OUT"
pids=()
for ((i = 0; i < NGPU; i++)); do
    CUDA_VISIBLE_DEVICES=$i $PY extra_eval/xr/finetune_xr.py cache --placement "$PL" \
        --processed_dir "$P/xr_$PL/scannet200" --cache_dir "$P/cache_$PL" --shard "$i/$NGPU" \
        > "$OUT/cache_shard$i.log" 2>&1 &
    pids+=($!)
done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
[ $rc = 0 ] || { tail -n 20 "$OUT"/cache_shard*.log; echo "CACHE failed"; exit 1; }
CUDA_VISIBLE_DEVICES=0 $PY extra_eval/xr/finetune_xr.py train --placement "$PL" \
    --processed_dir "$P/xr_$PL/scannet200" --cache_dir "$P/cache_$PL" \
    --train_items "$P/xr_train_items.json" --val_items "$P/xr_val_items.json" --out_dir "$OUT" \
    --epochs "${EPOCHS:-3}" --lr "${LR:-2e-4}" --batch_size "${BATCH_SIZE:-32}" >> "$OUT/train.log" 2>&1 || rc=1
tail -n 3 "$OUT/train.log"
for d in "$OUT"/epoch*; do
    for t in qa cap; do
        $PY extra_eval/xr/score_xr.py --preds "$d/xr${t}_${PL}.jsonl" --items "$P/xr_val_items.json" \
            --out "$d/xr${t}_${PL}.scores.json" || rc=1
    done
done
echo "XR_FT_EXIT=$rc out=$OUT"
