#!/bin/bash
# Download the released Grounded 3D-LLM checkpoint + the files inference needs, into $G3D_ROOT,
# and symlink them where the code expects them (./pretrained, ./saved, ./data/langdata).
#   bash extra_eval/download.sh
# Network only, no GPU. Paths come from env.sh (see env.sh.example).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
[ -f env.sh ] && source env.sh
ROOT="${G3D_ROOT:?set G3D_ROOT}"
HF="uvx --from huggingface_hub hf"
REL="$ROOT/release"
$HF download chenyilun95/Grounded_3D-LLM_data --repo-type dataset --local-dir "$REL" \
    --include "saved/step3_mask3d_lang_4GPUS/*" --include "pretrained/*" \
    --include "langdata/scanrefer_format.json" --include "langdata/m3dref_format.json" \
    --include "langdata/scanqa_format.json"
$HF download Jiayi-Pan/Tiny-Vicuna-1B --exclude "*.safetensors" --local-dir "$REL/pretrained/llm_weight/Tiny-Vicuna-1B"
mkdir -p data
ln -sfn "$REL/pretrained" pretrained
ln -sfn "$REL/saved" saved
ln -sfn "$REL/langdata" data/langdata
ls -la pretrained/ saved/*/ data/
echo DOWNLOAD_DONE
