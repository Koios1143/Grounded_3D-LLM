#!/bin/bash
# Aligned re-scoring of the saved predictions (CPU only, needs java for PTB / METEOR):
#   bash extra_eval/aligned/run_score_aligned.sh
# Writes extra_eval/results/aligned/. NATIVE_DIR (default outputs/native_nocache) is the
# infer.py output dir of the native test pass.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
[ -f env.sh ] && source env.sh
[ -x deps/jre11/bin/java ] && export JAVA_HOME="$PWD/deps/jre11"
export PATH="${JAVA_HOME:?set JAVA_HOME}/bin:$PATH" OMP_NUM_THREADS=4
PY="${G3D_ENV:-.venv}/bin/python"
$PY extra_eval/aligned/score_aligned.py --native_dir "${NATIVE_DIR:-outputs/native_nocache}" "$@"
