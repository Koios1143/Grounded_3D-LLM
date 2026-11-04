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
# XR-SceneCaption SPICE, same setup as just3d: SPICE 1.0 jars in deps/spice, JDK 17 in deps/jdk17
if [ -f deps/spice/spice-1.0.jar ] && [ -x deps/jdk17/bin/java ]; then
    export ALIGNED_SPICE_DIR="$PWD/deps/spice" ALIGNED_SPICE_JAVA="$PWD/deps/jdk17/bin/java"
    export ALIGNED_SPICE_CACHE="${ALIGNED_SPICE_CACHE:-$PWD/deps/spice_cache}"
    export ALIGNED_SPICE_SHARDS="${ALIGNED_SPICE_SHARDS:-6}" ALIGNED_SPICE_HEAP="${ALIGNED_SPICE_HEAP:-6G}"
else
    echo "[warn] deps/spice or deps/jdk17 missing: XR-SceneCaption scored without SPICE" >&2
fi
PY="${G3D_ENV:-.venv}/bin/python"
$PY extra_eval/aligned/score_aligned.py --native_dir "${NATIVE_DIR:-outputs/native_nocache}" "$@"
