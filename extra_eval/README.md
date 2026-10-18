# Grounded 3D-LLM on SQA3D, Real-3DQA and Beacon3D

Evaluation of the **released** Grounded 3D-LLM checkpoint
(`saved/step3_mask3d_lang_4GPUS/last-epoch.ckpt` from the HF dataset
[`chenyilun95/Grounded_3D-LLM_data`](https://huggingface.co/datasets/chenyilun95/Grounded_3D-LLM_data))
on benchmarks the paper does not report. No training or fine-tuning: SQA3D and Real-3DQA
are **zero-shot** for this model (it never saw SQA3D).

## Results

All numbers are percentages, ScanNet val, released checkpoint, zero-shot for SQA3D / Real-3DQA.

**SQA3D** (test, 3,519) — `results/sqa3d/scores.json`

| EM | EM-R | What | Is | How | Can | Which | Others |
|---|---|---|---|---|---|---|---|
| 33.70 | 36.74 | 21.77 | 49.23 | 28.82 | 47.63 | 38.75 | 32.51 |

(per-type columns are EM; 46 empty generations counted wrong)

**Real-3DQA** — `results/real3dqa/` (official scripts' output in `official_scripts.txt`)

| Debiased EM | Debiased Refined EM | One | Two | Three | Four | VRS |
|---|---|---|---|---|---|---|
| 18.25 | 21.82 | 38.8 | 6.0 | 0.7 | 0.1 | 11.4 |

Per rotation (EM / Refined EM): 0° 9.73 / 13.47, 90° 8.93 / 10.53, 180° 10.00 / 12.80,
270° 6.80 / 8.80.

**Beacon3D, ScanNet** — `results/beacon3d_scannet/` (QA judge: gpt-4o-mini-2024-07-18)

| Grounding variant | Case | Class | App. | Geo. | Spa. | Func. | Chain | Object |
|---|---|---|---|---|---|---|---|---|
| any overlap | 57.37 | 57.37 | 50.92 | 49.32 | 52.05 | 11.11 | 50.22 | 44.42 |
| IoU ≥ 0.25 | 56.35 | 56.35 | 49.69 | 47.49 | 51.03 | 11.11 | 49.35 | 43.64 |

| QA | EM | EM-R | GPT-Score (case) | Object | Class | App. | Geo. | Spa. | Exist. | Func. |
|---|---|---|---|---|---|---|---|---|---|---|
| | 41.21 | 43.90 | 44.57 | 8.05 | 18.24 | 40.69 | 46.99 | 51.07 | 52.24 | 33.33 |

GQA-chain analysis vs. the "any overlap" grounding: R1 37.85, R2 54.10 (vs. IoU ≥ 0.25:
R1 37.66, R2 55.15; `score_qa_iou25.txt`). The GPT judge is stochastic, so a re-score moves
the GPT-Score slightly (44.57 vs 44.74 in the two passes over the same predictions).

## What is run

Per scene, the frozen Mask3D-CLASP detector runs exactly as in the official test pass
(`final_scripts/step3_train_grounded3dllm.sh`, TEST block): `data=indoor_dialog`,
`model=mask3d_lang`, `conf/llm/tiny_vicuna_len512.json`, ScanNet val, axis-aligned scans. The
detector's BiFormer also reads text (ScanNet200 class names + the scene's ScanRefer /
Multi3DRefer sentences); that input is kept identical, so the 100 object queries the LLM
sees are the ones behind the paper's ScanQA numbers. Our questions then go through
`llama_model.evaluate` with the official decoding (beam 3, repetition penalty 1.2, ≤150 new
tokens, batch 50). See the docstring of `infer.py`.

| Benchmark | Input | Prompt (`lang_type`) | Scoring |
|---|---|---|---|
| SQA3D test (3,519) | official balanced test json | `<situation> <question> Please answer with a single word or phrase.` (ScanQA prompt) | EM / EM-R with LEO's `clean_answer` + refined match (as LEO / Chat-Scene); per type by first word |
| Real-3DQA (1,485 + 4×750) | `debiased_test.jsonl`, `rotation_{0,90,180,270}.jsonl` | same as SQA3D | the two official scripts from [xianzhengma/Real-3DQA](https://github.com/xianzhengma/Real-3DQA), vendored in `real3dqa/` |
| Beacon3D ScanNet grounding (1,611) | Beacon3D `scannet_grounding.json` | first ScanRefer template + ` (with grounding)` | first `<ref>` → proposal → GT object of highest point IoU; Beacon3D's `evaluate_grounding.py` |
| Beacon3D ScanNet QA (1,155) | Beacon3D `scannet_qa.json` | ScanQA prompt | Beacon3D's `evaluate_qa.py`, judge **gpt-4o-mini-2024-07-18** |

Notes:

* **Grounding to predicted masks.** Beacon3D was designed around models that pick one of
  the GT objects (e.g. Chat-Scene with GT masks). Grounded 3D-LLM can only point at its own
  detector proposals, so its pick is mapped to the GT object with the highest point IoU.
  Two variants are scored: any overlap (closest to "picked the right object"; the
  `infer_grounding*.json` without suffix) and IoU ≥ 0.25 (`_iou25`).
* **Empty generations** (only `</s>`) are replaced by `<empty>` before scoring: under the
  refined / relaxed match an empty string is a substring of every answer and would count as
  correct (the repo's own `evaluate_LLM.py` guards this with `set_zero_as_error`). Counts are
  in the score files (`empty_preds`).
* **Real-3DQA ids** are written as integers: the official rotation script `int()`s the
  suffix-stripped ids of the 90/180/270 files but not the 0° ones.
* **Beacon3D QA judge:** gpt-4o-mini-2024-07-18 instead of the upstream gpt-4o-2024-11-20.
  The GPT-Score is therefore not directly comparable to numbers judged by GPT-4o.
* Beacon3D 3RScan / MultiScan are not run: the released detector is trained on ScanNet
  meshes and needs mesh superpoints, which those domains do not provide.

## Environment (Blackwell GPUs)

The release pins torch 1.12.1+cu116, which has no kernels for sm_120. `build_env.sh` builds a
uv venv with torch 2.7.1+cu128, MinkowskiEngine at the authors' commit + `minkowski_cuda12.patch`
(thrust includes, NVTX3 clash with CUDA 12's bundled copy, a `shared_ptr` / `__to_address`
ambiguity under GCC 13), and the repo's pointnet2 ops (`.data<T>()` → `.data_ptr<T>()`).
`requirements.txt` here lists the runtime deps (mmcv-full → mmcv-lite, mmengine 0.10.7,
pytorch-lightning 1.9.5). The checkpoint loads with 0 missing / 0 unexpected keys.

## Data

* ScanNet val scans: per scene `_vh_clean_2.ply`, `_vh_clean_2.0.010000.segs.json`,
  `.aggregation.json`, `.txt` (axis alignment). The **official** segs.json is required —
  the aggregation's object masks are lists of its segment ids, and re-running ScanNet's
  Segmentator does not reproduce them. 310 of the 312 val scenes were available
  (scene0011_01 and scene0217_00 missing; no benchmark question uses them).
* `preprocess_scannet.py` writes the Mask3D-style `data/processed/scannet200` the code
  reads. `color_mean_std.yaml` is the ScanNet200 *train* statistics, taken from Mask3D's
  inference code (the train split is not needed otherwise).
* SQA3D: `sqa_task.zip` from [Zenodo 7792397](https://zenodo.org/record/7792397)
  (`balanced/v1_balanced_*_test_scannetv2.json`).
* Real-3DQA: HF dataset [`Oliver-Ma/Real-3DQA`](https://huggingface.co/datasets/Oliver-Ma/Real-3DQA) `data/test/`.
* Beacon3D: vendored in `beacon3d/Beacon3D/` (commit `bda1203`, see `VENDORED.md`).

## Reproduce

```bash
cp env.sh.example env.sh        # fill in paths (+ OPENAI_API_KEY for Beacon3D QA)
bash extra_eval/build_env.sh    # GPU node, compiles CUDA code
bash extra_eval/download.sh     # checkpoint, BERT, CLIP label features, Tiny-Vicuna-1B, langdata
bash extra_eval/preprocess_scannet.sh
bash extra_eval/run_infer.sh    # OUT=outputs/<name>; one process per visible GPU
PREDS_DIR=outputs/<name> bash extra_eval/score.sh   # -> extra_eval/results/
```

`results/` holds the per-question predictions (`infer_*.json`, `pred_*.json`), the official
scripts' output (`score_*.txt`, `official_scripts.txt`) and `summary.json`.
