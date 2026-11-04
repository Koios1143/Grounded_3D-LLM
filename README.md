# Grounded 3D-LLM Benchmark Evaluation

This is a fork of [Grounded 3D-LLM](https://github.com/OpenRobotLab/Grounded_3D-LLM)
(arXiv 2405.10370) whose purpose is to **evaluate the released Grounded 3D-LLM checkpoint
on benchmarks the paper does not report**, under one reproducible setup, so its scores can
be compared with other models on equal footing. Wired in so far: the paper's own
ScanNet benchmarks ScanRefer, Multi3DRefer, Scan2Cap and ScanQA (re-run as a check on the
setup), [SQA3D](https://sqa3d.github.io/) (ICLR 2023), [Real-3DQA](https://real-3dqa.github.io/)
(ICLR 2026, a debiased SQA3D subset with viewpoint rotations) and
[Beacon3D](https://beacon-3d.github.io/) (CVPR 2025; ScanNet, 3RScan, MultiScan). The upstream model,
training and data code are untouched apart from a torch-2 compatibility fix in the
pointnet2 ops and an optional per-scene prefix cache for decoding (`PREFIX_CACHE=1`, off by
default; see [Running](#running)); everything this fork adds lives under [`extra_eval/`](extra_eval/), and
the per-question outputs behind the numbers below are committed in
[`extra_eval/results/`](extra_eval/results/).

Upstream Grounded 3D-LLM's own README is in the git history (`git show 9de50f1:README.md`).

## Results

Released checkpoint `saved/step3_mask3d_lang_4GPUS/last-epoch.ckpt`, ScanNet val, no
training or fine-tuning of any kind. All numbers are percentages.

### ScanRefer, Multi3DRefer, Scan2Cap, ScanQA (val; the paper's benchmarks)

These run the repository's own test pass (`eval_step` → `llama_model.evaluate` →
`eval_llm_iou_score`, the instructions built by the release's `langdata/`) and are scored
with the metric code of `models/metrics/evaluate_LLM.py` and the official Multi3DRefer
evaluator. They reproduce the paper's Table III (generalist model) to within 0.3:

| | ScanRefer Acc@0.25 | Acc@0.5 | M3DRef F1@0.25 | F1@0.5 | ScanQA B-4 | CIDEr | Scan2Cap B-4@0.5 | C@0.5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| paper | 48.6 | 44.0 | 44.7 | 40.8 | 13.2 | 75.9 | 35.0 | 70.2 |
| this run | 48.62 | 44.31 | 44.58 | 40.92 | 13.29 | 75.98 | 35.24 | 70.39 |
| this run, `PREFIX_CACHE=1` | 48.63 | 44.32 | 44.61 | 40.96 | 13.29 | 75.99 | 35.22 | 70.44 |

- **ScanRefer** (9,257 of the 9,503 val sentences asked, see below): unique 84.34 / 76.80,
  multiple 39.73 / 36.23. Counting the 246 unasked sentences as misses: 47.36 / 43.17.
- **Multi3DRefer** (10,855 of 11,113 asked), F1@0.25 / @0.5 per type: single target with /
  without distractors 43.50 / 40.00 and 80.85 / 73.85, multiple targets 34.01 / 31.45, zero
  target 0 / 0 (the model always outputs at least one object). Unasked queries as F1 = 0:
  43.54 / 39.97.
- **ScanQA** (4,675 questions): BLEU-1 39.38, METEOR 15.33, ROUGE-L 37.06, EM 19.94, EM-R
  35.51.
- **Scan2Cap** (2,068 objects): @0.25 BLEU-4 37.36, METEOR 27.18, ROUGE-L 59.25, CIDEr
  74.36; @0.5 METEOR 26.26, ROUGE-L 56.82. The best proposal reaches IoU ≥ 0.25 / 0.5 for
  93.5 / 86.8% of the objects.

Per-question outputs: `extra_eval/results/{scanrefer,m3dref,scan2cap,scanqa}/predictions.jsonl`;
the Multi3DRefer evaluator's printout is in `results/m3dref/official_evaluator.txt`.

### SQA3D (test, 3,519 questions)

| EM | EM-R | What | Is | How | Can | Which | Others |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 33.70 | 36.74 | 21.77 | 49.23 | 28.82 | 47.63 | 38.75 | 32.51 |

Per-type columns are EM. EM-R is the refined EM of LEO / Chat-Scene.

### Real-3DQA

| Debiased EM | Debiased Refined EM | One | Two | Three | Four | VRS |
| --- | --- | --- | --- | --- | --- | --- |
| 18.25 | 21.82 | 38.8 | 6.0 | 0.7 | 0.1 | 11.4 |

Debiased set: 1,485 questions. Rotation set: 750 questions × 4 viewpoints; *One…Four* = share of
questions answered (relaxed match) correctly under at least 1…4 of the viewpoints, VRS =
their mean. Per viewpoint (EM / Refined EM): 0° 9.73 / 13.47, 90° 8.93 / 10.53,
180° 10.00 / 12.80, 270° 6.80 / 8.80.

### Beacon3D

All three domains (ScanNet, 3RScan, MultiScan). *Case* = per-question accuracy; *Chain* =
grounding chains with every step correct; *Obj* = object counted correct only if all its
grounding chains (grounding) or all its questions (QA) are. Beacon3D defines no
cross-domain score, so two overall rows are given: **pooled** = every case / chain /
object of the three domains together; **mean** = average of the three domain scores. The
*ScanNet control* row is not part of the overall: it reruns ScanNet under the 3RScan /
MultiScan input conditions (see below).

**Grounding** (proposal → best-IoU GT object; in parentheses: also requiring IoU ≥ 0.25)

| | Case | Class | App. | Geo. | Spa. | Func. | Chain | Obj |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Overall, pooled** | 45.26 (43.17) | 45.26 | 39.70 | 39.24 | 39.85 | 11.11 | 37.70 (36.07) | 34.31 (32.72) |
| **Overall, mean** | 42.09 (39.52) | 42.09 | 37.38 | 33.99 | 37.29 | 11.11 | 34.18 (32.17) | 31.21 (29.29) |
| ScanNet | 57.37 (56.35) | 57.37 | 50.92 | 49.32 | 52.05 | 11.11 | 50.22 (49.35) | 44.42 (43.64) |
| 3RScan | 36.79 (34.67) | 36.79 | 30.04 | 30.77 | 30.96 | – | 27.57 (25.98) | 27.21 (25.37) |
| MultiScan | 32.12 (27.55) | 32.12 | 31.17 | 21.88 | 28.86 | – | 24.74 (21.17) | 22.01 (18.87) |
| *ScanNet control* | *54.75 (52.58)* | *54.75* | *48.47* | *46.58* | *49.18* | *11.11* | *47.62 (46.23)* | *42.08 (40.26)* |

**QA** (GPT columns judged by gpt-4o-mini-2024-07-18)

| | EM | EM-R | Case (GPT) | Obj | Class | App. | Geo. | Spa. | Exist. | Func. | R1 ↑ | R2 ↓ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Overall, pooled** | 36.76 | 39.62 | 39.78 | 4.17 | 17.58 | 36.96 | 44.09 | 43.51 | 45.12 | 33.33 | 31.31 | 63.02 |
| **Overall, mean** | 34.19 | 37.00 | 37.11 | 3.23 | 18.09 | 34.63 | 42.22 | 40.24 | 42.84 | 33.33 | 27.22 | 67.38 |
| ScanNet | 41.21 | 43.90 | 44.57 | 8.05 | 18.24 | 40.69 | 46.99 | 51.07 | 52.24 | 33.33 | 37.85 | 54.10 |
| 3RScan | 38.73 | 42.16 | 41.45 | 0.37 | 25.54 | 39.66 | 41.14 | 44.10 | 44.08 | – | 24.39 | 72.37 |
| MultiScan | 22.64 | 24.95 | 25.31 | 1.26 | 10.48 | 23.54 | 38.54 | 25.55 | 32.20 | – | 19.42 | 75.65 |
| *ScanNet control* | *41.39* | *43.72* | *44.61* | *8.05* | *18.92* | *40.62* | *47.94* | *50.49* | *52.35* | *33.33* | *37.97* | *56.53* |

R1 / R2 are Beacon3D's grounding→QA chain metrics against the first grounding variant
(R1 = among wrong answers without extra knowledge, the share whose object was grounded
correctly; R2 = among correct answers, the share whose object was not grounded); against the IoU ≥ 0.25 variant see
`results/*/scores.json`. 3RScan and MultiScan have no *functionality* questions, so that
column of the overall rows is ScanNet's. Per-domain files:
`extra_eval/results/beacon3d_<domain>/`, overall: `extra_eval/results/beacon3d_overall/`.

### XR-QA / XR-SceneCaption (LSceneLLM, val)

[LSceneLLM](https://arxiv.org/abs/2412.01292)'s large-scene benchmarks: an XR scene is the
union of several SceneVerse HM3D regions (3 for QA, 5 for captions; ~924k points). Grounded
3D-LLM is not in LSceneLLM's Table 1, so there is no paper number to match; the setup
mirrors our Chat-Scene run (same items, same XR-only finetuning, 3 epochs). The detector's
100 queries look at the whole point cloud, so there is no choice of which >100 objects to
keep. Two placements of the regions: **rc** as LSceneLLM's loader (region +
`room_center`; rooms ~0.3–0.9 m off each other) and **glb** the real house layout (offsets
registered against the HM3D v0.2 meshes, ~1 cm). Scores are CIDEr / ROUGE-L / METEOR with
LSceneLLM's capeval, keyed by its episode ids (2,981 QA keys, 283 captions), on text
normalised as the repository does (`special_token_filter`, + LEO's `clean_answer` for QA,
PTB tokenizer).

Region normals come from the HM3D meshes (see **Normals** below); QA CIDEr of the first run
(PCA normals) in brackets.

| Setting | XR-QA | XR-SceneCaption | Caption, `g3d` (256-char cut) |
| --- | --- | --- | --- |
| zero-shot, rc | 79.08 / 28.25 / 9.63 (78.44) | 1.86 / 17.70 / 8.90 | 8.12 / 19.16 / 8.72 |
| zero-shot, glb | 77.61 / 27.91 / 9.43 (77.67) | 2.22 / 17.91 / 8.77 | 8.16 / 19.43 / 8.63 |
| finetuned, rc, epoch 3 | **126.27 / 38.49 / 14.92** (126.94) | 6.20 / 24.62 / 14.00 | 27.27 / 27.39 / 13.39 |
| finetuned, glb, epoch 3 | 125.94 / 38.31 / 14.82 (125.21) | 5.53 / 24.52 / 13.89 | 26.49 / 27.06 / 13.45 |

- **Captions: quote the `g3d_full` column (the middle one).** `special_token_filter` cuts
  predictions *and* references to 256 characters. Harmless for ScanQA / Scan2Cap, but XR
  captions average 87 words and the cut keeps only their formulaic openings ("This
  multi-room scene consists of …"), inflating CIDEr ~5×. `score_xr.py` reports both
  (`g3d`, `g3d_full`; identical for QA) and LSceneLLM's raw-text scoring (`raw`).
- **Finetuning** (`finetune_xr.py`) follows the released step-3 recipe where it can:
  detector frozen, the `train_layer_list` parameters trained (LoRA r64 on q/v, query
  projections, embeddings, lm_head; 157.7 M), its loss, AdamW (wd 1e-4), OneCycle, gradient
  clipping 0.1. Data: XR-QA train (17,881) + XR-SceneCaption train (1,559 captions), one
  item per answer; batch 32, peak lr 2e-4 (not tuned), 3 epochs (~7 min each on one GPU).
  The detector's queries are computed once per scene without training augmentations.
- **Text normalisation moves QA scores a lot**: under LSceneLLM's raw-text scoring the
  finetuned rc model gets 115.82 CIDEr, about what our Chat-Scene finetune gets (116.11),
  while each model's own normalisation gives 126.27 vs ~98.
- **Normals.** The detector does not read normals (`add_normals: false`); they only shape
  the superpoints (ScanNet Segmentator) its masks are pooled over. `preprocess_xr.py` takes
  them from the HM3D v0.2 meshes (`NORMALS=glb`, the default): regions are moved into the
  glb frame by `region_offsets.json` and each point gets the normal of its nearest triangle;
  points > 5 cm from the mesh fall back to the kNN estimate (0.11% of the points; median
  point-to-mesh distance > 1 cm in 5 regions, `normals_stats.json`, 1,905 of 2,035 regions
  recorded). The first run (`NORMALS=pca`, Open3D kNN normals oriented to the region
  centre, wrong side for 5–25% of the points) is kept in
  `extra_eval/results/xr/pca_normals/`. Every score moved by ≤ 2.2 CIDEr (QA ≤ 0.7 after
  finetuning), within run-to-run spread.
- Placement barely matters: rc vs glb differ by ≤ 1.5 CIDEr on QA, with no consistent
  order on captions. Finetuned QA on scenes with > 100 SceneVerse objects is 1–2 CIDEr
  lower than on the rest.

Per-epoch predictions and scores (all, ≤ 100 / > 100 objects, three normalisations):
`extra_eval/results/xr/{zeroshot,finetune_rc/epoch<k>,finetune_glb/epoch<k>}/`, summary in
`extra_eval/results/xr/summary.json`; the PCA-normal run in `extra_eval/results/xr/pca_normals/`.

### Read this before quoting the numbers

- **The four paper benchmarks follow the repository's test pass, including its quirks.**
  Questions are drawn from the release's instruction templates (`random.choice`, seeded per
  scene here so runs repeat). Not every val item is asked: the detector's text input holds
  the scene's ScanRefer / Multi3DRefer sentences up to 48 × 256 tokens, and items whose
  target is a wall or floor are skipped; the tables count asked items, as the paper's code
  does. Mini-batches are right-padded, so an item's answer can depend on its batch-mates;
  batches are formed exactly as upstream. The detection prompts of the test pass (198 per
  scene, not a benchmark asked for here) are left out. SPICE is not computed. Scan2Cap is
  scored as `evaluate_LLM.py` does (one caption per object, box IoU of the object's
  best-matching proposal); `evaluate_LLM.py` keys objects by `scene0011`, not
  `scene0011_00`, which could merge two scans of one scene, so `score_native.py` also scores
  with full scene ids. On ScanRefer val the two are identical.

- **SQA3D and Real-3DQA are zero-shot for this model.** Grounded 3D-LLM was never trained
  on SQA3D (its instruction data includes ScanQA, not SQA3D), whereas most models in SQA3D
  tables were. Questions are asked with the model's own ScanQA prompt, with the situation
  sentence in front of the question.
- **Beacon3D grounding goes through predicted masks.** Beacon3D's leaderboard protocol
  lets a model pick one of the GT objects (e.g. Chat-Scene with GT-mask object tokens).
  Grounded 3D-LLM can only point at one of its own 100 detector proposals, so its pick is
  mapped to the GT object with the highest point IoU (first row: any overlap; second row:
  also IoU ≥ 0.25). This is the closest equivalent, not the same protocol.
- **Beacon3D QA is judged by GPT-4o-mini** (`gpt-4o-mini-2024-07-18`), not the upstream
  default `gpt-4o-2024-11-20`, so the GPT-scored columns are not comparable with numbers
  judged by GPT-4o (e.g. the Beacon3D leaderboard). EM / EM-R and grounding do not involve
  GPT. The judge is stochastic: two passes over the same ScanNet predictions gave 44.57
  and 44.74.
- **3RScan / MultiScan run under different input conditions than ScanNet.** They come as
  SceneVerse point clouds without meshes, so (a) the superpoints the detector pools over
  are rebuilt with ScanNet Segmentator's algorithm on a kNN graph instead of mesh edges
  (0.9–1.2k per scene; 93–96% instance-pure on ScanNet scene0011_00, vs 1,611 official
  ones), with estimated normals for MultiScan,
  and (b) the detector's text input is the ScanNet200 class names only, since no
  ScanRefer / Multi3DRefer sentences exist for those scenes. The *ScanNet control* applies
  exactly (a) + (b) to ScanNet: grounding drops by 2.6 points (57.37 → 54.75) and QA is
  unchanged, so most of the gap to 3RScan / MultiScan is the domain shift itself (the
  detector is trained on ScanNet only), not the substitutes.

## Repository layout

| Path | What |
| --- | --- |
| `extra_eval/build_env.sh`, `requirements.txt`, `overrides.txt`, `minkowski_cuda12.patch` | Builds the runtime as a uv venv for current GPUs (see [Environment](#environment)). |
| `extra_eval/download.sh` | Checkpoint, BERT, CLIP label features and ScanRefer / Multi3DRefer / ScanQA language data from the HF release, plus Tiny-Vicuna-1B; links them to `./saved`, `./pretrained`, `./data/langdata`. |
| `extra_eval/preprocess_scannet.py`, `.sh` | ScanNet val scans → the Mask3D-style `data/processed/scannet200` the code reads; links `data/rawscannet` for the axis-alignment matrices. |
| `extra_eval/preprocess_pointcloud.py`, `.sh` | 3RScan / MultiScan point clouds (and the ScanNet control) → the same layout, with kNN superpoints; see the 3RScan / MultiScan caveat above. |
| `extra_eval/infer.py`, `run_infer.sh` | Inference on all seven benchmarks, one scene at a time (one process per GPU, scenes sharded); `DOMAIN=3rscan|multiscan|scannet_pcseg` for the other Beacon3D domains. |
| `extra_eval/score.py`, `score_native.py`, `score.sh` | Scoring with the official scripts / the repository's metric code; writes `extra_eval/results/`. |
| `models/LLM/LLama3d.py`, `modeling_llama.py` | Upstream, plus the opt-in per-scene prefix cache (`_use_scene_prefix_cache` and the first-step branch of `prepare_inputs_for_generation`). |
| `extra_eval/real3dqa/` | The two official Real-3DQA evaluators, vendored unmodified (CC BY-SA 4.0, see `VENDORED.md`). |
| `extra_eval/beacon3d/Beacon3D/` | Vendored Beacon3D repo at `bda1203` (Apache-2.0): evaluators, `utils.py`, annotations, metadata and GPT system prompt. Unmodified; see `VENDORED.md`. `beacon_grounding_shim.py` only skips knowledge tags with zero cases. |
| `extra_eval/xr/` | XR-QA / XR-SceneCaption: `preprocess_xr.py` (SceneVerse HM3D regions → scenes, both placements; `region_offsets.json` = glb offsets), `run_xr.sh` (zero-shot), `finetune_xr.py` + `run_xr_finetune.sh`, `score_xr.py`; `capeval/` vendored from LSceneLLM (see `VENDORED.md`). |
| `extra_eval/results/` | Per-question predictions, the official evaluators' reports and `summary.json` for the run in the tables. |

## Setup

### External dependencies (not in this repo)

| Needed for | What | Where | Env var |
| --- | --- | --- | --- |
| inference | release checkpoint, `pretrained/` (BERT, CLIP label features), `langdata/` | [HF `chenyilun95/Grounded_3D-LLM_data`](https://huggingface.co/datasets/chenyilun95/Grounded_3D-LLM_data) (`download.sh`) | `G3D_ROOT` |
| inference | Tiny-Vicuna-1B (`pytorch_model.bin`; the code `torch.load`s it) | [HF `Jiayi-Pan/Tiny-Vicuna-1B`](https://huggingface.co/Jiayi-Pan/Tiny-Vicuna-1B) (`download.sh`) | `G3D_ROOT` |
| inference | ScanNet val scans: `<scene>_vh_clean_2.ply`, `_vh_clean_2.0.010000.segs.json`, `.aggregation.json`, `.txt` | ScanNet (ToS) | `SCANNET_SCANS` |
| inference | `scannetv2-labels.combined.tsv` | ScanNet | `SCANNET_LABEL_TSV` |
| SQA3D | `balanced/v1_balanced_{questions,sqa_annotations}_test_scannetv2.json` | `sqa_task.zip` on [Zenodo 7792397](https://zenodo.org/record/7792397) | `SQA3D_DIR` |
| Real-3DQA | `debiased_test.jsonl`, `rotation_{0,90,180,270}.jsonl` | [HF `Oliver-Ma/Real-3DQA`](https://huggingface.co/datasets/Oliver-Ma/Real-3DQA) `data/test/` | `REAL3DQA_DIR` |
| Beacon3D 3RScan / MultiScan | SceneVerse `scan_data/{3RScan,MultiScan}/`: `pcd_with_global_alignment/<scene>.pth` (+ `normal/<scene>.npy` for 3RScan) | [SceneVerse](https://scene-verse.github.io/) release | `SCENEVERSE_DIR` |
| XR-QA / XR-SceneCaption | `XR-QA/`, `XR-SceneCaption/` | [HF `Hoyard/XR-Scene`](https://huggingface.co/datasets/Hoyard/XR-Scene) | `XR_DIR` |
| XR-QA / XR-SceneCaption | SceneVerse HM3D `scan_data/` (`pcd_with_global_alignment/`, `instance_id_to_label/`) + `room_center/` | [SceneVerse](https://scene-verse.github.io/), [HF `Hoyard/HM3D_SceneVerse_RoomCenter`](https://huggingface.co/datasets/Hoyard/HM3D_SceneVerse_RoomCenter) | `XR_SCAN_DATA` |
| XR-QA / XR-SceneCaption (normals) | HM3D v0.2 meshes `<scan>/<hash>.glb` | [HM3D](https://aihabitat.org/datasets/hm3d/) | `HM3D_GLB` |
| Beacon3D QA scoring | an OpenAI API key | | `OPENAI_API_KEY` |
| Scan2Cap / ScanQA scoring | Java ≥ 8 on `PATH` (pycocoevalcap's PTB tokenizer and METEOR) | | `JAVA_HOME` |

Keep a private copy of all of these: `env.sh.example` points every variable into a
gitignored `deps/` in the repo (symlinks into storage only this checkout uses, hard links or
copies otherwise), so cleaning up a shared data directory cannot break a re-run.

The **official** `.segs.json` is required: the aggregation's object masks are lists of its
segment ids, and re-running ScanNet's Segmentator with the same parameters does not
reproduce them (~25% of ids matched on `scene0011_00`). The run in the tables used 310 of
the 312 val scenes (`scene0011_01`, `scene0217_00` were unavailable; no benchmark
question uses them).

### Environment

The release pins torch 1.12.1+cu116, which has no kernels for Blackwell GPUs (sm_120).
`build_env.sh` builds a uv venv with torch 2.7.1+cu128 and compiles:

- MinkowskiEngine at the authors' commit `02fc608` + `minkowski_cuda12.patch` (missing
  thrust includes; its vendored NVTX3 header clashes with the one bundled in CUDA 12.x; a
  `shared_ptr`-from-`unique_ptr` conversion that is ambiguous with `cuda::std` under GCC 13);
- the repo's pointnet2 ops (`.data<T>()` → `.data_ptr<T>()`).

Python deps are in `extra_eval/requirements.txt` (mmcv-full 1.5 → mmcv-lite 2.1,
mmengine 0.10.7, pytorch-lightning 1.9.5, transformers 4.36.1, peft 0.8.2). It needs a
CUDA **12.x** toolkit ≥ 12.8 (`G3D_CUDA_HOME`; a 13.x toolkit does not match torch's
cu128) and an OpenBLAS with `cblas.h`. The checkpoint loads with 0 missing / 0
unexpected keys.

```shell
cp env.sh.example env.sh        # fill in paths (+ OPENAI_API_KEY); env.sh is gitignored
bash extra_eval/build_env.sh    # GPU node with internet access; -> $G3D_ENV
bash extra_eval/download.sh
bash extra_eval/preprocess_scannet.sh
bash extra_eval/preprocess_pointcloud.sh   # Beacon3D 3RScan / MultiScan + ScanNet control
```

Every script in `extra_eval/*.sh` sources `env.sh` from the repo root and reads its paths
from environment variables; the header of each script lists them, and a value given on
the command line overrides `env.sh`. The scripts are plain bash, so on a cluster wrap them
in your own job script.

## Running

```shell
# 1. inference, all seven benchmarks; one process per visible GPU (~20 min on 2 GPUs)
OUT=outputs/extra_eval bash extra_eval/run_infer.sh
#    PREFIX_CACHE=1 to reuse each scene's prompt prefix across its questions (below)
#    BENCHES=scanrefer,m3dref,scan2cap,scanqa,sqa3d,real3dqa,beacon3d to pick a subset
for d in 3rscan multiscan scannet_pcseg; do DOMAIN=$d OUT=outputs/extra_eval bash extra_eval/run_infer.sh; done
#    OUT=outputs/debug MAX_SCENES=2 BENCHES=sqa3d bash extra_eval/run_infer.sh   # smoke test

# 2. score; Beacon3D QA calls the GPT judge (QA_MODEL, default gpt-4o-mini-2024-07-18)
PREDS_DIR=outputs/extra_eval bash extra_eval/score.sh
#    -> extra_eval/results/{scanrefer,m3dref,scan2cap,scanqa,sqa3d,real3dqa,beacon3d_<domain>,beacon3d_overall}/
#       and summary.json (merged: benchmarks not rescored keep their entry)
#    RESULT_DIR to write elsewhere; BENCHES / BEACON_DOMAINS to rescore a subset
```

XR-QA / XR-SceneCaption (CPU preprocessing ~45 min for train + val, then GPU):

```shell
SPLITS="train val" bash extra_eval/xr/preprocess_xr.sh   # -> $XR_PC (mesh normals: HM3D_GLB)
bash extra_eval/xr/run_xr.sh                              # zero-shot, rc + glb, scored (~13 min on 2 GPUs)
PLACEMENT=rc bash extra_eval/xr/run_xr_finetune.sh        # cache detector queries, finetune, score every epoch (~1 h)
```

**Prefix cache (`PREFIX_CACHE=1`).** All questions about a scene are asked in one pass over
that scene, and every prompt starts with the same 140 tokens: the 100 object queries, the
system prompt and `USER:`. With the cache, that prefix is encoded once per scene and its
keys / values are reused by every mini-batch; batches, padding, attention mask and
position ids stay those of upstream. Results differ from the default only through
floating-point noise (the prefix is computed at batch size 1). In the run above, 95% of the
paper-benchmark answers were identical, and 99.5% of SQA3D / Real-3DQA / Beacon3D answers.
Scores moved by ≤ 0.05 (table above). The gain is modest, 1.18× on the LLM part (1,752 →
1,479 GPU-s on the four paper benchmarks), because 3-beam decoding of up to 150 tokens
dominates. The committed results are from the default path.

## How the evaluation is wired, and what to be careful about

- **One pass per scene.** For each scene the detector runs once, then the scene's
  paper-benchmark items (the upstream test pass) and the items of the other benchmarks are
  answered from the same 100 object queries.
- **Same scene features as the paper's test pass.** Per scene, the frozen Mask3D-CLASP
  detector runs exactly as in the TEST block of `final_scripts/step3_train_grounded3dllm.sh`
  (`data=indoor_dialog`, `model=mask3d_lang`, `conf/llm/tiny_vicuna_len512.json`,
  axis-aligned scans). Its BiFormer also reads text — the ScanNet200 class names and the
  scene's ScanRefer / Multi3DRefer sentences — and that input is kept identical, so the
  100 object queries the LLM sees are those behind the paper's ScanQA numbers. Only the
  LLM-side instruction lists differ: `data.lang_data_conf=scanrefer+m3dref` (+`scanqa`,
  `scan2cap` when those benchmarks are asked), which changes only what the LLM is asked,
  not the detector's text; SQA3D / Real-3DQA / Beacon3D questions are added on top.
- **Decoding** is the official `llama_model.evaluate`: beam 3, repetition penalty 1.2,
  ≤ 150 new tokens, mini-batches of 50. Identical (task, text) pairs within a scene are
  generated once (the Real-3DQA splits overlap with each other and with SQA3D).
- **Prompts.** QA: `<situation> <question> Please answer with a single word or phrase.`
  (`lang_type` `scanqa:…:text_only`; no situation for Beacon3D). Grounding: the first
  ScanRefer template, `Can you help me find what's described here: <utterance>?` +
  ` (with grounding)` (`scanrefer:…:with_grounding`). The first `<ref>` token is matched to
  a proposal as in `LLama3d.evaluate`; every one of the 1,611 Beacon3D items produced one.
- **Empty generations** (the model emitting only `</s>`; 46 on SQA3D, 25 on the Real-3DQA
  debiased set, 10 on Beacon3D QA) are scored as wrong. Under the refined / relaxed match
  an empty string is a substring of every answer, so the official Real-3DQA scripts would
  count it as correct; `score.py` replaces it by `<empty>` first, as the repo's own
  `evaluate_LLM.py` does with `set_zero_as_error`.
- **Real-3DQA ids are written as integers.** The official rotation script `int()`s the
  suffix-stripped ids of the 90° / 180° / 270° files but not the 0° ones, so with the
  released string ids no question would intersect.
- **`evaluate_qa.py` caches.** Beacon3D's QA evaluator silently reuses an existing
  processed output instead of re-judging; `score.py` deletes it before every call.
- **Beacon3D metrics are re-derived.** `score.py` runs the official evaluators and then
  recomputes every number from their processed files (to pool domains and to get the chain
  analysis for the IoU ≥ 0.25 variant without a second GPT pass); it asserts that its
  per-domain case scores equal the official printout. QA is judged once per domain.
- **Colour normalisation.** `data/processed/scannet200/color_mean_std.yaml` must hold the
  ScanNet200 *train*-split statistics; they are taken from Mask3D's inference code
  (`cvg/Mask3D`, `mask3d/__init__.py`), so the train scans are not needed.
- **Loading the checkpoint.** It is a Lightning checkpoint with pickled OmegaConf
  hyper-parameters, which torch ≥ 2.6 refuses under `weights_only=True`; `infer.py` loads
  this one file with `weights_only=False`.

## Citation

This fork evaluates the model of

```BibTeX
@article{chen2024grounded3dllm,
      title={Grounded 3D-LLM with Referent Tokens},
      author={Chen, Yilun and Yang, Shuai and Huang, Haifeng and Wang, Tai and Lyu, Ruiyuan and Xu, Runsen and Lin, Dahua and Pang, Jiangmiao},
      journal={arXiv preprint arXiv:2405.10370},
      year={2024},
}
```

using the data and evaluators of

```BibTeX
@inproceedings{huang2025unveiling,
  title={Unveiling the Mist over 3D Vision-Language Understanding: Object-centric Evaluation with Chain-of-Analysis},
  author={Huang, Jiangyong and Jia, Baoxiong and Wang, Yan and Zhu, Ziyu and Linghu, Xiongkun and Li, Qing and Zhu, Song-Chun and Huang, Siyuan},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2025}
}

@inproceedings{ma2022sqa3d,
  title={SQA3D: Situated Question Answering in 3D Scenes},
  author={Ma, Xiaojian and Yong, Silong and Zheng, Zilong and Li, Qing and Liang, Yitao and Zhu, Song-Chun and Huang, Siyuan},
  booktitle={International Conference on Learning Representations},
  year={2023},
  url={https://openreview.net/forum?id=IDJx97BC38}
}

@inproceedings{ma2026real3dqa,
  title={Do 3D Large Language Models Really Understand 3D Spatial Relationships?},
  author={Xianzheng Ma and Tao Sun and Shuai Chen and Yash Bhalgat and Jindong Gu and Angel X Chang and Iro Armeni and Iro Laina and Songyou Peng and Victor Adrian Prisacariu},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026},
  url={https://arxiv.org/abs/2603.23523}
}
```

## License

The upstream repository ships no license file. The vendored Beacon3D evaluators and data
under `extra_eval/beacon3d/Beacon3D/` are Apache-2.0 (`LICENSE` there); the vendored
Real-3DQA evaluators under `extra_eval/real3dqa/` are CC BY-SA 4.0
([xianzhengma/Real-3DQA](https://github.com/xianzhengma/Real-3DQA)). The predictions in
`extra_eval/results/` are derived from ScanNet, SQA3D, Real-3DQA and Beacon3D and are
subject to their terms.
