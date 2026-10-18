# Grounded 3D-LLM Benchmark Evaluation

This is a fork of [Grounded 3D-LLM](https://github.com/OpenRobotLab/Grounded_3D-LLM)
(arXiv 2405.10370) whose purpose is to **evaluate the released Grounded 3D-LLM checkpoint
on benchmarks the paper does not report**, under one reproducible setup, so its scores can
be compared with other models on equal footing. Wired in so far:
[SQA3D](https://sqa3d.github.io/) (ICLR 2023), [Real-3DQA](https://real-3dqa.github.io/)
(ICLR 2026, a debiased SQA3D subset with viewpoint rotations) and
[Beacon3D](https://beacon-3d.github.io/) (CVPR 2025, ScanNet). The upstream model,
training and data code are untouched apart from a torch-2 compatibility fix in the
pointnet2 ops; everything this fork adds lives under [`extra_eval/`](extra_eval/), and
the per-question outputs behind the numbers below are committed in
[`extra_eval/results/`](extra_eval/results/).

Upstream Grounded 3D-LLM's own README is in the git history (`git show 9de50f1:README.md`).

## Results

Released checkpoint `saved/step3_mask3d_lang_4GPUS/last-epoch.ckpt`, ScanNet val, no
training or fine-tuning of any kind. All numbers are percentages.

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

### Beacon3D, ScanNet

*Case* = per-question accuracy; *Obj* = all questions about an object answered correctly.

| Grounding variant | Case | Class | App. | Geo. | Spa. | Func. | Chain | Obj |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| proposal → best-IoU GT object | 57.37 | 57.37 | 50.92 | 49.32 | 52.05 | 11.11 | 50.22 | 44.42 |
| … and IoU ≥ 0.25 | 56.35 | 56.35 | 49.69 | 47.49 | 51.03 | 11.11 | 49.35 | 43.64 |

| QA (judge gpt-4o-mini-2024-07-18) | EM | EM-R | Case (GPT) | Obj | Class | App. | Geo. | Spa. | Exist. | Func. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| | 41.21 | 43.90 | 44.57 | 8.05 | 18.24 | 40.69 | 46.99 | 51.07 | 52.24 | 33.33 |

Grounding→QA chain (against the first grounding row): R1 37.85, R2 54.10.

### Read this before quoting the numbers

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
  GPT. The judge is stochastic: two passes over the same predictions gave 44.57 and 44.74.
- **Beacon3D 3RScan / MultiScan are not run.** The released detector is trained on ScanNet
  meshes and pools over mesh superpoints, which those point-cloud releases do not provide.

## Repository layout

| Path | What |
| --- | --- |
| `extra_eval/build_env.sh`, `requirements.txt`, `overrides.txt`, `minkowski_cuda12.patch` | Builds the runtime as a uv venv for current GPUs (see [Environment](#environment)). |
| `extra_eval/download.sh` | Checkpoint, BERT, CLIP label features and ScanRefer / Multi3DRefer / ScanQA language data from the HF release, plus Tiny-Vicuna-1B; links them to `./saved`, `./pretrained`, `./data/langdata`. |
| `extra_eval/preprocess_scannet.py`, `.sh` | ScanNet val scans → the Mask3D-style `data/processed/scannet200` the code reads; links `data/rawscannet` for the axis-alignment matrices. |
| `extra_eval/infer.py`, `run_infer.sh` | Inference on all three benchmarks (one process per GPU, scenes sharded). |
| `extra_eval/score.py`, `score.sh` | Scoring with the official scripts; writes `extra_eval/results/`. |
| `extra_eval/real3dqa/` | The two official Real-3DQA evaluators, vendored unmodified (CC BY-SA 4.0, see `VENDORED.md`). |
| `extra_eval/beacon3d/Beacon3D/` | Vendored Beacon3D repo at `bda1203` (Apache-2.0): evaluators, `utils.py`, annotations, metadata and GPT system prompt. Unmodified; see `VENDORED.md`. `beacon_grounding_shim.py` only skips knowledge tags with zero cases. |
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
| Beacon3D QA scoring | an OpenAI API key | | `OPENAI_API_KEY` |

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
```

Every script in `extra_eval/*.sh` sources `env.sh` from the repo root and reads its paths
from environment variables; the header of each script lists them, and a value given on
the command line overrides `env.sh`. The scripts are plain bash, so on a cluster wrap them
in your own job script.

## Running

```shell
# 1. inference, all three benchmarks; one process per visible GPU (~5 min on 2 GPUs)
OUT=outputs/extra_eval bash extra_eval/run_infer.sh
#    OUT=outputs/debug MAX_SCENES=2 BENCHES=sqa3d bash extra_eval/run_infer.sh   # smoke test

# 2. score; Beacon3D QA calls the GPT judge (QA_MODEL, default gpt-4o-mini-2024-07-18)
PREDS_DIR=outputs/extra_eval bash extra_eval/score.sh
#    -> extra_eval/results/{sqa3d,real3dqa,beacon3d_scannet}/, summary.json; RESULT_DIR to write elsewhere
```

## How the evaluation is wired, and what to be careful about

- **Same scene features as the paper's test pass.** Per scene, the frozen Mask3D-CLASP
  detector runs exactly as in the TEST block of `final_scripts/step3_train_grounded3dllm.sh`
  (`data=indoor_dialog`, `model=mask3d_lang`, `conf/llm/tiny_vicuna_len512.json`,
  axis-aligned scans). Its BiFormer also reads text — the ScanNet200 class names and the
  scene's ScanRefer / Multi3DRefer sentences — and that input is kept identical, so the
  100 object queries the LLM sees are those behind the paper's ScanQA numbers. Only the
  LLM-side instruction lists (ScanQA, object descriptions, …) are replaced by our
  questions (`data.lang_data_conf=scanrefer+m3dref`).
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
```

## License

The upstream repository ships no license file. The vendored Beacon3D evaluators and data
under `extra_eval/beacon3d/Beacon3D/` are Apache-2.0 (`LICENSE` there); the vendored
Real-3DQA evaluators under `extra_eval/real3dqa/` are CC BY-SA 4.0
([xianzhengma/Real-3DQA](https://github.com/xianzhengma/Real-3DQA)). The predictions in
`extra_eval/results/` are derived from ScanNet, SQA3D, Real-3DQA and Beacon3D and are
subject to their terms.
