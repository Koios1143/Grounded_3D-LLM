#!/usr/bin/env python3
"""Aligned re-scoring of Grounded 3D-LLM's saved predictions (no re-inference).

One text normalisation for every baseline (``aligned_eval.py``, identical in every repo):
here we only load the saved predictions, strip Grounded 3D-LLM's own special tokens and
build ``refs`` / ``preds`` dicts.

Special-token stripping (``strip_special``): the tokens ``special_token_filter`` of
``models/metrics/evaluate_LLM.py`` removes (= ``clean_generation`` of ``extra_eval/infer.py``):
``ASSISTANT:``, newlines, ``<s>``, ``</s>``, ``<unk>``, ``<p>``, ``</p>``, ``<ref>`` (each
replaced by nothing, as upstream), then whitespace collapsed. No 256-character truncation, no ``clean_answer``, no lower-casing.
The model's decoded text is ``<s> ASSISTANT: <answer></s></s>...`` (Vicuna-style template;
grounding phrases are wrapped in ``<p> ... </p> <ref>``), so these are all the template /
special tokens it emits.

Benchmarks:

* Scan2Cap (ScanNet val, native test pass ``<native_dir>/native/<scene>.json``): references
  from the common untruncated corpus ``deps/aligned/scan2cap_val_corpus.json`` (keys
  ``scene|objid``, ScanRefer object ids). Grounded 3D-LLM's own protocol is kept: every
  (object, ScanRefer sentence) pair is one question; per object the caption of the row with
  the highest ``gt_predicted_iou`` (box IoU of the GT object's best-matching proposal; first
  row on ties) is used, and below the IoU threshold (0.25 / 0.5, ``>=``) the prediction is
  ``""``. The model's object ids are its own instance indices (remapped from ScanRefer's in
  the dataset), so they are mapped back to ScanRefer ids through the release's
  ``data/langdata/scanrefer_format.json`` (each row's reference sentence -> object id; the
  intersection over an object's rows must be a single id).
* ScanQA val (same native pass): every question is asked once per reference answer; the
  prediction is chosen with ``evaluate_LLM.py``'s rule (first row, replaced by a later one
  while the filtered prediction has < 3 or > 20 characters; the rule is evaluated on the
  upstream-filtered text so the same row is chosen as in ``score_native.py``, but the chosen
  prediction is passed on only special-token-stripped). References = all saved ``gt``.
* SQA3D test: ``extra_eval/results/sqa3d/preds.json`` (the committed run; ``raw`` is
  re-stripped). EM / EM-R only.
* XR-QA / XR-SceneCaption: ``extra_eval/results/xr/{zeroshot,finetune_rc/epoch<k>,
  finetune_glb/epoch<k>}/xr{qa,cap}_<placement>.jsonl``, keyed per item (``idx``, never
  deduplicated by episode id; ``n_unique_episodes`` recorded). The README headline of the
  finetunes is epoch 3 (the last).

Writes ``extra_eval/results/aligned/<bench>__<variant>.json`` and ``summary.json``.
"""

import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))

SPECIAL = ["ASSISTANT:", "\n", "<s>", "</s>", "<unk>", "<p>", "</p>", "<ref>"]

N_SCAN2CAP_OBJECTS = 2068
N_SCANQA = 4675
N_SQA3D = 3519
N_XR = {"qa": 2994, "cap": 283}
HEADLINE_EPOCH = 3


def strip_special(text):
    for t in SPECIAL:
        text = text.replace(t, "")
    return " ".join(text.split())


def upstream_filter(text):
    """``Evaluator.special_token_filter`` of models/metrics/evaluate_LLM.py (special tokens,
    256-character cut, LEO ``clean_answer`` -- the same rules as aligned_eval's), re-implemented
    because importing ``models`` loads MinkowskiEngine. Only used to pick ScanQA's prediction."""
    from aligned_eval import clean_answer
    for t in SPECIAL + ["<|endoftext|>"]:
        text = text.replace(t, "")
    text = re.sub(r"\s{2,}", " ", text.strip())
    return clean_answer(text[:256])


def rel(p):
    return os.path.relpath(os.path.abspath(p), REPO)


def load_native(native_dir):
    files = sorted(glob.glob(os.path.join(native_dir, "native", "scene*.json")))
    if not files:
        raise SystemExit(f"no native predictions in {native_dir}/native")
    for p in files:
        yield os.path.basename(p)[:-len(".json")], json.load(open(p))["prediction"]


def build_scan2cap(native_dir, corpus):
    lang = defaultdict(set)
    for x in json.load(open(REPO / "data" / "langdata" / "scanrefer_format.json")):
        lang[(x["scene_id"], x["description"])].add(int(x["object_ids"][0][0]))
    cand, cap, iou = {}, {}, {}
    n_rows = 0
    for scene, preds in load_native(native_dir):
        for p in preds:
            if not p["type"].startswith("scan2cap:"):
                continue
            n_rows += 1
            k = (scene, p["gt_inst_ids"]["inst_ids_question"][0][0])
            ids = lang[(scene, p["gt"])]
            cand[k] = cand[k] & ids if k in cand else set(ids)
            c = strip_special(p["output_language"])
            if k not in iou:
                cap[k], iou[k] = c, p["gt_predicted_iou"]
            elif p["gt_predicted_iou"] > iou[k]:
                cap[k], iou[k] = c, p["gt_predicted_iou"]
    bad = {k: v for k, v in cand.items() if len(v) != 1}
    if bad:
        raise SystemExit(f"Scan2Cap: {len(bad)} objects without a unique ScanRefer id, e.g. {list(bad.items())[:3]}")
    key = {k: f"{k[0]}|{next(iter(v))}" for k, v in cand.items()}
    if len(set(key.values())) != len(key):
        raise SystemExit("Scan2Cap: two model objects map to the same corpus key")
    absent = sorted(set(key.values()) - set(corpus))
    missing = sorted(set(corpus) - set(key.values()))
    if absent:
        raise SystemExit(f"Scan2Cap: {len(absent)} objects absent from the corpus, e.g. {absent[:5]}")
    if len(corpus) != N_SCAN2CAP_OBJECTS:
        raise SystemExit(f"Scan2Cap corpus has {len(corpus)} objects, expected {N_SCAN2CAP_OBJECTS}")
    out = {}
    for thr in (0.25, 0.5):
        preds = {ck: "" for ck in corpus}  # objects the model never captioned: empty
        n_matched = 0
        for k, ck in key.items():
            if iou[k] >= thr:
                preds[ck] = cap[k]
                n_matched += 1
        out[f"iou{thr}"] = dict(refs=corpus, preds=preds, info={
            "n_rows": n_rows, "n_model_objects": len(key), "n_corpus_objects_missing": len(missing),
            "n_matched": n_matched, "iou_threshold": thr, "source": rel(os.path.join(native_dir, "native"))})
    return out


def build_scanqa(native_dir):
    pred, sel, refs = {}, {}, {}
    for _, preds in load_native(native_dir):
        for p in preds:
            if not (p["type"].startswith("scanqa:") and "text_only" in p["type"]):
                continue
            qid = p["type"].split(":")[1]
            if qid in refs:  # evaluate_LLM.py's rule for which prediction counts
                refs[qid].append(p["gt"])
                if len(sel[qid]) < 3 or len(sel[qid]) > 20:
                    sel[qid] = upstream_filter(p["output_language"])
                    pred[qid] = strip_special(p["output_language"])
            else:
                refs[qid] = [p["gt"]]
                sel[qid] = upstream_filter(p["output_language"])
                pred[qid] = strip_special(p["output_language"])
    if len(refs) != N_SCANQA:
        raise SystemExit(f"ScanQA: {len(refs)} questions, expected {N_SCANQA}")
    return dict(refs=refs, preds=pred, info={"source": rel(os.path.join(native_dir, "native")),
                                             "n_rows": sum(len(v) for v in refs.values())})


def build_sqa3d():
    src = REPO / "extra_eval" / "results" / "sqa3d" / "preds.json"
    rows = json.load(open(src))
    if len(rows) != N_SQA3D:
        raise SystemExit(f"SQA3D: {len(rows)} rows, expected {N_SQA3D}")
    preds = {r["id"]: strip_special(r["raw"]) for r in rows}
    refs = {r["id"]: list(r["answers"]) for r in rows}
    assert len(preds) == N_SQA3D, "duplicate SQA3D ids"
    existing = json.load(open(src.parent / "scores.json"))
    return dict(refs=refs, preds=preds, info={"source": rel(src),
                                              "existing": {k: existing[k] for k in ("EM", "EM-R")}})


def build_xr():
    root = REPO / "extra_eval" / "results" / "xr"
    out = {}
    files = sorted(glob.glob(str(root / "zeroshot" / "xr*.jsonl")) + glob.glob(str(root / "finetune_*" / "epoch*" / "xr*.jsonl")))
    for f in files:
        parts = Path(f).relative_to(root).parts
        bench, placement = Path(f).stem.split("_")  # xrqa_rc
        task = bench[2:]
        # zeroshot_rc | finetune_rc_epoch3
        variant = f"zeroshot_{placement}" if parts[0] == "zeroshot" else "_".join(parts[:-1])
        rows = [json.loads(l) for l in open(f)]
        rows.sort(key=lambda r: r["idx"])
        if len(rows) != N_XR[task] or len({r["idx"] for r in rows}) != len(rows):
            raise SystemExit(f"{f}: {len(rows)} items, expected {N_XR[task]} distinct")
        info = {"source": rel(f), "n_unique_episodes": len({r["key"] for r in rows}), "placement": placement}
        if parts[0].startswith("finetune"):
            ep = int(parts[1][len("epoch"):])
            info.update(epoch=ep, headline=ep == HEADLINE_EPOCH)
        out[(bench, variant)] = dict(refs={r["idx"]: list(r["answers"]) for r in rows},
                                     preds={r["idx"]: strip_special(r["raw"]) for r in rows}, info=info)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--native_dir", default="outputs/native_nocache",
                    help="infer.py --out_dir of the committed native pass (has native/<scene>.json)")
    ap.add_argument("--corpus", default="deps/aligned/scan2cap_val_corpus.json")
    ap.add_argument("--out", default="extra_eval/results/aligned")
    ap.add_argument("--dry_run", action="store_true", help="build and check inputs only, no scoring")
    args = ap.parse_args()
    native_dir = str((REPO / args.native_dir) if not os.path.isabs(args.native_dir) else Path(args.native_dir))
    corpus = json.load(open(REPO / args.corpus))

    jobs = []  # (bench, variant, kind, modes, data)
    for v, d in build_scan2cap(native_dir, corpus).items():
        jobs.append(("scan2cap", v, "cap", ("s2c",), d))
    jobs.append(("scanqa", "val", "qa", ("clean_ptb", "raw"), build_scanqa(native_dir)))
    jobs.append(("sqa3d", "test", "em", (), build_sqa3d()))
    for (bench, v), d in build_xr().items():
        jobs.append((bench, v, "cap" if bench == "xrcap" else "qa",
                     ("ptb", "raw") if bench == "xrcap" else ("clean_ptb", "raw"), d))

    for bench, v, kind, modes, d in jobs:
        print(f"{bench:9} {v:22} n={len(d['preds']):5} {kind:3} {','.join(modes):14} {d['info']}", flush=True)
    if args.dry_run:
        return

    from aligned_eval import em_scores, score_captions, score_qa
    out = REPO / args.out
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for bench, v, kind, modes, d in jobs:
        if kind == "cap":
            metrics = score_captions(d["refs"], d["preds"], modes)
        elif kind == "qa":
            metrics = score_qa(d["refs"], d["preds"], modes)
        else:
            metrics = {"em": em_scores(d["preds"], d["refs"])}
        res = {"bench": bench, "variant": v, "n": len(d["preds"]), "metrics": metrics, **d["info"]}
        if bench == "sqa3d":
            delta = {k: metrics["em"][k] - d["info"]["existing"][k] for k in ("EM", "EM-R")}
            res["delta_vs_existing"] = delta
            if max(abs(x) for x in delta.values()) > 0.1:
                print(f"WARNING: SQA3D aligned EM differs from the existing one by {delta}")
        json.dump(res, open(out / f"{bench}__{v}.json", "w"), indent=1)
        summary[f"{bench}__{v}"] = res
        print(bench, v, json.dumps(metrics), flush=True)
    json.dump(summary, open(out / "summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
