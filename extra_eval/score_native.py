"""Score the native benchmarks of ``extra_eval/infer.py`` (``<pred_dir>/native/<scene>.json``,
the per-scene files of the upstream test pass) with the upstream metrics of
``models/metrics/evaluate_LLM.py`` / ``multi3drefer_evaluator.py``:

* ScanRefer: Acc@0.25 / Acc@0.5 of the top-1 box (``top_1_bbox_iou`` > threshold), overall /
  unique / multiple.
* Multi3DRefer: F1@0.25 / F1@0.5 from the official ``Multi3DReferEvaluator`` on the saved
  boxes, overall and per eval type.
* ScanQA: BLEU-1..4 / METEOR / ROUGE-L / CIDEr (pycocoevalcap) plus EM and refined EM, on the
  cleaned answers, with evaluate_LLM.py's choice of one prediction per question.
* Scan2Cap: the same captioning metrics @0.25 / @0.5 IoU (one caption per object, box IoU of
  the object's best-matching proposal; below the threshold the caption counts as empty).

Two deliberate differences from running ``evaluate_LLM.py`` itself, both reported:
SPICE is skipped (it needs Stanford CoreNLP downloads; no benchmark table uses it), and
Scan2Cap is also scored with the object key ``<scene_id>_<object_id>``. ``evaluate_LLM.py``
derives the scene from the file name with ``path.split("_")[-2]``, i.e. ``scene0011`` for
``scene0011_00`` and ``scene0011_01``, so objects with the same id in two scans of one scene
share a key (the first caption is kept, the references are pooled); the ``upstream_key``
numbers reproduce that.
"""

import glob
import json
import os
import pickle
from collections import defaultdict
from contextlib import redirect_stdout
from io import StringIO

import numpy as np

SCANREFER_TYPES = ("unique", "multiple")
M3DREF_TYPES = ("zt_w_d", "zt_wo_d", "st_w_d", "st_wo_d", "mt")


def load_native(pred_dir):
    rows = {}
    for p in sorted(glob.glob(os.path.join(pred_dir, "native", "scene*.json"))):
        rows[os.path.basename(p)[:-len(".json")]] = json.load(open(p))["prediction"]
    return rows


def caption_scores(gts, preds):
    """pycocoevalcap BLEU / METEOR / ROUGE-L / CIDEr as in Evaluator.evaluate, without SPICE."""
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.meteor.meteor import Meteor
    from pycocoevalcap.rouge.rouge import Rouge
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    from models.metrics.evaluate_LLM import Evaluator
    tok = PTBTokenizer()
    ref = tok.tokenize(Evaluator.to_coco(gts, gts.keys()))
    hyp = tok.tokenize(Evaluator.to_coco(preds, gts.keys()))
    out = {}
    for scorer, names in [(Bleu(4), ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4"]), (Meteor(), "METEOR"),
                          (Rouge(), "ROUGE-L"), (Cider(), "CIDEr")]:
        score, _ = scorer.compute_score(ref, hyp)
        for m, s in (zip(names, score) if isinstance(names, list) else [(names, score)]):
            out[m] = 100 * float(s)
    return out


def score_scanrefer(rows, langdata):
    ious = defaultdict(list)
    for preds in rows.values():
        for p in preds:
            if p["type"].startswith("scanrefer:"):
                ious[p["type"].split(":")[1]].append(p.get("top_1_bbox_iou") or 0.0)
    if not ious:
        return None
    ious["overall"] = ious["unique"] + ious["multiple"]
    n_val = sum(1 for x in langdata["scanrefer"])
    res = {"n": len(ious["overall"]), "n_val": n_val}
    for k in ("overall",) + SCANREFER_TYPES:
        v = np.asarray(ious[k], dtype=float)
        res[k] = {"n": len(v), "Acc@0.25": 100 * float((v > 0.25).mean()),
                  "Acc@0.5": 100 * float((v > 0.5).mean())}
    # sentences never asked (the detector's text input is capped at 48 clips of 256 tokens,
    # and upstream skips targets labelled wall / floor) counted as misses
    v = np.asarray(ious["overall"], dtype=float)
    res["overall_all_val"] = {"Acc@0.25": 100 * float((v > 0.25).sum()) / n_val,
                              "Acc@0.5": 100 * float((v > 0.5).sum()) / n_val}
    return res


def score_m3dref(pred_dir, rows, langdata):
    from models.metrics.multi3drefer_evaluator import Multi3DReferEvaluator, parse_prediction
    if not glob.glob(os.path.join(pred_dir, "native", "m3drefer", "*.pkl")):
        return None, ""
    pred, gt = parse_prediction(os.path.join(pred_dir, "native", "m3drefer"))
    ev = Multi3DReferEvaluator("m3dv", verbose=True)
    ev.set_ground_truths(gt)
    log = StringIO()
    with redirect_stdout(log):
        r = ev.evaluate(pred)
    res = {"n": len(pred), "n_val": len(langdata["m3dref"])}
    for thr, key in (("F1@0.25", "m3dv@0.25"), ("F1@0.5", "m3dv@0.5")):
        res[thr] = {k: 100 * float(v) for k, v in r[key].items()}
    # queries never asked counted as F1 = 0, over the whole val set
    f1 = defaultdict(list)
    for preds in rows.values():
        for p in preds:
            if p["type"].startswith("m3dref:"):
                f1["F1@0.25"].append(p.get("bbox_iou_25_f1_score") or 0.0)
                f1["F1@0.5"].append(p.get("bbox_iou_50_f1_score") or 0.0)
    res["overall_all_val"] = {k: 100 * float(np.sum(v)) / res["n_val"] for k, v in f1.items()}
    return res, log.getvalue()


def score_scanqa(rows):
    from models.metrics.evaluate_LLM import Evaluator
    ev = Evaluator(directory_path=None)
    lan, gt = {}, {}
    for preds in rows.values():
        for p in preds:
            if not (p["type"].startswith("scanqa:") and "text_only" in p["type"]):
                continue
            qid = p["type"].split(":")[1]
            if qid in gt:  # one row per reference answer; evaluate_LLM.py's rule for the prediction
                gt[qid].append(ev.special_token_filter(p["gt"]))
                if len(lan[qid][0]) < 3 or len(lan[qid][0]) > 20:
                    lan[qid] = [ev.special_token_filter(p["output_language"])]
            else:
                lan[qid] = [ev.special_token_filter(p["output_language"])]
                gt[qid] = [ev.special_token_filter(p["gt"])]
    if not gt:
        return None
    res = {"n": len(gt), **caption_scores(gt, lan)}
    res["EM"] = 100 * float(np.mean(ev.refined_EM(lan, gt, not_refine=True)))
    res["EM-R"] = 100 * float(np.mean(ev.refined_EM(lan, gt, False)))
    res["empty_preds"] = sum(1 for v in lan.values() if not v[0].strip())
    return res


def score_scan2cap(rows):
    from models.metrics.evaluate_LLM import Evaluator
    ev = Evaluator(directory_path=None)
    out = {}
    for variant in ("fixed_key", "upstream_key"):
        lan, gt, iou = {}, {}, {}
        for scene, preds in rows.items():
            scene_key = scene if variant == "fixed_key" else scene.split("_")[0]
            for p in preds:
                if not p["type"].startswith("scan2cap:"):
                    continue
                key = f'{scene_key}_{p["gt_inst_ids"]["inst_ids_question"][0][0]}'
                cap = "sos " + ev.special_token_filter(p["output_language"], clean=False) + " eos"
                ref = "sos " + ev.special_token_filter(p["gt"], clean=False) + " eos"
                lan.setdefault(key, [cap])
                gt.setdefault(key, []).append(ref)
                iou.setdefault(key, p["gt_predicted_iou"])
                if p["gt_predicted_iou"] > iou[key]:
                    iou[key] = p["gt_predicted_iou"]
                    lan[key] = [cap]
        if not gt:
            return None
        res = {"n_objects": len(gt)}
        for thr in (0.25, 0.5):
            hyp = {k: (v if iou[k] >= thr else ["sos eos"]) for k, v in lan.items()}
            res[f"@{thr}"] = caption_scores(gt, hyp)
            res[f"@{thr}"]["recall"] = 100 * float(np.mean([iou[k] >= thr for k in gt]))
        out[variant] = res
    return out


def load_langdata(langdata_dir, scenes):
    data = {}
    for k, f in (("scanrefer", "scanrefer_format.json"), ("m3dref", "m3dref_format.json")):
        data[k] = [x for x in json.load(open(os.path.join(langdata_dir, f))) if x["scene_id"] in scenes]
    return data


def score_native(pred_dir, out, val_scenes, langdata_dir="data/langdata", benches=("scanrefer", "m3dref", "scanqa", "scan2cap")):
    rows = load_native(pred_dir)
    if not rows:
        return {}
    langdata = load_langdata(langdata_dir, set(val_scenes))
    summary = {}
    if "scanrefer" in benches:
        summary["scanrefer"] = score_scanrefer(rows, langdata)
    if "m3dref" in benches:
        summary["m3dref"], log = score_m3dref(pred_dir, rows, langdata)
        if log:
            os.makedirs(out / "m3dref", exist_ok=True)
            open(out / "m3dref" / "official_evaluator.txt", "w").write(log)
    if "scanqa" in benches:
        summary["scanqa"] = score_scanqa(rows)
    if "scan2cap" in benches:
        summary["scan2cap"] = score_scan2cap(rows)
    return summary
