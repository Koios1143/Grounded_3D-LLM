#!/usr/bin/env python3
"""Score ``extra_eval/infer.py`` outputs: SQA3D EM / EM-R, Real-3DQA (official scripts),
Beacon3D (official evaluators). CPU only.

* SQA3D: EM and EM-R (refined EM) with LEO's ``clean_answer`` / ``answer_match`` (the copy
  vendored with the Real-3DQA scorers), overall and per question type (first word:
  What / Is / How / Can / Which / Others), as in SQA3D / LEO / Chat-Scene.
* Real-3DQA: ``evaluate_debiased_exact_match.py`` on the debiased set (EM, Refined EM) and
  ``evaluate_rotation_robustness.py`` on the four rotation splits (One..Four, VRS), plus
  per-rotation EM / Refined EM.
* Beacon3D: ``infer_grounding.json`` (``correct`` = predicted object is a target) and
  ``infer_qa.json`` (``pred``), scored by Beacon3D's ``evaluate_grounding.py`` (through
  ``beacon_grounding_shim.py``) and ``evaluate_qa.py`` (GPT-4o, needs OPENAI_API_KEY).

Writes ``<result_dir>/...`` and a ``summary.json``.
"""

import argparse
import glob
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent


def rel(text):
    """Logs of the evaluators print absolute paths; keep them repo-relative."""
    return text.replace(str(REPO) + "/", "")
sys.path.insert(0, str(HERE / "real3dqa"))
from evaluate_debiased_exact_match import answer_match, clean_answer  # noqa: E402


EMPTY = "<empty>"  # stands in for an empty generation, so it cannot substring-match


def load(pred_dir, bench):
    """Rows of one benchmark (all shards). An empty generation (the model emitted only
    ``</s>``) would be a substring of every answer under the refined/relaxed match, so it
    is replaced by EMPTY, as ``set_zero_as_error`` does in the repo's evaluate_LLM.py."""
    rows = []
    for p in sorted(glob.glob(os.path.join(pred_dir, f"{bench}.jsonl")) +
                    glob.glob(os.path.join(pred_dir, f"{bench}.shard*.jsonl"))):
        rows += [json.loads(l) for l in open(p)]
    for r in rows:
        r["empty_pred"] = not clean_answer(r["pred"]).strip()
        if r["empty_pred"]:
            r["pred"] = EMPTY
    return rows


def em_scores(rows):
    em = emr = 0
    for r in rows:
        e, er = answer_match(clean_answer(r["pred"]), [clean_answer(a) for a in r["answers"]])
        em, emr = em + e, emr + er
    n = max(len(rows), 1)
    return {"n": len(rows), "EM": 100 * em / n, "EM-R": 100 * emr / n,
            "empty_preds": sum(r.get("empty_pred", False) for r in rows)}


def sqa3d_type(q):
    w = q.strip().split(" ")[0].lower()
    return {"what": "What", "is": "Is", "how": "How", "can": "Can", "which": "Which"}.get(w, "Others")


def score_sqa3d(pred_dir, out):
    rows = load(pred_dir, "sqa3d")
    if not rows:
        return None
    assert len(rows) == 3519, f"SQA3D test has 3519 questions, got {len(rows)}"
    res = em_scores(rows)
    by = defaultdict(list)
    for r in rows:
        by[sqa3d_type(r["question"])].append(r)
    res["by_type"] = {k: em_scores(v) for k, v in sorted(by.items())}
    json.dump([{k: r[k] for k in ["id", "scene_id", "question", "answers", "pred", "raw"]} for r in rows],
              open(out / "sqa3d" / "preds.json", "w"), indent=1)
    json.dump(res, open(out / "sqa3d" / "scores.json", "w"), indent=1)
    return res


def score_real3dqa(pred_dir, out):
    res, files = {}, {}
    expected = {"debiased_test": 1485, "rotation_0": 750, "rotation_90": 750,
                "rotation_180": 750, "rotation_270": 750}
    for split, n in expected.items():
        rows = load(pred_dir, f"real3dqa_{split}")
        if not rows:
            return None
        assert len(rows) == n, f"{split}: expected {n}, got {len(rows)}"
        res.setdefault("empty_preds", {})[split] = sum(r["empty_pred"] for r in rows)
        # int ids: the official rotation script int()s the suffix-stripped 90/180/270 ids
        # but not the 0-degree ones, so the released string ids would never intersect
        preds = [{"question_id": int(r["id"]), "response_gt": r["answers"], "response_pred": r["pred"],
                  "question_type": r["question_type"], "scene_id": r["scene_id"]} for r in rows]
        files[split] = out / "real3dqa" / f"pred_{split}.json"
        json.dump(preds, open(files[split], "w"), indent=1)
    sys.path.insert(0, str(HERE / "real3dqa"))
    import evaluate_debiased_exact_match as dem
    import evaluate_rotation_robustness as rot
    res["debiased_test"] = dem.evaluate_em(str(files["debiased_test"]))
    for split in ["rotation_0", "rotation_90", "rotation_180", "rotation_270"]:
        res[split] = dem.evaluate_em(str(files[split]))
    res["rotation"] = rot.quick_consistency_score(
        [str(files[s]) for s in ["rotation_0", "rotation_90", "rotation_180", "rotation_270"]])
    # the same numbers through the unmodified scripts' CLIs, for the record
    log = []
    for cmd in [[sys.executable, str(HERE / "real3dqa" / "evaluate_debiased_exact_match.py"),
                 str(files["debiased_test"]), "--name", "Grounded 3D-LLM"],
                [sys.executable, str(HERE / "real3dqa" / "evaluate_rotation_robustness.py")]
                + [str(files[s]) for s in ["rotation_0", "rotation_90", "rotation_180", "rotation_270"]]
                + ["--name", "Grounded 3D-LLM"]]:
        log.append(subprocess.run(cmd, capture_output=True, text=True, check=True).stdout)
    (out / "real3dqa" / "official_scripts.txt").write_text(rel("\n".join(log)))
    json.dump(res, open(out / "real3dqa" / "scores.json", "w"), indent=1)
    return res


def score_beacon3d(pred_dir, out, beacon_root, qa_model, domain="scannet"):
    res = {}
    bdir = out / f"beacon3d_{domain}"
    # G3D-LLM grounds to its own proposals: the proposal is mapped to the GT object of
    # highest point IoU. "" accepts any overlap (closest to picking an object, as GT-mask
    # models do); "_iou25" also needs IoU >= 0.25 (ScanRefer Acc@0.25 style).
    variants = {"": 0.0, "_iou25": 0.25}
    for task in ["grounding", "qa"]:
        rows = load(pred_dir, f"beacon3d_{domain}_{task}")
        if not rows:
            return None
        src = json.load(open(os.path.join(beacon_root, "data", domain, f"{domain}_{task}.json")))
        rows = sorted(rows, key=lambda r: r["idx"])
        assert [r["idx"] for r in rows] == list(range(len(src))), f"{task}: {len(rows)} vs {len(src)}"
        if task == "grounding":
            for tag, thr in variants.items():
                infer = [{"item_id": s["item_id"], "scan_id": s["scan_id"], "pred": r["pred"],
                          "pred_object_id": r["pred_object_id"], "pred_iou": r["pred_iou"],
                          "target_id": s["target_id"],
                          "correct": (r["pred_object_id"] is not None and r["pred_object_id"] in s["target_id"]
                                      and r["pred_iou"] >= thr)}
                         for s, r in zip(src, rows)]
                res[f"grounding_case_acc{tag or '_best'}"] = 100 * sum(x["correct"] for x in infer) / len(infer)
                json.dump(infer, open(bdir / f"infer_grounding{tag}.json", "w"), indent=1)
            res["grounding_no_ref_token"] = sum(r["pred_query"] is None for r in rows)
        else:
            infer = [{"question_id": s["question_id"], "scene_id": s["scene_id"],
                      "question": s["question"], "pred": r["pred"], "answers": s["answers"]}
                     for s, r in zip(src, rows)]
            res["qa_local"] = em_scores(rows)
            path = bdir / "infer_qa.json"
            json.dump(infer, open(path, "w"), indent=1)
        jobs = []
        if task == "grounding":
            for tag in variants:
                jobs.append((f"grounding{tag}", [
                    sys.executable, str(HERE / "beacon3d" / "beacon_grounding_shim.py"),
                    "--infer", str(bdir / f"infer_grounding{tag}.json"), "--domain", domain]))
        elif os.environ.get("OPENAI_API_KEY"):
            for tag in variants:  # chain analysis against each grounding variant
                jobs.append((f"qa{tag}", [
                    sys.executable, "evaluate_qa.py", "--infer", str(path), "--domain", domain,
                    "--model", qa_model, "--output", str(bdir / f"infer_qa{tag}_processed.json"),
                    "--grounding", str(bdir / f"infer_grounding{tag}_processed.json")]))
        else:
            print(f"[beacon3d_{domain}] qa: OPENAI_API_KEY unset, GPT-scored evaluation skipped")
        for name, cmd in jobs:
            print(f"$ {' '.join(cmd)}", flush=True)
            p = subprocess.run(cmd, cwd=beacon_root, capture_output=True, text=True)
            (bdir / f"score_{name}.txt").write_text(rel(p.stdout + ("" if p.returncode == 0 else "\n[stderr]\n" + p.stderr)))
            print(p.stdout[-3000:])
            if p.returncode != 0:
                print(p.stderr[-3000:])
                raise SystemExit(f"beacon3d {name} evaluator failed")
    res["qa_judge"] = qa_model if os.environ.get("OPENAI_API_KEY") else None
    json.dump(res, open(bdir / "scores_local.json", "w"), indent=1)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pred_dir", required=True, help="infer.py --out_dir")
    ap.add_argument("--result_dir", default="extra_eval/results")
    ap.add_argument("--beacon_root", default=str(HERE / "beacon3d" / "Beacon3D"))
    ap.add_argument("--benches", default="sqa3d,real3dqa,beacon3d")
    ap.add_argument("--qa_model", default="gpt-4o-mini-2024-07-18",
                    help="Beacon3D QA judge (upstream default is gpt-4o-2024-11-20)")
    args = ap.parse_args()
    out = Path(args.result_dir).resolve()
    for d in ["sqa3d", "real3dqa", "beacon3d_scannet"]:
        (out / d).mkdir(parents=True, exist_ok=True)
    summary = {}
    if "sqa3d" in args.benches:
        summary["sqa3d"] = score_sqa3d(args.pred_dir, out)
    if "real3dqa" in args.benches:
        summary["real3dqa"] = score_real3dqa(args.pred_dir, out)
    if "beacon3d" in args.benches:
        summary["beacon3d_scannet"] = score_beacon3d(args.pred_dir, out, args.beacon_root, args.qa_model)
    json.dump(summary, open(out / "summary.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
