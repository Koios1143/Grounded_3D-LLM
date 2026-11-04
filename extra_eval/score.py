#!/usr/bin/env python3
"""Score ``extra_eval/infer.py`` outputs: ScanRefer / Multi3DRefer / Scan2Cap / ScanQA (upstream
metrics, see score_native.py), SQA3D EM / EM-R, Real-3DQA (official scripts), Beacon3D
(official evaluators). CPU only; METEOR and the PTB tokenizer need Java (JAVA_HOME).

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
import shutil
import subprocess
import sys

import numpy as np
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent


def rel(text):
    """Logs of the evaluators print absolute paths; keep them repo-relative."""
    return text.replace(str(REPO) + "/", "")
sys.path.insert(0, str(HERE / "real3dqa"))
from evaluate_debiased_exact_match import answer_match, clean_answer  # noqa: E402
sys.path.insert(0, str(REPO))
from extra_eval.score_native import load_native, score_native  # noqa: E402

NATIVE = ["scanrefer", "m3dref", "scan2cap", "scanqa"]


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


KNOWLEDGE = ["class", "appearance", "geometry", "spatial", "functionality"]
QA_TAGS = ["class", "appearance", "geometry", "spatial", "existence", "functionality"]
BEACON_DOMAINS = ["scannet", "3rscan", "multiscan"]
# G3D-LLM grounds to its own proposals: the proposal is mapped to the GT object of highest
# point IoU. "" accepts any overlap (closest to picking an object, as GT-mask models do);
# "_iou25" also needs IoU >= 0.25 (ScanRefer Acc@0.25 style).
VARIANTS = {"": 0.0, "_iou25": 0.25}


def beacon_units(bdir, src_g, variant=""):
    """Per-unit outcomes behind Beacon3D's numbers, re-derived from the evaluators' own
    inputs/outputs so domains can be pooled: grounding cases (deduplicated by
    (scene, utterance) as in ``build_results_mapping``), chains, objects, and QA cases
    (GPT score 1-5) with the grounding->QA chain flag of ``eval_chain``."""
    infer = json.load(open(bdir / f"infer_grounding{variant}.json"))
    cases = {}
    for s, r in zip(src_g, infer):
        tag = frozenset(s["item_id"].split("_")[-1].lower().split("-"))
        cases[(s["scan_id"], s["utterance"].lower())] = (bool(r["correct"]), tag)
    pg = json.load(open(bdir / f"infer_grounding{variant}_processed.json"))
    chains = [all(c["correct"]) for c in pg]
    objects = [all(c["correct"][-1] for c in pg[i:i + 3]) for i in range(0, len(pg), 3)]
    pq = json.load(open(bdir / "infer_qa_processed.json"))
    qa = [(it["score"], it["tag"].lower(), bool(it["extra_knowledge"]), objects[i // 3])
          for i, it in enumerate(pq)]
    return {"cases": list(cases.values()), "chains": chains, "objects": objects, "qa": qa}


def beacon_metrics(u):
    pct = lambda xs: 100 * float(np.mean(xs)) if len(xs) else None  # noqa: E731
    gpt = lambda xs: (float(np.mean(xs)) - 1) / 4 * 100 if len(xs) else None  # noqa: E731
    g = {"case": pct([c for c, _ in u["cases"]])}
    g.update({k: pct([c for c, t in u["cases"] if k in t]) for k in KNOWLEDGE})
    g.update(chain=pct(u["chains"]), object=pct(u["objects"]), n_cases=len(u["cases"]),
             n_objects=len(u["objects"]))
    sc = [q[0] for q in u["qa"]]
    q = {"case": gpt(sc), "object": pct([all(x >= 4 for x in sc[i:i + 3]) for i in range(0, len(sc), 3)])}
    q.update({k: gpt([x[0] for x in u["qa"] if x[1] == k]) for k in QA_TAGS})
    t_all, t_wo = np.zeros(4), np.zeros(4)
    for score, _, extra, gok in u["qa"]:
        k = (0 if gok else 2) + (0 if score >= 4 else 1)
        t_all[k] += 1
        t_wo[k] += not extra
    q["chain_overall"] = [round(100 * x / t_all.sum(), 2) for x in t_all]  # G1Q1, G1Q0, G0Q1, G0Q0
    q["chain_wo_extra"] = [round(100 * x / t_wo.sum(), 2) for x in t_wo]
    q["R1"] = 100 * t_wo[1] / (t_wo[1] + t_wo[3])
    q["R2"] = 100 * t_all[2] / (t_all[0] + t_all[2])
    q["n_cases"] = len(sc)
    return {"grounding": g, "qa": q}


def run_eval(name, cmd, beacon_root, bdir):
    print(f"$ {' '.join(cmd)}", flush=True)
    p = subprocess.run(cmd, cwd=beacon_root, capture_output=True, text=True)
    (bdir / f"score_{name}.txt").write_text(rel(p.stdout + ("" if p.returncode == 0 else "\n[stderr]\n" + p.stderr)))
    print(p.stdout[-3000:])
    if p.returncode != 0:
        print(p.stderr[-3000:])
        raise SystemExit(f"{bdir.name} {name} evaluator failed")


def score_beacon3d(pred_dir, out, beacon_root, qa_model, domain="scannet", qa_from=None):
    """One domain: infer files, the official evaluators (grounding for both variants, QA
    once with the GPT judge), then our re-derived metrics (checked against the official
    printout) incl. the chain analysis for the IoU >= 0.25 variant.
    ``scannet_pcseg`` is the ScanNet control run under the point-cloud conditions.
    ``qa_from``: a result dir whose QA of this domain (``infer_qa.json`` and the judged
    ``infer_qa_processed.json``) is reused instead of QA rows in ``pred_dir`` (for a
    grounding-only re-run); evaluate_qa.py then reads the judged file (no GPT calls) and
    only the chain analysis changes."""
    data_domain = "scannet" if domain == "scannet_pcseg" else domain
    bdir = out / f"beacon3d_{domain}"
    bdir.mkdir(parents=True, exist_ok=True)
    res = {}
    src = {}
    for task in ["grounding", "qa"]:
        src[task] = json.load(open(os.path.join(beacon_root, "data", data_domain, f"{data_domain}_{task}.json")))
        if task == "qa" and qa_from:
            old = Path(qa_from) / f"beacon3d_{domain}"
            old_judge = json.load(open(old / "scores.json")).get("qa_judge")
            assert old_judge == qa_model, f"{old}: QA judged by {old_judge}, not {qa_model}"
            infer = json.load(open(old / "infer_qa.json"))
            assert len(infer) == len(src[task]), f"{old}/infer_qa.json: {len(infer)} vs {len(src[task])}"
            json.dump(infer, open(bdir / "infer_qa.json", "w"), indent=1)
            shutil.copyfile(old / "infer_qa_processed.json", bdir / "infer_qa_processed.json")
            res["qa_em"] = em_scores(infer)
            res["qa_reused_from"] = rel(str(old))
            run_eval("qa", [sys.executable, "evaluate_qa.py", "--infer", str(bdir / "infer_qa.json"),
                            "--domain", data_domain, "--model", qa_model,
                            "--output", str(bdir / "infer_qa_processed.json"),
                            "--grounding", str(bdir / "infer_grounding_processed.json")], beacon_root, bdir)
            continue
        rows = load(pred_dir, f"beacon3d_{domain}_{task}")
        if not rows:
            return None
        rows = sorted(rows, key=lambda r: r["idx"])
        assert [r["idx"] for r in rows] == list(range(len(src[task]))), f"{domain}/{task}: {len(rows)} vs {len(src[task])}"
        if task == "grounding":
            for tag, thr in VARIANTS.items():
                infer = [{"item_id": s["item_id"], "scan_id": s["scan_id"], "pred": r["pred"],
                          "pred_object_id": r["pred_object_id"], "pred_iou": r["pred_iou"],
                          "target_id": s["target_id"],
                          "correct": (r["pred_object_id"] is not None and r["pred_object_id"] in s["target_id"]
                                      and r["pred_iou"] >= thr)}
                         for s, r in zip(src[task], rows)]
                json.dump(infer, open(bdir / f"infer_grounding{tag}.json", "w"), indent=1)
                run_eval(f"grounding{tag}", [sys.executable, str(HERE / "beacon3d" / "beacon_grounding_shim.py"),
                                             "--infer", str(bdir / f"infer_grounding{tag}.json"),
                                             "--domain", data_domain], beacon_root, bdir)
            res["grounding_no_ref_token"] = sum(r["pred_query"] is None for r in rows)
        else:
            infer = [{"question_id": s["question_id"], "scene_id": s["scene_id"],
                      "question": s["question"], "pred": r["pred"], "answers": s["answers"]}
                     for s, r in zip(src[task], rows)]
            path = bdir / "infer_qa.json"
            json.dump(infer, open(path, "w"), indent=1)
            res["qa_em"] = em_scores(rows)
            if not os.environ.get("OPENAI_API_KEY"):
                print(f"[beacon3d_{domain}] qa: OPENAI_API_KEY unset, GPT-judged evaluation skipped")
                json.dump(res, open(bdir / "scores.json", "w"), indent=1)
                return res
            processed = bdir / "infer_qa_processed.json"
            processed.unlink(missing_ok=True)  # evaluate_qa.py silently reuses an existing one
            run_eval("qa", [sys.executable, "evaluate_qa.py", "--infer", str(path), "--domain", data_domain,
                            "--model", qa_model, "--output", str(processed),
                            "--grounding", str(bdir / "infer_grounding_processed.json")], beacon_root, bdir)
    res["qa_judge"] = qa_model
    for tag in VARIANTS:
        res[f"metrics{tag or '_best'}"] = beacon_metrics(beacon_units(bdir, src["grounding"], tag))
    # our re-derivation must agree with the official printout
    txt = (bdir / "score_grounding.txt").read_text()
    official = float(txt.split("[overall]:")[1].split()[0])
    assert abs(official - res["metrics_best"]["grounding"]["case"]) < 0.01, (official, res["metrics_best"])
    txt = (bdir / "score_qa.txt").read_text()
    official = float(txt.split("GPT-Score =")[1].split()[0])
    assert abs(official - res["metrics_best"]["qa"]["case"]) < 0.01, (official, res["metrics_best"])
    json.dump(res, open(bdir / "scores.json", "w"), indent=1)
    return res


def beacon3d_overall(out, beacon_root, pred_dir):
    """Beacon3D defines no cross-domain score; two are reported: ``pooled`` (every case /
    chain / object of ScanNet, 3RScan and MultiScan together) and ``mean`` (average of the
    three per-domain numbers)."""
    res = {}
    for tag in VARIANTS:
        units = {}
        for d in BEACON_DOMAINS:
            src = json.load(open(os.path.join(beacon_root, "data", d, f"{d}_grounding.json")))
            units[d] = beacon_units(out / f"beacon3d_{d}", src, tag)
        pooled = {k: sum((units[d][k] for d in BEACON_DOMAINS), []) for k in units["scannet"]}
        per = {d: beacon_metrics(units[d]) for d in BEACON_DOMAINS}
        mean = {t: {k: float(np.mean([per[d][t][k] for d in BEACON_DOMAINS if per[d][t][k] is not None]))
                    for k in per["scannet"][t] if isinstance(per["scannet"][t][k], float)}
                for t in ["grounding", "qa"]}
        res[f"pooled{tag or '_best'}"] = beacon_metrics(pooled)
        res[f"mean{tag or '_best'}"] = mean
    # QA rows from pred_dir, or the infer_qa.json reused with --beacon_qa_from
    rows = {d: load(pred_dir, f"beacon3d_{d}_qa") or json.load(open(out / f"beacon3d_{d}" / "infer_qa.json"))
            for d in BEACON_DOMAINS}
    res["qa_em_pooled"] = em_scores(sum(rows.values(), []))
    res["qa_em_mean"] = {k: float(np.mean([em_scores(rows[d])[k] for d in BEACON_DOMAINS])) for k in ["EM", "EM-R"]}
    (out / "beacon3d_overall").mkdir(exist_ok=True)
    json.dump(res, open(out / "beacon3d_overall" / "scores.json", "w"), indent=1)
    return res


def export_native(pred_dir, out, benches):
    """Per-question rows of the native benchmarks (from native/<scene>.json), one jsonl each."""
    keep = ["type", "input_language", "output_language", "gt", "grounding_result", "score",
            "top_1_bbox_iou", "bbox_iou_25_f1_score", "bbox_iou_50_f1_score", "gt_predicted_iou"]
    files = {}
    for b in benches:
        (out / b).mkdir(parents=True, exist_ok=True)
        files[b] = open(out / b / "predictions.jsonl", "w")
    for scene, preds in load_native(pred_dir).items():
        for p in preds:
            b = p["type"].split(":")[0]
            if b in files:
                files[b].write(json.dumps({"scene_id": scene, **{k: p[k] for k in keep if k in p}}) + "\n")
    for f in files.values():
        f.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pred_dir", required=True, help="infer.py --out_dir")
    ap.add_argument("--result_dir", default="extra_eval/results")
    ap.add_argument("--beacon_root", default=str(HERE / "beacon3d" / "Beacon3D"))
    ap.add_argument("--benches", default="scanrefer,m3dref,scan2cap,scanqa,sqa3d,real3dqa,beacon3d")
    ap.add_argument("--qa_model", default="gpt-4o-mini-2024-07-18",
                    help="Beacon3D QA judge (upstream default is gpt-4o-2024-11-20)")
    ap.add_argument("--beacon_domains", default="scannet,3rscan,multiscan,scannet_pcseg",
                    help="Beacon3D domains to (re)score; the overall score reads all three from result_dir")
    ap.add_argument("--beacon_qa_from", default=None,
                    help="result dir whose judged Beacon3D QA is reused (grounding-only re-runs)")
    args = ap.parse_args()
    out = Path(args.result_dir).resolve()
    for d in ["sqa3d", "real3dqa"]:
        if d in args.benches.split(","):
            (out / d).mkdir(parents=True, exist_ok=True)
    summary_path = out / "summary.json"
    summary = json.load(open(summary_path)) if summary_path.exists() else {}
    native = [b for b in NATIVE if b in args.benches.split(",")]
    if native:
        val_scenes = open(REPO / "data" / "processed" / "scannet200" / "validation_database.yaml").read()
        val_scenes = set(l.split("/")[-2] for l in val_scenes.splitlines() if "raw_filepath:" in l)
        summary.update(score_native(args.pred_dir, out, val_scenes, str(REPO / "data" / "langdata"), native))
        export_native(args.pred_dir, out, native)
    if "sqa3d" in args.benches:
        summary["sqa3d"] = score_sqa3d(args.pred_dir, out)
    if "real3dqa" in args.benches:
        summary["real3dqa"] = score_real3dqa(args.pred_dir, out)
    if "beacon3d" in args.benches:
        for d in [d for d in args.beacon_domains.split(",") if d]:
            summary[f"beacon3d_{d}"] = score_beacon3d(args.pred_dir, out, args.beacon_root, args.qa_model, d,
                                                          args.beacon_qa_from)
        if all((out / f"beacon3d_{d}" / "infer_qa_processed.json").exists() for d in BEACON_DOMAINS):
            summary["beacon3d_overall"] = beacon3d_overall(out, args.beacon_root, args.pred_dir)
    json.dump(summary, open(out / "summary.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
