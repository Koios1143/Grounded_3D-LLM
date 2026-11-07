"""ScanRefer / Multi3DRefer of the native test pass, re-scored over the OFFICIAL val sets.

The repository's test pass (``extra_eval/infer.py`` native step, scored by ``score_native.py``)
only asks part of the val items, and its tables divide by the asked items, as the paper does:

* the detector's text input holds a scene's ScanRefer / Multi3DRefer sentences only up to
  48 clips x 256 tokens; later sentences are never asked,
* items whose target is a wall or floor are skipped,
* a few official val items are absent from the release langdata (``data/langdata``).

Here every official val item counts (ScanRefer_filtered_val: 9508, Multi3DRefer val: 11120):
an item that was never asked scores 0 (Acc / F1). The asked-only numbers are written next to
them, so both protocols stay on record. Per-type labels: Multi3DRefer ``eval_type`` of the
official file; ScanRefer unique / multiple from ScanRefer's own split in ``scanrefer``
(lookup built by build_scanrefer_lookup.py from ScanRefer's label table, as its lib/dataset.py
does; the other baselines are scored with the same file), and the release langdata's labels in
``scanrefer_repo_labels`` (what score_native.py reports). The two agree on every item; a
lookup built from ScanNet's own table ("refridgerator") would label 86 val sentences multiple.

    python extra_eval/score_official_val.py --native_dir outputs/native_nocache \
        --scanrefer_val ... --m3dref_val ... --scanrefer_lookup ...
"""

import argparse
import glob
import json
import os
import re
from collections import Counter, defaultdict

import numpy as np

SR_TYPES = ("unique", "multiple")
M3D_TYPES = ("zt_wo_d", "zt_w_d", "st_wo_d", "st_w_d", "mt")
THR_SR = (("Acc@0.25", 0.25), ("Acc@0.5", 0.5))
THR_M3D = (("F1@0.25", "bbox_iou_25_f1_score"), ("F1@0.5", "bbox_iou_50_f1_score"))


def load_asked(native_dir):
    """Asked items: ScanRefer as (scene, prompt, repo label, top-1 IoU); Multi3DRefer F1 by type."""
    sr, m3d = [], defaultdict(lambda: defaultdict(list))
    files = sorted(glob.glob(os.path.join(native_dir, "native", "scene*.json")))
    assert files, f"no native predictions under {native_dir}/native"
    for p in files:
        for q in json.load(open(p))["prediction"]:
            t = q["type"]
            if t.startswith("scanrefer:"):
                scene = os.path.basename(p)[:-len(".json")]
                sr.append((scene, q["input_language"], t.split(":")[1], q.get("top_1_bbox_iou") or 0.0))
            elif t.startswith("m3dref:"):
                for name, key in THR_M3D:
                    m3d[name][t.split(":")[1]].append(q.get(key) or 0.0)
    return sr, m3d


def val_scenes():
    v = open("data/processed/scannet200/validation_database.yaml").read()
    return set(l.split("/")[-2] for l in v.splitlines() if "raw_filepath:" in l)


def langdata_missing(official, langdata_file, key_off, key_lang):
    """Official items absent from the release langdata (multiset match)."""
    scenes = val_scenes()
    have = Counter(key_lang(x) for x in json.load(open(langdata_file)) if x["scene_id"] in scenes)
    miss = []
    for x in official:
        k = key_off(x)
        if have[k] > 0:
            have[k] -= 1
        else:
            miss.append(k)
    return miss


def block(n_asked, n_official, asked, official):
    return {"n_asked": n_asked, "n_official": n_official, "asked": asked, "official_val": official}


def norm_text(t):
    return re.sub(r"[\s.?!;,]+$", "", t.strip()).lower()


def official_labels(sr, langdata_file, lookup):
    """ScanRefer's own unique / multiple label (the lookup of build_scanrefer_lookup.py, the
    lib/dataset.py definition, used for every baseline) for each asked item. The native
    predictions carry no ScanNet object id (``gt_inst_ids`` are the repository's instance
    indices), so an item is matched to its langdata sentence by scene + text (the prompt wraps
    the description and changes its final punctuation). Sentences repeated in a scene all
    resolve to the same label (checked)."""
    scenes = val_scenes()
    by_scene = defaultdict(lambda: defaultdict(set))
    for x in json.load(open(langdata_file)):
        if x["scene_id"] in scenes:
            by_scene[x["scene_id"]][norm_text(x["description"])].add(
                "multiple" if lookup[x["scene_id"]][str(x["object_ids"][0][0])] else "unique")
    labels = []
    for scene, prompt, _, _ in sr:
        m = re.search(r"search for: (.*) \(with grounding\)$", prompt, re.S)
        cand = by_scene[scene].get(norm_text(m.group(1))) if m else None
        if not cand:  # other instruction templates: the sentence is a substring of the prompt
            low = prompt.lower()
            cand = set().union(*[v for k, v in by_scene[scene].items() if k and k in low] or [set()])
        assert len(cand) == 1, (scene, prompt, cand)
        labels.append(next(iter(cand)))
    return labels


def acc_block(groups, n_off):
    out, all_iou = {}, []
    for t in SR_TYPES:
        v = np.asarray(groups[t], dtype=float)
        all_iou.append(v)
        out[t] = block(len(v), n_off[t], {m: 100 * float((v > th).mean()) for m, th in THR_SR},
                       {m: 100 * float((v > th).sum()) / n_off[t] for m, th in THR_SR})
    v = np.concatenate(all_iou)
    n = sum(n_off.values())
    out["overall"] = block(len(v), n, {m: 100 * float((v > th).mean()) for m, th in THR_SR},
                           {m: 100 * float((v > th).sum()) / n for m, th in THR_SR})
    return out


def score_scanrefer(sr, official, lookup, langdata_file, missing):
    """Two unique / multiple splits (overall is the same):
    ``official``: ScanRefer's lookup for numerator and denominator (comparable to the other
    baselines); ``repo_labels``: the release langdata ``eval_type`` (what score_native.py
    reports), with the lookup only for the official items absent from the langdata."""
    n_lookup = Counter("multiple" if lookup[x["scene_id"]][str(x["object_id"])] else "unique" for x in official)
    lab = official_labels(sr, langdata_file, lookup)
    g_off, g_repo = defaultdict(list), defaultdict(list)
    for (_, _, repo_t, iou), t in zip(sr, lab):
        g_off[t].append(iou)
        g_repo[repo_t].append(iou)
    scenes = val_scenes()
    n_repo = Counter(x["eval_type"] for x in json.load(open(langdata_file)) if x["scene_id"] in scenes)
    n_repo.update("multiple" if lookup[s][str(o)] else "unique" for s, o, _ in missing)
    assert sum(n_lookup.values()) == sum(n_repo.values()) == len(official)
    return acc_block(g_off, n_lookup), acc_block(g_repo, n_repo)


def score_m3dref(m3d, official):
    n_off = Counter(x["eval_type"] for x in official)
    out = {}
    for t in M3D_TYPES:
        n = len(m3d["F1@0.25"][t])
        out[t] = block(n, n_off[t], {m: 100 * float(np.mean(m3d[m][t])) if n else 0.0 for m, _ in THR_M3D},
                       {m: 100 * float(np.sum(m3d[m][t])) / n_off[t] for m, _ in THR_M3D})
    allv = {m: np.concatenate([np.asarray(m3d[m][t], dtype=float) for t in M3D_TYPES]) for m, _ in THR_M3D}
    out["overall"] = block(len(allv["F1@0.25"]), len(official),
                           {m: 100 * float(allv[m].mean()) for m, _ in THR_M3D},
                           {m: 100 * float(allv[m].sum()) / len(official) for m, _ in THR_M3D})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--native_dir", default="outputs/native_nocache")
    ap.add_argument("--scanrefer_val", required=True, help="ScanRefer_filtered_val.json (9508)")
    ap.add_argument("--m3dref_val", required=True, help="multi3drefer_val.json (11120)")
    ap.add_argument("--scanrefer_lookup", required=True, help="ScanRefer's unique / multiple lookup (build_scanrefer_lookup.py)")
    ap.add_argument("--langdata", default="data/langdata")
    ap.add_argument("--out", default="extra_eval/results/official_val/scanrefer_m3dref.json")
    args = ap.parse_args()

    sr_off = json.load(open(args.scanrefer_val))
    m3d_off = json.load(open(args.m3dref_val))
    lookup = json.load(open(args.scanrefer_lookup))
    assert len(sr_off) == 9508 and len(m3d_off) == 11120, (len(sr_off), len(m3d_off))

    sr, m3d = load_asked(args.native_dir)
    sr_miss = langdata_missing(sr_off, os.path.join(args.langdata, "scanrefer_format.json"),
                               lambda x: (x["scene_id"], int(x["object_id"]), int(x["ann_id"])),
                               lambda x: (x["scene_id"], int(x["object_ids"][0][0]), int(x["ann_id"])))
    m3d_miss = langdata_missing(m3d_off, os.path.join(args.langdata, "m3dref_format.json"),
                                lambda x: (x["scene_id"], x["object_name"], int(x["ann_id"]), x["eval_type"]),
                                lambda x: (x["scene_id"], x["object_name"], int(x["ann_id"]), x["eval_type"]))
    sr_official, sr_repo = score_scanrefer(sr, sr_off, lookup, os.path.join(args.langdata, "scanrefer_format.json"), sr_miss)
    res = {
        "note": "Grounded 3D-LLM release checkpoint, native test pass. 'asked' = upstream protocol "
                "(mean over the items its test pass asks; matches the paper). 'official_val' = every "
                "official val item, items not asked score 0. Not asked: sentences beyond the detector's "
                "48 x 256-token text input, wall / floor targets, and official items absent from the "
                "release langdata (listed below).",
        "native_dir": args.native_dir,
        "scanrefer": sr_official,
        "scanrefer_repo_labels": sr_repo,
        "m3dref": score_m3dref(m3d, m3d_off),
        "not_in_langdata": {"scanrefer": [list(k) for k in sr_miss], "m3dref": [list(k) for k in m3d_miss]},
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=1)
    for bench in ("scanrefer", "scanrefer_repo_labels", "m3dref"):
        for k, v in res[bench].items():
            fmt = lambda d: " / ".join(f"{x:.2f}" for x in d.values())
            print(f"{bench:21s} {k:8s} asked {v['n_asked']:5d}: {fmt(v['asked'])}   "
                  f"official {v['n_official']:5d}: {fmt(v['official_val'])}")
    print(f"not in langdata: scanrefer {len(sr_miss)}, m3dref {len(m3d_miss)} -> {args.out}")


if __name__ == "__main__":
    main()
