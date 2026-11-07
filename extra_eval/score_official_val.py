"""ScanRefer / Multi3DRefer of the native test pass, re-scored over the OFFICIAL val sets.

The repository's test pass (``extra_eval/infer.py`` native step, scored by ``score_native.py``)
only asks part of the val items, and its tables divide by the asked items, as the paper does:

* the detector's text input holds a scene's ScanRefer / Multi3DRefer sentences only up to
  48 clips x 256 tokens; later sentences are never asked,
* items whose target is a wall or floor are skipped,
* a few official val items are absent from the release langdata (``data/langdata``).

Here every official val item counts (ScanRefer_filtered_val: 9508, Multi3DRefer val: 11120):
an item that was never asked scores 0 (Acc / F1). The asked-only numbers are written next to
them, so both protocols stay on record. Per-type denominators: Multi3DRefer ``eval_type`` of
the official file; ScanRefer unique / multiple as the release langdata labels them (the asked
items carry those labels), with scanrefer_unique_multiple_lookup.json only for the official
items absent from the langdata.

    python extra_eval/score_official_val.py --native_dir outputs/native_nocache \
        --scanrefer_val ... --m3dref_val ... --scanrefer_lookup ...
"""

import argparse
import glob
import json
import os
from collections import Counter, defaultdict

import numpy as np

SR_TYPES = ("unique", "multiple")
M3D_TYPES = ("zt_wo_d", "zt_w_d", "st_wo_d", "st_w_d", "mt")
THR_SR = (("Acc@0.25", 0.25), ("Acc@0.5", 0.5))
THR_M3D = (("F1@0.25", "bbox_iou_25_f1_score"), ("F1@0.5", "bbox_iou_50_f1_score"))


def load_asked(native_dir):
    """Per-item scores of the asked items, grouped by type."""
    sr, m3d = defaultdict(list), defaultdict(lambda: defaultdict(list))
    files = sorted(glob.glob(os.path.join(native_dir, "native", "scene*.json")))
    assert files, f"no native predictions under {native_dir}/native"
    for p in files:
        for q in json.load(open(p))["prediction"]:
            t = q["type"]
            if t.startswith("scanrefer:"):
                sr[t.split(":")[1]].append(q.get("top_1_bbox_iou") or 0.0)
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


def score_scanrefer(sr, official, lookup, langdata_file, missing):
    # unique / multiple as the repository labels them (its langdata ``eval_type``; Chat-Scene's
    # lookup labels 86 more val items multiple), so numerator and denominator agree; only the
    # official items absent from the langdata take their label from the lookup.
    scenes = val_scenes()
    n_off = Counter(x["eval_type"] for x in json.load(open(langdata_file)) if x["scene_id"] in scenes)
    n_off.update("multiple" if lookup[s][str(o)] else "unique" for s, o, _ in missing)
    assert sum(n_off.values()) == len(official), (n_off, len(official))
    out, all_iou = {}, []
    for t in SR_TYPES:
        v = np.asarray(sr[t], dtype=float)
        all_iou.append(v)
        out[t] = block(len(v), n_off[t], {m: 100 * float((v > th).mean()) for m, th in THR_SR},
                       {m: 100 * float((v > th).sum()) / n_off[t] for m, th in THR_SR})
    v = np.concatenate(all_iou)
    out["overall"] = block(len(v), len(official), {m: 100 * float((v > th).mean()) for m, th in THR_SR},
                           {m: 100 * float((v > th).sum()) / len(official) for m, th in THR_SR})
    return out


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
    ap.add_argument("--scanrefer_lookup", required=True, help="scanrefer_unique_multiple_lookup.json")
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
    res = {
        "note": "Grounded 3D-LLM release checkpoint, native test pass. 'asked' = upstream protocol "
                "(mean over the items its test pass asks; matches the paper). 'official_val' = every "
                "official val item, items not asked score 0. Not asked: sentences beyond the detector's "
                "48 x 256-token text input, wall / floor targets, and official items absent from the "
                "release langdata (listed below).",
        "native_dir": args.native_dir,
        "scanrefer": score_scanrefer(sr, sr_off, lookup, os.path.join(args.langdata, "scanrefer_format.json"), sr_miss),
        "m3dref": score_m3dref(m3d, m3d_off),
        "not_in_langdata": {"scanrefer": [list(k) for k in sr_miss], "m3dref": [list(k) for k in m3d_miss]},
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=1)
    for bench in ("scanrefer", "m3dref"):
        for k, v in res[bench].items():
            fmt = lambda d: " / ".join(f"{x:.2f}" for x in d.values())
            print(f"{bench:9s} {k:8s} asked {v['n_asked']:5d}: {fmt(v['asked'])}   "
                  f"official {v['n_official']:5d}: {fmt(v['official_val'])}")
    print(f"not in langdata: scanrefer {len(sr_miss)}, m3dref {len(m3d_miss)} -> {args.out}")


if __name__ == "__main__":
    main()
