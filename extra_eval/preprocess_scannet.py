#!/usr/bin/env python3
"""ScanNet val scans -> the processed ``scannet200`` layout the released model reads.

Same output as ``datasets/preprocessing/scannet_preprocessing.py`` (per-scene ``.npy`` with
coords, colors, normals, segment id, ScanNet200 label, instance id; ``instance_gt/``;
``*_database.yaml``; ``label_database.yaml``), but for inference-only inputs:

* Needs ``_vh_clean_2.ply``, ``_vh_clean_2.0.010000.segs.json``, ``.aggregation.json``
  and ``.txt`` per scene. The official segs.json is required: the aggregation's object
  masks are lists of its segment ids, and a re-run of ScanNet's Segmentator (same
  kThresh/segMinVerts) does not reproduce them (~25% of ids on scene0011_00). With
  ``--segmentator`` a scene without segs.json falls back to it, but then its GT
  instances are wrong (a warning is printed); only the detector input is unaffected.
* ``_vh_clean_2.labels.ply`` is not read. The official script takes the NYU40 label
  from it only for vertices that belong to no annotated object (in ScanNet those carry
  label 0) and overwrites every annotated vertex with the ScanNet200 id from the
  aggregation, so 0 is used for unannotated vertices.
* ``color_mean_std.yaml`` must be the *train* split statistics; they are the ScanNet200
  constants Mask3D ships for inference (cvg/Mask3D ``mask3d/__init__.py``).

Writes ``<out_dir>/scannet200/`` and ``<out_dir>/scans/<scene>/`` (symlinks to the
inputs + the generated segs.json).
"""

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets.scannet200.scannet200_constants import (  # noqa: E402
    CLASS_LABELS_200, SCANNET_COLOR_MAP_200, VALID_CLASS_IDS_200)
from utils.point_cloud_utils import load_ply_with_normals  # noqa: E402

COLOR_MEAN = [0.47793125906962, 0.4303257521323044, 0.3749598901421883]
COLOR_STD = [0.2834475483823543, 0.27566157565723015, 0.27018971370874995]


def process_scene(scene, args, raw2id):
    src = Path(args.scans) / scene
    dst = Path(args.out_dir) / "scans" / scene
    dst.mkdir(parents=True, exist_ok=True)
    for suffix in ["_vh_clean_2.ply", "_vh_clean_2.0.010000.segs.json", ".aggregation.json", ".txt"]:
        link = dst / f"{scene}{suffix}"
        if not link.is_symlink() and (src / f"{scene}{suffix}").exists():
            link.symlink_to(src / f"{scene}{suffix}")
    ply = dst / f"{scene}_vh_clean_2.ply"
    segs_path = dst / f"{scene}_vh_clean_2.0.010000.segs.json"
    if not segs_path.exists():
        if not args.segmentator:
            raise FileNotFoundError(f"{src}: no official segs.json (and no --segmentator)")
        print(f"WARNING {scene}: no official segs.json, running Segmentator; GT instances will be wrong")
        subprocess.run([args.segmentator, str(ply), "0.01", "20"], check=True,
                       stdout=subprocess.DEVNULL)

    coords, features, _ = load_ply_with_normals(ply)
    points = np.hstack((coords, features))
    with open(dst / f"{scene}.txt") as f:
        scene_type = f.read().split("\n")[:-1][-1].split(" = ")[1]
    segments = np.array(json.load(open(segs_path))["segIndices"])
    assert len(segments) == len(coords), scene
    points = np.hstack((points, np.unique(segments, return_inverse=True)[1][..., None]))
    labels = np.zeros((len(coords), 2), dtype=np.int64)
    labels[:, 1] = -1
    seg_set = set(np.unique(segments).tolist())
    for inst in json.load(open(dst / f"{scene}.aggregation.json"))["segGroups"]:
        if not set(inst["segments"]) <= seg_set:
            print(f"WARNING {scene}: object {inst['id']} has segments missing from segs.json")
        occ = np.isin(segments, np.array(inst["segments"]))
        labels[occ, 1] = inst["id"]
        labels[occ, 0] = raw2id.get(inst["label"], 0)
    points = np.hstack((points, labels))
    gt_data = points[:, -2] * 1000 + points[:, -1] + 1

    s, ss = int(scene[5:9]), int(scene[10:12])
    save = Path(args.out_dir) / "scannet200"
    npy = save / "validation" / f"{s:04}_{ss:02}.npy"
    np.save(npy, points.astype(np.float32))
    gt = save / "instance_gt" / "validation" / f"scene{s:04}_{ss:02}.txt"
    np.savetxt(gt, gt_data.astype(np.int32), fmt="%d")
    return {
        "filepath": str(npy), "scene": s, "sub_scene": ss, "raw_filepath": str(ply),
        "file_len": len(coords), "scene_type": scene_type,
        "raw_description_filepath": str(dst / f"{scene}.txt"),
        "raw_instance_filepath": str(dst / f"{scene}.aggregation.json"),
        "raw_segmentation_filepath": str(segs_path),
        "instance_gt_filepath": str(gt),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scans", required=True, help="ScanNet scans/ (val scenes)")
    ap.add_argument("--label_tsv", required=True, help="scannetv2-labels.combined.tsv")
    ap.add_argument("--segmentator", default=None, help="ScanNet Segmentator, fallback only")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--scenes", default=None, help="text file with scene ids (default: all of --scans)")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    scenes = (open(args.scenes).read().split() if args.scenes
              else sorted(d for d in os.listdir(args.scans) if d.startswith("scene")))
    save = Path(args.out_dir) / "scannet200"
    (save / "validation").mkdir(parents=True, exist_ok=True)
    (save / "instance_gt" / "validation").mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.label_tsv, sep="\t", header=0)
    raw2id = {}
    for raw, i in zip(df["raw_category"], df["id"]):
        raw2id.setdefault(raw, int(i))  # first row wins, as in the official script

    with ProcessPoolExecutor(args.workers) as ex:
        db = list(ex.map(process_scene, scenes, [args] * len(scenes), [raw2id] * len(scenes)))
    db.sort(key=lambda x: (x["scene"], x["sub_scene"]))

    label_db = {cid: {"color": list(SCANNET_COLOR_MAP_200[cid]), "name": CLASS_LABELS_200[i],
                      "validation": True} for i, cid in enumerate(VALID_CLASS_IDS_200)}
    with open(save / "label_database.yaml", "w") as f:
        yaml.safe_dump(label_db, f, default_flow_style=False, allow_unicode=True)
    with open(save / "validation_database.yaml", "w") as f:
        yaml.safe_dump(db, f, default_flow_style=False, allow_unicode=True)
    with open(save / "color_mean_std.yaml", "w") as f:
        yaml.safe_dump({"mean": COLOR_MEAN, "std": COLOR_STD}, f)
    print(f"processed {len(db)} scenes -> {save}")


if __name__ == "__main__":
    main()
