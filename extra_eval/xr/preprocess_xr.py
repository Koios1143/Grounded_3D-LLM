#!/usr/bin/env python3
"""XR-QA / XR-SceneCaption (LSceneLLM) scenes -> the processed ``scannet200`` layout.

An XR scene is a set of SceneVerse HM3D regions (XR-QA ``qa_pairs[].region_id``,
XR-SceneCaption ``room_id``). Each region's point cloud is self-centred (xy mean 0, z min
0); the scene is the union of its regions placed by ``--placement``:

* ``rc``  LSceneLLM's loader (``SceneVerseDataset.__getitem__``): region + its
  ``room_center/<scan>.pth[sub]['center']``. Only approximate: regions of one scene end up
  ~0.3-0.9 m off from each other.
* ``glb`` region + its exact translation into the HM3D v0.2 glb frame
  (``extra_eval/xr/region_offsets.json``, from Chat-Scene's ``xr_eval/register_regions.py``),
  i.e. the real layout of the house.

Normals and superpoints are computed once per region, as for 3RScan / MultiScan in
``preprocess_pointcloud.py`` (Open3D normals oriented towards the region centre, ScanNet
Segmentator on a kNN graph), and cached in ``<out_dir>/regions/``. Instance ids are made
unique across the regions of a scene; there are no ScanNet200 labels (QA / captioning
only).

Writes ``<out_dir>/xr_<placement>/scannet200/`` and ``<out_dir>/xr_<split>_items.json``
(one record per question: scene alias, task, question, answers, LSceneLLM episode key) +
``xr_<split>_scene_map.json``.
"""

import argparse
import json
import os
import re
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from preprocess_pointcloud import estimate_normals, superpoints  # noqa: E402


def qa_items(xr_dir, split):
    """LSceneLLM's HD_Hm3dQADataset order; key = its episode_id."""
    items = []
    for scan, eps in json.load(open(os.path.join(xr_dir, "XR-QA", f"{split}.json"))).items():
        for ep in eps:
            for qa in ep["qa_pairs"]:
                items.append(dict(task="qa", scan=scan, regions=qa["region_id"].split("-"),
                                  question=qa["question"], answers=[qa["answer"]], type=qa["type"],
                                  key="HM3D#{}#{}#{}".format(scan, qa["region_id"], qa["question"])))
    return items


def cap_items(xr_dir, split):
    items = []
    for i, x in enumerate(json.load(open(os.path.join(xr_dir, "XR-SceneCaption", f"{split}.json")))):
        q = re.sub(r"^\s*###\s*human:\s*|\s*###\s*assistant:\s*$", "", x["question"]).strip()
        items.append(dict(task="cap", scan=x["scan_id"], regions=x["room_id"].split("-"), question=q,
                          answers=x["answers"], key="HM3D#{}#{}#{}#{}".format(x["scan_id"], x["room_id"], q, i)))
    return items


def region_job(job):
    reg, scan_data, out = job
    if out.exists():
        return reg
    xyz, rgb, inst = torch.load(Path(scan_data) / "pcd_with_global_alignment" / f"{reg}.pth", weights_only=False)
    normals = estimate_normals(xyz)
    seg = np.unique(superpoints(xyz, normals), return_inverse=True)[1]
    np.savez(out.with_suffix(".tmp.npz"), xyz=xyz.astype(np.float32), rgb=rgb.astype(np.float32),
             normals=normals.astype(np.float32), seg=seg.astype(np.int64), inst=inst.astype(np.int64))
    os.replace(out.with_suffix(".tmp.npz"), out)
    return reg


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--xr_dir", required=True, help="dir with XR-QA/ and XR-SceneCaption/")
    ap.add_argument("--scan_data", required=True, help="SceneVerse HM3D scan_data (with room_center/)")
    ap.add_argument("--offsets", default="extra_eval/xr/region_offsets.json")
    ap.add_argument("--scannet_processed", default="data/processed/scannet200")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--splits", nargs="+", default=["val"])
    ap.add_argument("--placements", nargs="+", default=["rc", "glb"], choices=["rc", "glb"])
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    out = Path(args.out_dir)
    (out / "regions").mkdir(parents=True, exist_ok=True)
    offsets = json.load(open(args.offsets))

    scenes = {}  # (split, task, scan, regions) -> alias
    for split in args.splits:
        items = []
        for task, its in (("qa", qa_items(args.xr_dir, split)), ("cap", cap_items(args.xr_dir, split))):
            prefix = {"qa": "xq", "cap": "xc"}[task] + ("t" if split == "train" else "")
            local = {}
            for it in its:
                k = (it["scan"], tuple(it["regions"]))
                if k not in local:
                    local[k] = f"{prefix}{len(local):04d}"
                    scenes[(split, task) + k] = local[k]
                it["scene_id"] = local[k]
            items += its
        for i, it in enumerate(items):
            it["idx"] = i
        json.dump(items, open(out / f"xr_{split}_items.json", "w"))
        json.dump({a: dict(scan=k[2], regions=list(k[3])) for k, a in scenes.items() if k[0] == split},
                  open(out / f"xr_{split}_scene_map.json", "w"), indent=0)
        print(f"{split}: {len(items)} items, {sum(k[0] == split for k in scenes)} scenes")

    regions = sorted({f"{k[2]}_{r}" for k in scenes for r in k[3]})
    jobs = [(r, args.scan_data, out / "regions" / f"{r}.npz") for r in regions]
    with ProcessPoolExecutor(args.workers) as ex:
        for i, _ in enumerate(ex.map(region_job, jobs, chunksize=1)):
            if (i + 1) % 100 == 0:
                print(f"regions {i + 1}/{len(jobs)}", flush=True)

    room_center = {}
    for placement in args.placements:
        save = out / f"xr_{placement}" / "scannet200"
        (save / "validation").mkdir(parents=True, exist_ok=True)
        (save / "instance_gt" / "validation").mkdir(parents=True, exist_ok=True)
        db = []
        for (split, _, scan, regs), alias in sorted(scenes.items(), key=lambda x: x[1]):
            parts, seg_off, inst_off = [], 0, 0
            for sub in regs:
                r = np.load(out / "regions" / f"{scan}_{sub}.npz")
                if placement == "rc":
                    if scan not in room_center:
                        room_center[scan] = torch.load(Path(args.scan_data) / "room_center" / f"{scan}.pth",
                                                       weights_only=False)
                    t = np.asarray(room_center[scan][sub]["center"], np.float32)
                else:
                    t = np.asarray(offsets[scan][sub]["T"], np.float32)
                inst = np.where(r["inst"] > 0, r["inst"] + inst_off, 0)  # 0 = no instance in SceneVerse
                parts.append(np.hstack([r["xyz"] + t, r["rgb"], r["normals"], (r["seg"] + seg_off)[:, None],
                                        np.zeros((len(inst), 1)), inst[:, None]]))
                seg_off += int(r["seg"].max()) + 1
                inst_off += int(r["inst"].max()) + 1
            points = np.concatenate(parts).astype(np.float32)
            npy = save / "validation" / f"{alias}.npy"
            np.save(npy, points)
            gt = save / "instance_gt" / "validation" / f"{alias}.txt"
            np.savetxt(gt, (points[:, 11] + 1).astype(np.int32), fmt="%d")
            # the loader takes the scene name from raw_filepath's parent directory
            db.append({"filepath": str(npy.resolve()), "raw_filepath": f"xr/{alias}/{alias}.pth",
                       "instance_gt_filepath": str(gt.resolve()), "file_len": len(points), "scene": alias,
                       "num_superpoints": seg_off})
        for f in ["label_database.yaml", "color_mean_std.yaml"]:
            shutil.copy(Path(args.scannet_processed) / f, save / f)
        with open(save / "validation_database.yaml", "w") as f:
            yaml.safe_dump(db, f)
        print(f"xr_{placement}: {len(db)} scenes, points {np.mean([d['file_len'] for d in db]):.0f} avg, "
              f"superpoints {np.mean([d['num_superpoints'] for d in db]):.0f} avg -> {save}")


if __name__ == "__main__":
    sys.exit(main())
