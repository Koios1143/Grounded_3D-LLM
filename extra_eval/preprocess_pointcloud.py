#!/usr/bin/env python3
"""Point-cloud scenes (no mesh) -> the processed ``scannet200`` layout, for Beacon3D's
3RScan / MultiScan domains and a ScanNet control that runs under the same conditions.

The released detector pools over mesh superpoints (ScanNet's ``.segs.json``). 3RScan and
MultiScan come as SceneVerse point clouds without faces, so superpoints are rebuilt with
ScanNet Segmentator's own algorithm (Felzenszwalb merging, edge weight ``1 - n1.n2``,
squared on convex edges, kThresh 0.01, segments < 20 points merged), on a symmetric kNN
graph instead of mesh edges. Normals: 3RScan ships them (``normal/<scene>.npy``);
otherwise they are estimated with Open3D and oriented towards the scene centre (scans are
seen from inside the room).

* ``--domain 3rscan|multiscan``: SceneVerse ``pcd_with_global_alignment/<scene>.pth``
  (xyz, rgb 0-255, instance id = Beacon3D object id), already axis-aligned.
* ``--domain scannet_pcseg``: the ScanNet scenes already processed by
  ``preprocess_scannet.py``, with their mesh normals and official superpoints replaced by
  estimated normals and kNN superpoints (labels, instance ids, coordinates unchanged).

Only the scenes Beacon3D uses are processed. Writes ``<out_dir>/<domain>/scannet200/``.
"""

import argparse
import json
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial import cKDTree

SEG_K, SEG_THR, SEG_MIN = 8, 0.01, 20


def estimate_normals(xyz):
    import open3d as o3d
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn=16))
    n = np.asarray(pc.normals)
    centre = np.array([*np.median(xyz[:, :2], 0), np.percentile(xyz[:, 2], 50)])
    flip = ((centre - xyz) * n).sum(1) < 0  # point the normal towards the room interior
    n[flip] *= -1
    return n


def superpoints(xyz, normals):
    """ScanNet Segmentator (segment_graph + small-segment merge) on a kNN graph."""
    n = len(xyz)
    _, nbr = cKDTree(xyz).query(xyz, k=SEG_K + 1)
    a = np.repeat(np.arange(n), SEG_K)
    b = nbr[:, 1:].reshape(-1)
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    edges = np.unique(np.stack([lo, hi], 1), axis=0)
    a, b = edges[:, 0], edges[:, 1]
    nrm = normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    d = xyz[b] - xyz[a]
    d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
    w = 1.0 - (nrm[a] * nrm[b]).sum(1)
    convex = (nrm[b] * d).sum(1) > 0
    w = np.where(convex, w * w, w)
    order = np.argsort(w, kind="stable")
    a, b, w = a[order].tolist(), b[order].tolist(), w[order].tolist()

    parent = list(range(n))
    rank = [0] * n
    size = [1] * n
    thr = [SEG_THR] * n

    def find(x):
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def join(x, y):
        if rank[x] > rank[y]:
            parent[y] = x
            size[x] += size[y]
            return x
        parent[x] = y
        size[y] += size[x]
        if rank[x] == rank[y]:
            rank[y] += 1
        return y

    for i in range(len(w)):
        x, y = find(a[i]), find(b[i])
        if x != y and w[i] <= thr[x] and w[i] <= thr[y]:
            r = join(x, y)
            thr[r] = w[i] + SEG_THR / size[r]
    for i in range(len(w)):  # Segmentator merges small segments in the same edge order
        x, y = find(a[i]), find(b[i])
        if x != y and (size[x] < SEG_MIN or size[y] < SEG_MIN):
            join(x, y)
    return np.array([find(i) for i in range(n)])


def process(job):
    domain, scene, args = job
    save = Path(args.out_dir) / domain / "scannet200"
    if domain == "scannet_pcseg":
        s, ss = int(scene[5:9]), int(scene[10:12])
        pts = np.load(Path(args.scannet_processed) / "validation" / f"{s:04}_{ss:02}.npy")
        xyz, rgb, inst, sem = pts[:, :3], pts[:, 3:6], pts[:, 11], pts[:, 10]
        normals = estimate_normals(xyz)
        name = f"{s:04}_{ss:02}"
    else:
        import torch
        root = Path(args.sceneverse) / {"3rscan": "3RScan", "multiscan": "MultiScan"}[domain]
        xyz, rgb, inst = torch.load(root / "pcd_with_global_alignment" / f"{scene}.pth", weights_only=False)
        nfile = root / "normal" / f"{scene}.npy"
        normals = np.load(nfile) if nfile.exists() else estimate_normals(xyz)
        sem = np.zeros(len(xyz))  # not a ScanNet200 id -> ignored by the loader
        name = scene
    seg = np.unique(superpoints(xyz, normals), return_inverse=True)[1]
    points = np.hstack([xyz, rgb, normals, seg[:, None], sem[:, None], inst[:, None]]).astype(np.float32)
    npy = save / "validation" / f"{name}.npy"
    np.save(npy, points)
    gt = save / "instance_gt" / "validation" / f"{scene}.txt"
    np.savetxt(gt, (sem * 1000 + inst + 1).astype(np.int32), fmt="%d")
    # the loader takes the scene name from raw_filepath's parent directory
    return {"filepath": str(npy), "raw_filepath": f"{domain}/{scene}/{scene}.pth",
            "instance_gt_filepath": str(gt), "file_len": len(xyz), "scene": name,
            "num_superpoints": int(seg.max() + 1)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--domain", required=True, choices=["3rscan", "multiscan", "scannet_pcseg"])
    ap.add_argument("--beacon_data", default="extra_eval/beacon3d/Beacon3D/data")
    ap.add_argument("--sceneverse", help="dir with 3RScan/ and MultiScan/ (SceneVerse scan_data layout)")
    ap.add_argument("--scannet_processed", default="data/processed/scannet200",
                    help="output of preprocess_scannet.py (for scannet_pcseg)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    bdomain = "scannet" if args.domain == "scannet_pcseg" else args.domain
    scenes = sorted({x["scan_id"] for x in json.load(open(
        os.path.join(args.beacon_data, bdomain, f"{bdomain}_grounding.json")))})
    save = Path(args.out_dir) / args.domain / "scannet200"
    (save / "validation").mkdir(parents=True, exist_ok=True)
    (save / "instance_gt" / "validation").mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(args.workers) as ex:
        db = list(ex.map(process, [(args.domain, s, args) for s in scenes]))
    for f in ["label_database.yaml", "color_mean_std.yaml"]:
        shutil.copy(Path(args.scannet_processed) / f, save / f)
    with open(save / "validation_database.yaml", "w") as f:
        yaml.safe_dump(db, f)
    sp = [d["num_superpoints"] for d in db]
    print(f"{args.domain}: {len(db)} scenes, points {np.mean([d['file_len'] for d in db]):.0f} avg, "
          f"superpoints {np.mean(sp):.0f} avg -> {save}")


if __name__ == "__main__":
    sys.exit(main())
