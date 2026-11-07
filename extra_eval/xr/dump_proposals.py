#!/usr/bin/env python3
"""Dump the frozen Mask3D-CLASP detector's top-100 proposals for the XR val scenes.

One detector pass per scene (as ``finetune_xr.py cache``: eval mode, no augmentation),
then the 100 query masks at full resolution (``infer.scene_forward``: segment masks mapped
back to every point of ``preprocess_xr.py``'s ``<alias>.npy``, i.e. the union of the
scene's SceneVerse regions in ``scene_map`` order). Written per scene alias:

    <out_dir>/<alias>.npz
        masks   bool    [N, 100]  query k's mask over the N points
        scores  float32 [100]     max class probability excluding "no object"
        classes int32   [100]     its ScanNet200 class index (model output space)
        n_points int64  []        N

``scores`` / ``classes`` come from the detector's class head. The released ScanNet
``grounded-3d-llm-proposal-top100`` files instead took them from the LLM's per-class
detection logits; masks are what downstream readers use.

    python extra_eval/xr/dump_proposals.py --placement rc \
        --processed_dir $XR_PC/xr_rc/scannet200 --out_dir <dir> [--scenes xq0027,xc0003]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from infer import build_model, scene_forward  # noqa: E402  (chdirs to the repo root)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--placement", required=True, choices=["rc", "glb"])
    ap.add_argument("--processed_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--ckpt", default="saved/step3_mask3d_lang_4GPUS/last-epoch.ckpt")
    ap.add_argument("--scenes", default="", help="comma-separated aliases (default: all val)")
    ap.add_argument("--shard", default="0/1", help="i/n")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    _, model = build_model(args.ckpt, str(out / "g3d_save"), f"xr_{args.placement}",
                           str(Path(args.processed_dir).resolve()))
    import hydra
    ds = model.test_dataset
    scene_of = [d["raw_filepath"].split("/")[-2] for d in ds.data]
    want = set(filter(None, args.scenes.split(",")))
    k, n = map(int, args.shard.split("/"))
    # shard the fixed scene list first, then skip finished ones: shards stay disjoint
    # even when processes start at different times
    mine = [i for i, s in enumerate(scene_of) if not want or s in want][k::n]
    todo = [i for i in mine if not (out / f"{scene_of[i]}.npz").exists()]
    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(ds, todo), batch_size=1, shuffle=False, num_workers=4,
        collate_fn=hydra.utils.instantiate(model.config.data.test_collation))
    for si, batch in enumerate(loader):
        scene = batch[2][0]
        output, _, masks = scene_forward(model, batch)
        masks = np.asarray(masks).astype(bool)
        n_pts = len(np.load(Path(args.processed_dir) / "validation" / f"{scene}.npy", mmap_mode="r"))
        assert masks.shape == (n_pts, 100), (scene, masks.shape, n_pts)
        prob = torch.softmax(output["pred_logits"][0].float(), dim=-1)[:, :-1].cpu()
        score, cls = prob.max(dim=-1)
        tmp = out / f"{scene}.tmp.npz"
        np.savez_compressed(tmp, masks=masks, scores=score.numpy().astype(np.float32),
                            classes=cls.numpy().astype(np.int32), n_points=np.int64(n_pts))
        tmp.rename(out / f"{scene}.npz")
        print(f"[{si + 1}/{len(todo)}] {scene} points={n_pts} "
              f"nonempty={(masks.sum(0) > 0).sum()}/100", flush=True)
    print("PROPOSALS_DONE")


if __name__ == "__main__":
    main()
