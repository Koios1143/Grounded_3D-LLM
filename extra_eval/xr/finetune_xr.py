#!/usr/bin/env python3
"""Finetune the released Grounded 3D-LLM on XR-QA + XR-SceneCaption train, evaluating on val
after every epoch.

LSceneLLM's Table 1 finetunes its baselines on "LL3DA and XR-Scene data" without a recipe;
this follows the released step-3 recipe where it can: the Mask3D-CLASP detector stays frozen
(``llm_only``), the LLM trains the parameters of ``conf/llm/tiny_vicuna_len512.json``
``train_layer_list`` (LoRA r64 on q/v, instance / query projections, embeddings, lm_head)
with its loss (``LLama3dForCausalLM.model_forward``), AdamW (weight decay 1e-4) under a
OneCycle schedule and gradient-norm clipping at 0.1. Differences: XR data only, and the
detector is run once per scene in eval mode without the training augmentations, its 100
query features cached (``cache`` stage), since it is frozen anyway.

Training items use the evaluation prompts: XR-QA questions get the ScanQA suffix
(``scanqa:...:text_only``), XR-SceneCaption asks its own question (``globalscenecap:...:
text_only``), one item per reference caption.

    python extra_eval/xr/finetune_xr.py cache --placement glb --processed_dir ... --cache_dir ...
    python extra_eval/xr/finetune_xr.py train --placement glb --processed_dir ... --cache_dir ... \
        --train_items xr_train_items.json --val_items xr_val_items.json --out_dir outputs/xr_ft_glb
Per epoch: ``<out_dir>/epoch<k>/xr{qa,cap}_<placement>.jsonl`` (infer.py's format, scored by
score_xr.py) and ``<out_dir>/trainable_epoch<k>.pt`` (the trained parameters only).
"""

import argparse
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import infer  # noqa: E402  (chdirs to the repo root)
from infer import ask, build_model, clean_generation, scene_forward  # noqa: E402


def cache(args):
    _, model = build_model(args.ckpt, str(Path(args.cache_dir) / "g3d_save"), f"xr_{args.placement}",
                           str(Path(args.processed_dir).resolve()))
    import hydra
    ds = model.test_dataset
    scene_of = [d["raw_filepath"].split("/")[-2] for d in ds.data]
    out = Path(args.cache_dir)
    out.mkdir(parents=True, exist_ok=True)
    k, n = map(int, args.shard.split("/"))
    todo = [i for i, s in enumerate(scene_of) if not (out / f"{s}.pt").exists()][k::n]
    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(ds, todo), batch_size=1, shuffle=False, num_workers=4,
        collate_fn=hydra.utils.instantiate(model.config.data.test_collation))
    for si, batch in enumerate(loader):
        scene = batch[2][0]
        output, _, _ = scene_forward(model, batch)
        torch.save({"hidden": output["queries_hidden_state"][0].float().cpu(),
                    "embed": output["queries_normalized_embed"][0].float().cpu()}, out / f"{scene}.pt")
        print(f"[{si + 1}/{len(todo)}] {scene}", flush=True)
    print("CACHE_DONE")


def train_items(items_file):
    from datasets.language_info import lang_info_data
    out = []
    for it in json.load(open(items_file)):
        lang_type = "scanqa:xr:text_only" if it["task"] == "qa" else "globalscenecap:xr:text_only"
        for ans in it["answers"]:
            out.append((it["scene_id"], it["question"], ans, lang_type))
    return out, lang_info_data


def evaluate(model, items_file, feats, out_dir, placement):
    out_dir.mkdir(parents=True, exist_ok=True)
    by_scene = defaultdict(list)
    for it in infer.load_xr(items_file, placement):
        by_scene[it["scene_id"]].append(it)
    files = {}
    model.llama_model.eval()
    with torch.no_grad():
        for scene, its in by_scene.items():
            f = feats[scene]
            outs = ask(model, its, f["hidden"].cuda(), f["embed"].cuda())
            for it, o in zip(its, outs):
                if it["bench"] not in files:
                    files[it["bench"]] = open(out_dir / f"{it['bench']}.jsonl", "w")
                files[it["bench"]].write(json.dumps(dict(it, raw=o["output_language"],
                                                         pred=clean_generation(o["output_language"]))) + "\n")
    for f in files.values():
        f.close()


def train(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    _, model = build_model(args.ckpt, str(Path(args.out_dir) / "g3d_save"), f"xr_{args.placement}",
                           str(Path(args.processed_dir).resolve()))
    llm = model.llama_model
    params = [p for p in llm.parameters() if p.requires_grad]
    print(f"trainable: {sum(p.numel() for p in params) / 1e6:.1f} M parameters in {len(params)} tensors")
    feats = {p.stem: torch.load(p) for p in Path(args.cache_dir).glob("*.pt")}
    items, lang_info_data = train_items(args.train_items)
    missing = {s for s, *_ in items} - set(feats)
    assert not missing, f"{len(missing)} train scenes not cached, e.g. {sorted(missing)[:3]}"
    steps_per_epoch = math.ceil(len(items) / args.batch_size)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps_per_epoch * args.epochs)
    out = Path(args.out_dir)
    print(f"{len(items)} train items, {steps_per_epoch} steps/epoch x {args.epochs}, batch {args.batch_size}, "
          f"max lr {args.lr}", flush=True)
    for epoch in range(args.epochs):
        llm.train()
        random.shuffle(items)
        t0 = time.time()
        for step in range(steps_per_epoch):
            batch = items[step * args.batch_size:(step + 1) * args.batch_size]
            infos = []
            for _, q, a, lt in batch:
                info = lang_info_data(question=q, answer=a, lang_type=lt)
                info.append_prompt_postfix()
                info.query_ids_question, info.query_ids_answer = [], []
                infos.append(info)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = llm(batch_input_text_list=[i.question for i in infos],
                        batch_output_text_list=[i.answer for i in infos],
                        batch_instance_queries_hidden_state=[feats[s]["hidden"].cuda() for s, *_ in batch],
                        batch_instance_queries_normalized_embed=[feats[s]["embed"].cuda() for s, *_ in batch],
                        batch_eval_types=[i.lang_type for i in infos],
                        batch_gt_inst_ids=[(0, i) for i in infos])
            loss = o["lm_loss"] + o["match_loss"]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 0.1)
            opt.step()
            sched.step()
            if step % 20 == 0:
                print(f"epoch {epoch} step {step}/{steps_per_epoch} lm_loss {float(o['lm_loss']):.4f} "
                      f"lr {sched.get_last_lr()[0]:.2e} {time.time() - t0:.0f}s", flush=True)
        torch.save({n: p.detach().cpu() for n, p in llm.named_parameters() if p.requires_grad},
                   out / f"trainable_epoch{epoch}.pt")
        evaluate(model, args.val_items, feats, out / f"epoch{epoch}", args.placement)
        print(f"EPOCH_DONE {epoch}", flush=True)
    print("FINETUNE_DONE")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stage", choices=["cache", "train"])
    ap.add_argument("--placement", required=True, choices=["rc", "glb"])
    ap.add_argument("--processed_dir", required=True, help="preprocess_xr.py xr_<placement>/scannet200 (train + val)")
    ap.add_argument("--cache_dir", required=True, help="per-scene detector query features")
    ap.add_argument("--ckpt", default="saved/step3_mask3d_lang_4GPUS/last-epoch.ckpt")
    ap.add_argument("--shard", default="0/1", help="cache: i/n")
    ap.add_argument("--train_items")
    ap.add_argument("--val_items")
    ap.add_argument("--out_dir")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if args.out_dir:
        Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    cache(args) if args.stage == "cache" else train(args)


if __name__ == "__main__":
    main()
