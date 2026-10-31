#!/usr/bin/env python3
"""Run the released Grounded 3D-LLM on ScanRefer / Multi3DRefer / Scan2Cap / ScanQA / SQA3D / Real-3DQA / Beacon3D / XR-QA / XR-SceneCaption.

Per scene, the frozen Mask3D-CLASP detector is run exactly as in the official test pass
(``final_scripts/step3_train_grounded3dllm.sh`` TEST: ``data=indoor_dialog``,
``model=mask3d_lang``, ``tiny_vicuna_len512``), including the text it is fed (ScanNet200
class names + the scene's ScanRefer / Multi3DRefer sentences), so the 100 object queries
the LLM sees are the ones behind the paper's ScanQA numbers. Our questions are then asked
through ``llama_model.evaluate`` with the official decoding (beam 3, repetition penalty
1.2, max 150 new tokens):

* QA (SQA3D, Real-3DQA, Beacon3D QA): ``lang_type="scanqa:...:text_only"``, i.e. the
  ScanQA prompt ``<question> Please answer with a single word or phrase.``; situated
  benchmarks put the situation in front of the question (``<situation> <question>``).
  Grounded 3D-LLM is not trained on SQA3D, so these numbers are zero-shot.
* XR-QA / XR-SceneCaption (``--domain xr_rc|xr_glb``, scenes from ``extra_eval/xr/preprocess_xr.py``):
  XR-QA as QA above (ScanQA prompt); XR-SceneCaption asks its question ("Describe the
  room.") as is, with ``lang_type="globalscenecap:extra:text_only"`` (no suffix).
* Grounding (Beacon3D grounding): ``lang_type="scanrefer:...:with_grounding"`` with the
  first ScanRefer template + `` (with grounding)``; the first ``<ref>`` token is matched
  to a proposal as in ``LLama3d.evaluate`` and the proposal mask is mapped to the ScanNet
  object with the highest point IoU.

* Native benchmarks (ScanRefer, Multi3DRefer, Scan2Cap, ScanQA, all ScanNet val): the
  model's own test pass (``ModelingGrounded3DLLM.eval_step``) minus the detection prompts:
  ``prepare_llm`` builds the questions from the release's ``data/langdata`` (random prompt
  templates, here seeded per scene), ``llama_model.evaluate`` answers them, and
  ``eval_llm_iou_score`` scores the grounding, exactly as upstream; each scene is written as
  ``<out_dir>/native/<scene>.json`` (+ ``native/m3drefer/<scene>.pkl``) in the format that
  ``models/metrics/evaluate_LLM.py`` reads.

All questions of a scene are asked after one detector pass. ``--prefix_cache 1`` also
encodes the LLM prefix shared by every question of the scene (100 instance tokens + system
prompt) once and reuses its KV cache (``LLama3dForCausalLM._use_scene_prefix_cache``).

Writes ``<out_dir>/<bench>.jsonl`` (one line per question, with the raw generation).
"""

import argparse
import json
import os
import pickle
import random
import sys
import time
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
os.chdir(REPO)  # the code reads ./data, ./pretrained, ./conf relative to the repo root

SPECIAL = ["ASSISTANT:", "\n", "<s>", "</s>", "<unk>", "<p>", "</p>", "<ref>"]


def clean_generation(text):
    """``special_token_filter`` of models/metrics/evaluate_LLM.py without the LEO clean."""
    for t in SPECIAL:
        text = text.replace(t, "")
    return " ".join(text.split())


# ----------------------------------------------------------------------------- benchmarks
def load_sqa3d(d):
    q = json.load(open(os.path.join(d, "v1_balanced_questions_test_scannetv2.json")))["questions"]
    a = {x["question_id"]: x for x in json.load(
        open(os.path.join(d, "v1_balanced_sqa_annotations_test_scannetv2.json")))["annotations"]}
    for x in q:
        yield dict(bench="sqa3d", task="qa", id=str(x["question_id"]), scene_id=x["scene_id"],
                   text=f'{x["situation"]} {x["question"]}', question=x["question"],
                   answers=[y["answer"] for y in a[x["question_id"]]["answers"]])


def load_real3dqa(d):
    for split in ["debiased_test", "rotation_0", "rotation_90", "rotation_180", "rotation_270"]:
        for line in open(os.path.join(d, f"{split}.jsonl")):
            x = json.loads(line)
            yield dict(bench=f"real3dqa_{split}", task="qa", id=str(x["question_id"]),
                       scene_id=x["scene_id"], text=f'{x["situation"]} {x["question"]}',
                       question_type=x["question_type"], answers=[x["answer"]])


def load_beacon3d(d, domain="scannet"):
    for i, x in enumerate(json.load(open(os.path.join(d, domain, f"{domain}_grounding.json")))):
        yield dict(bench=f"beacon3d_{domain}_grounding", task="grounding", id=x["item_id"], idx=i,
                   scene_id=x["scan_id"], text=x["utterance"], target_id=x["target_id"])
    for i, x in enumerate(json.load(open(os.path.join(d, domain, f"{domain}_qa.json")))):
        yield dict(bench=f"beacon3d_{domain}_qa", task="qa", id=x["question_id"], idx=i,
                   scene_id=x["scene_id"], text=x["question"], answers=x["answers"])


def load_xr(items_file, placement):
    for it in json.load(open(items_file)):
        yield dict(bench=f"xr{it['task']}_{placement}", task=it["task"], id=it["key"], idx=it["idx"],
                   scene_id=it["scene_id"], text=it["question"], answers=it["answers"], key=it["key"],
                   **({"type": it["type"]} if "type" in it else {}))


NATIVE = ["scanrefer", "m3dref", "scan2cap", "scanqa"]  # lang_type prefixes in data/langdata


# ----------------------------------------------------------------------------- model
def build_model(ckpt, save_dir, domain="scannet", processed_dir=None, native=()):
    from hydra.experimental import compose, initialize_config_dir
    overrides = [
        "general.experiment_name=step3_mask3d_lang_4GPUS", "general.project_name=scannet200",
        "general.gpus=1", "data=indoor_dialog", "model=mask3d_lang", "data.batch_size=5",
        "data.num_workers=4", "trainer=trainer50", "optimizer.lr=0.0008",
        "general.train_mode=false", "general.filter_scene00=false",
        "general.topk_per_image=750", "general.llm_config=conf/llm/tiny_vicuna_len512.json",
        f"general.checkpoint={ckpt}", f"general.save_dir={save_dir}", "general.timestamp=infer",
        # inference only: the train split is not on disk, build that dataset from val
        "data.train_mode=validation",
        # the detector reads ScanNet200 class names + ScanRefer/Multi3DRefer sentences; the
        # instruction sets (scanqa, scan2cap, objdesc, ...) only feed the LLM's question list,
        # so only the native benchmarks we evaluate are loaded
        "data.lang_data_conf=" + "+".join(["scanrefer", "m3dref"] + [b for b in ("scanqa", "scan2cap") if b in native]),
    ]
    if domain != "scannet":
        # point-cloud domains (preprocess_pointcloud.py): no ScanRefer / Multi3DRefer text
        # exists for 3RScan / MultiScan, so the detector reads the class names only; the
        # ScanNet control drops the sentences too, to isolate the same conditions
        overrides += [f"data.test_dataset.data_dir={processed_dir}",
                      f"data.test_dataset.label_db_filepath={processed_dir}/label_database.yaml",
                      f"data.test_dataset.color_mean_std={processed_dir}/color_mean_std.yaml",
                      "data.lang_query=0"]
    if domain in ("3rscan", "multiscan") or domain.startswith("xr_"):
        overrides.append("model.axis_align_coord=false")  # SceneVerse clouds are aligned already
    with initialize_config_dir(config_dir=str(REPO / "conf")):
        cfg = compose(config_name="config_base.yaml", overrides=overrides)
    from trainer.trainer import ModelingGrounded3DLLM
    model = ModelingGrounded3DLLM(cfg)
    # The release is a Lightning checkpoint (pickled OmegaConf hparams), which torch>=2.6
    # refuses with weights_only=True; it is the authors' file, loaded only for its weights.
    state = torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]
    res = model.load_state_dict(state, strict=False)
    print(f"checkpoint {ckpt}: {len(state)} tensors, missing {len(res.missing_keys)} "
          f"{res.missing_keys[:5]}, unexpected {len(res.unexpected_keys)} {res.unexpected_keys[:5]}")
    assert not res.unexpected_keys, res.unexpected_keys
    return cfg, model.cuda().eval()


@torch.no_grad()
def scene_forward(model, batch):
    """eval_step up to the detector output: query features + full-res proposal masks."""
    import MinkowskiEngine as ME
    raw_data, target, file_names = batch
    target = [{k: (v.cuda() if torch.is_tensor(v) else v) for k, v in t.items()} for t in target]
    raw_coordinates = raw_data.features[:, -3:]
    raw_data.features = raw_data.features[:, :-3]
    data = ME.SparseTensor(coordinates=raw_data.coordinates, features=raw_data.features,
                           device=model.device)
    output = model.forward(data, point2segment=[t["point2segment"] for t in target],
                           raw_coordinates=raw_coordinates, extra_lang=raw_data.extra_lang,
                           is_eval=True)
    masks = (output["pred_masks"][0][target[0]["point2segment"].cpu()] > 0.).float().cpu()
    masks = model.get_full_res_mask(masks, raw_data.inverse_maps[0],
                                    raw_data.target_full[0]["point2segment"])
    return output, target, masks


def ask(model, items, hidden, embed):
    from datasets.language_info import lang_info_data, lang_template
    infos = []
    for it in items:
        if it["task"] == "qa":
            info = lang_info_data(question=it["text"], answer="NONE",
                                  lang_type="scanqa:extra:text_only")
        elif it["task"] == "cap":
            info = lang_info_data(question=it["text"], answer="NONE",
                                  lang_type="globalscenecap:extra:text_only")
        else:
            q = lang_template["grounding"]["questions"][0].format(grounding_text=it["text"])
            info = lang_info_data(question=q, answer="NONE",
                                  lang_type="scanrefer:extra:with_grounding")
        info.append_prompt_postfix()
        info.query_ids_question = []
        infos.append(info)
    cfg = model.llama_config
    return model.llama_model.evaluate(
        input_text_list=[i.question for i in infos],
        batch_instance_queries_hidden_state=[hidden] * len(infos),
        batch_instance_queries_normalized_embed=[embed] * len(infos),
        use_mini_batch=True, mini_batch_size=cfg.test_batch_size,
        batch_out_text=["NONE"] * len(infos), batch_eval_types=[i.lang_type for i in infos],
        batch_gt_inst_ids=[(0, i, np.nan) for i in infos])


def native_step(model, batch, output, target, masks, benches):
    """The LLM part of ModelingGrounded3DLLM.eval_step, restricted to the benchmarks asked for
    (the detection prompts and the other instruction sets are dropped). Returns the per-scene
    json of eval_step and the Multi3DRefer boxes for the official evaluator."""
    from models.metrics.evaluate_LLM import eval_llm_iou_score
    raw_data, _, file_names = batch
    infos, _ = model.prepare_llm(output, raw_data.extra_lang, None, target, raw_data,
                                 file_names=file_names)
    infos = [i for i in infos if i.lang_type.split(":")[0] in benches]
    if not infos:
        return None, []
    gt_ids = [(i.batch_idx, i, i.max_gt_iou) for i in infos]
    out_json = model.llama_model.evaluate(
        input_text_list=[i.question for i in infos],
        batch_instance_queries_hidden_state=[i.query_hidden_feature for i in infos],
        batch_instance_queries_normalized_embed=[i.query_normalized_embed for i in infos],
        use_mini_batch=True, mini_batch_size=model.llama_config.test_batch_size,
        batch_out_text=[i.answer for i in infos], batch_eval_types=[i.lang_type for i in infos],
        batch_gt_inst_ids=gt_ids)
    for item, info in zip(out_json, infos):
        if item["gt"] == "NONE":
            item["gt"] = info.answer
        item["type"] = info.lang_type
    out_json, score, b25, b50, m25, m50, m3dref_boxes = eval_llm_iou_score(
        out_json, {"pred_inst_masks": [masks], "target_full": raw_data.target_full,
                   "batch_gt_inst_ids": gt_ids, "original_coordinates": raw_data.original_coordinates})
    return dict(prediction=out_json, score=score, bbox_score_25=b25, bbox_score_50=b50,
                mask_score_25=m25, mask_score_50=m50), m3dref_boxes


class SeededScenes(torch.utils.data.Dataset):
    """Subset of the test dataset whose items seed random / np.random from the scene name, so
    the random prompt templates do not depend on worker or shard."""

    def __init__(self, ds, indices, scene_of):
        self.ds, self.indices, self.scene_of = ds, indices, scene_of

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        seed_scene(self.scene_of[self.indices[i]])
        return self.ds[self.indices[i]]


def seed_scene(scene, salt=""):
    seed = zlib.crc32(f"{scene}{salt}".encode())
    random.seed(seed)
    np.random.seed(seed)


def best_gt_object(prop_mask, inst_ids):
    """ScanNet object id (aggregation id) with the highest point IoU to a proposal mask."""
    best, best_iou = None, 0.0
    for oid in np.unique(inst_ids[prop_mask]):
        if oid < 0:
            continue
        gt = inst_ids == oid
        iou = (gt & prop_mask).sum() / (gt | prop_mask).sum()
        if iou > best_iou:
            best, best_iou = int(oid), float(iou)
    return best, best_iou


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--benches", default=",".join(NATIVE + ["sqa3d", "real3dqa", "beacon3d"]))
    ap.add_argument("--domain", default="scannet",
                    choices=["scannet", "scannet_pcseg", "3rscan", "multiscan", "xr_rc", "xr_glb"],
                    help="scene source; scannet_pcseg / 3rscan / multiscan run Beacon3D only, on "
                         "preprocess_pointcloud.py output, xr_* runs XR-QA / XR-SceneCaption on "
                         "xr/preprocess_xr.py output (--processed_dir, --xr_items)")
    ap.add_argument("--processed_dir", default=None)
    ap.add_argument("--ckpt", default="saved/step3_mask3d_lang_4GPUS/last-epoch.ckpt")
    ap.add_argument("--sqa3d_dir", default=os.environ.get("SQA3D_DIR"))
    ap.add_argument("--real3dqa_dir", default=os.environ.get("REAL3DQA_DIR"))
    ap.add_argument("--beacon3d_dir", default="extra_eval/beacon3d/Beacon3D/data")
    ap.add_argument("--xr_items", default=None, help="xr_<split>_items.json of preprocess_xr.py (xr_* domains)")
    ap.add_argument("--prefix_cache", type=int, default=0,
                    help="1: reuse the KV cache of the scene prefix shared by all its questions")
    ap.add_argument("--max_scenes", type=int, default=None, help="debug: first N scenes only")
    ap.add_argument("--shard", default="0/1", help="i/n: this process takes scenes i::n")
    args = ap.parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    items = []
    benches = args.benches.split(",")
    native = [b for b in NATIVE if b in benches] if args.domain == "scannet" else []
    if args.domain.startswith("xr_"):
        items += list(load_xr(args.xr_items, args.domain[3:]))
    elif args.domain != "scannet":
        for it in load_beacon3d(args.beacon3d_dir, "scannet" if args.domain == "scannet_pcseg" else args.domain):
            it["bench"] = it["bench"].replace("beacon3d_scannet_", f"beacon3d_{args.domain}_")
            items.append(it)
    else:
        if "sqa3d" in benches:
            items += list(load_sqa3d(args.sqa3d_dir))
        if "real3dqa" in benches:
            items += list(load_real3dqa(args.real3dqa_dir))
        if "beacon3d" in benches:
            items += list(load_beacon3d(args.beacon3d_dir, "scannet"))
    by_scene = defaultdict(list)
    for it in items:
        by_scene[it["scene_id"]].append(it)

    cfg, model = build_model(args.ckpt, str(out_dir / f"g3d_save_{args.domain}"), args.domain,
                             args.processed_dir and str(Path(args.processed_dir).resolve()), native)
    model.llama_model.config.eval_prefix_cache = bool(args.prefix_cache)
    print(f"prefix_cache={args.prefix_cache}")  # generate() always caches with inputs_embeds
    import hydra
    ds = model.test_dataset
    scene_of = [d["raw_filepath"].split("/")[-2] for d in ds.data]
    missing = sorted(set(by_scene) - set(scene_of))
    if missing:
        print(f"WARNING: {len(missing)} scenes not preprocessed, their questions are skipped: {missing}")
    # the native benchmarks cover (nearly) every val scene; scenes without any of their
    # sentences simply produce no native output
    todo = [i for i, s in enumerate(scene_of) if native or s in by_scene]
    if args.max_scenes:
        todo = todo[:args.max_scenes]
    k, n = map(int, args.shard.split("/"))
    todo = todo[k::n]
    loader = torch.utils.data.DataLoader(
        SeededScenes(ds, todo, scene_of), batch_size=1, shuffle=False, num_workers=4,
        collate_fn=hydra.utils.instantiate(cfg.data.test_collation))

    suffix = "" if n == 1 else f".shard{k}of{n}"
    files = {}
    if native:
        (out_dir / "native" / "m3drefer").mkdir(parents=True, exist_ok=True)
    timing = open(out_dir / f"timing_{args.domain}{suffix}.jsonl", "w")
    for si, batch in enumerate(loader):
        scene = batch[2][0]
        torch.cuda.synchronize()
        t0 = time.time()
        output, target, masks = scene_forward(model, batch)
        torch.cuda.synchronize()
        t1 = time.time()
        n_native = 0
        if native:
            seed_scene(scene, "llm")
            scene_json, m3dref_boxes = native_step(model, batch, output, target, masks, native)
            if scene_json is not None:
                n_native = len(scene_json["prediction"])
                with open(out_dir / "native" / f"{scene}.json", "w") as f:
                    json.dump(scene_json, f)
                if m3dref_boxes:
                    with open(out_dir / "native" / "m3drefer" / f"{scene}.pkl", "wb") as f:
                        pickle.dump(m3dref_boxes, f)
        torch.cuda.synchronize()
        t2 = time.time()
        uniq = {}
        if by_scene.get(scene):
            hidden, embed = output["queries_hidden_state"][0], output["queries_normalized_embed"][0]
            masks_np = np.asarray(masks).astype(bool)
            inst_ids = np.load(ds.data[todo[si]]["filepath"])[:, 11].astype(np.int64)
            assert len(inst_ids) == masks_np.shape[0], (scene, len(inst_ids), masks_np.shape)
            # one generation per distinct (task, text) in the scene; Real-3DQA splits overlap
            for it in by_scene[scene]:
                uniq.setdefault((it["task"], it["text"]), it)
            outs = ask(model, list(uniq.values()), hidden, embed)
            gen = dict(zip(uniq.keys(), outs))
            for it in by_scene[scene]:
                o = gen[(it["task"], it["text"])]
                rec = dict(it, raw=o["output_language"], pred=clean_generation(o["output_language"]))
                if it["task"] == "grounding":
                    q = None
                    for r in (o["grounding_result"] or []):
                        if r:
                            q = int(r[0])
                            break
                    rec["pred_query"] = q
                    rec["pred_object_id"], rec["pred_iou"] = (
                        best_gt_object(masks_np[:, q], inst_ids) if q is not None else (None, 0.0))
                if it["bench"] not in files:
                    files[it["bench"]] = open(out_dir / f'{it["bench"]}{suffix}.jsonl', "w")
                files[it["bench"]].write(json.dumps(rec) + "\n")
            for f in files.values():
                f.flush()
        torch.cuda.synchronize()
        t3 = time.time()
        timing.write(json.dumps(dict(scene=scene, detector=t1 - t0, native=t2 - t1, n_native=n_native,
                                     extra=t3 - t2, n_extra=len(uniq))) + "\n")
        timing.flush()
        print(f"[{si + 1}/{len(todo)}] {scene}: {n_native} native, {len(by_scene.get(scene, []))} other "
              f"items ({len(uniq)} generations) in {t3 - t0:.1f}s", flush=True)
    for f in files.values():
        f.close()
    timing.close()
    print("INFER_DONE")


if __name__ == "__main__":
    main()
