#!/usr/bin/env python3
"""Score Grounded 3D-LLM predictions on XR-QA / XR-SceneCaption.

Metrics are LSceneLLM's (``utils/capeval``, vendored in ``capeval/``: BLEU-4, CIDEr,
METEOR, ROUGE-L, x100), with predictions and references keyed by LSceneLLM's episode id
(repeated keys keep the last item, as in its validation). Two text treatments:

* ``raw``  LSceneLLM's validation: the decoded text, repeated spaces collapsed.
* ``g3d``  Grounded 3D-LLM's own (``models/metrics/evaluate_LLM.py``): ``special_token_filter``
  (+ LEO's ``clean_answer`` for QA, as for ScanQA; none for captions, as for Scan2Cap) on
  predictions and references, then the PTB tokenizer. LSceneLLM's Table 1 baselines match
  their papers' own normalisation (checked on Chat-Scene), so this is the comparable one.
  ``special_token_filter`` also cuts predictions and references to 256 characters, harmless
  for short answers but for the ~87-word XR captions it keeps only their formulaic first
  ~40 words and inflates CIDEr ~5x.
* ``g3d_full``  the same without the 256-character cut (identical to ``g3d`` for QA); the
  one to compare captions across models.

Also splits by the number of SceneVerse objects in the scene (<= 100 / > 100), as for
Chat-Scene.

    python extra_eval/xr/score_xr.py --preds <out>/xrqa_rc*.jsonl --items xr_val_items.json
"""

import argparse
import json
import os
import sys
from collections import OrderedDict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
from capeval.bleu.bleu import Bleu  # noqa: E402
from capeval.cider.cider import Cider  # noqa: E402
from capeval.meteor.meteor import Meteor  # noqa: E402
from capeval.rouge.rouge import Rouge  # noqa: E402


def score(corpus, cands):
    keys = list(cands)
    gts = OrderedDict((k, corpus[k]) for k in keys)
    res = OrderedDict((k, cands[k]) for k in keys)
    bleu = Bleu(4).compute_score(gts, res)
    cider = Cider().compute_score(gts, res)
    rouge = Rouge().compute_score(gts, res)
    meteor = Meteor().compute_score(gts, res)
    return {"n": len(keys), "CIDEr": 100 * cider[0], "BLEU-4": 100 * bleu[0][3],
            "METEOR": 100 * meteor[0], "ROUGE-L": 100 * rouge[0]}


def g3d_norm(corpus, cands, qa, truncation=True):
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    from models.metrics.evaluate_LLM import Evaluator
    ev = Evaluator.__new__(Evaluator)  # special_token_filter needs no state
    f = lambda s: ev.special_token_filter(s, clean=qa, truncation=truncation)  # noqa: E731
    tok = PTBTokenizer()
    return (tok.tokenize({k: [{"caption": f(c)} for c in v] for k, v in corpus.items()}),
            tok.tokenize({k: [{"caption": f(v[0])}] for k, v in cands.items()}))


def splits(corpus, cands, n_obj):
    out = {"all": score(corpus, cands)}
    for name, keep in (("le100", lambda n: n <= 100), ("gt100", lambda n: n > 100)):
        sub = {k: v for k, v in cands.items() if n_obj and keep(n_obj[k])}
        if sub:
            out[name] = score(corpus, sub)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--preds", nargs="+", required=True, help="<bench>[.shardKofN].jsonl of infer.py")
    ap.add_argument("--items", required=True, help="xr_<split>_items.json of preprocess_xr.py")
    ap.add_argument("--scene_objects", default=None,
                    help="SceneVerse scan_data dir; default $XR_SCAN_DATA (for the <=100 / >100 split)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    recs = {}
    for f in args.preds:
        for line in open(f):
            r = json.loads(line)
            recs[r["idx"]] = r
    items = json.load(open(args.items))
    task = recs[next(iter(recs))]["task"]
    items = [it for it in items if it["task"] == task]
    missing = [it["idx"] for it in items if it["idx"] not in recs]
    if missing:
        raise SystemExit(f"{len(missing)} of {len(items)} {task} items have no prediction")

    scan_data = args.scene_objects or os.environ.get("XR_SCAN_DATA")
    n_obj_scene = {}
    if scan_data:
        import torch
        for it in items:
            if it["scene_id"] not in n_obj_scene:
                n_obj_scene[it["scene_id"]] = sum(
                    len(torch.load(os.path.join(scan_data, "instance_id_to_label", f"{it['scan']}_{r}.pth"),
                                   weights_only=False)) for r in it["regions"])
    corpus, cands, n_obj = {}, {}, {}
    for it in items:  # LSceneLLM keys by episode id: repeated keys keep the last item
        corpus[it["key"]] = list(it["answers"])
        cands[it["key"]] = [" ".join(recs[it["idx"]]["pred"].split())]
        n_obj[it["key"]] = n_obj_scene.get(it["scene_id"])
    if not scan_data:
        n_obj = {}
    out = {"raw": splits(corpus, cands, n_obj),
           "g3d": splits(*g3d_norm(corpus, cands, task == "qa"), n_obj),
           "g3d_full": splits(*g3d_norm(corpus, cands, task == "qa", truncation=False), n_obj),
           "mean_pred_words": sum(len(v[0].split()) for v in cands.values()) / len(cands)}
    json.dump(out, open(args.out, "w"), indent=1)
    for norm in ("raw", "g3d", "g3d_full"):
        for sp, v in out[norm].items():
            print(f"{Path(args.out).stem:16} {norm:8} {sp:5}", {m: round(x, 2) for m, x in v.items()})
    print(f"{Path(args.out).stem:16} mean_pred_words {out['mean_pred_words']:.1f}")


if __name__ == "__main__":
    main()
