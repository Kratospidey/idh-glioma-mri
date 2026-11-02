#!/usr/bin/env python
"""Error audit + mask quality for MC-net (Model B) on test60 and ext143.
Read-only on existing artifacts; writes only into error_analysis_20261101/.
Dice/IoU: predicted fused tumor mask (arr>0) vs reference tumor_segmentation (>0).
Also measures DCF majority-vote post-processing time per case."""
import os, csv, json, time
import numpy as np
import SimpleITK as sitk
from scipy import ndimage

ROOT = os.path.expanduser("~/Repos/idh-mcnet")
OUT = os.path.join(ROOT, "error_analysis_20261101")
os.makedirs(OUT, exist_ok=True)

POOLS = {
    "test":   {"pred_dir": os.path.join(ROOT, "mcnet_eval/test60_preds"),
               "ref_tpl":  os.path.join(ROOT, "data/ucsf_core350/{p}/{p}_tumor_segmentation.nii.gz"),
               "dcf_json": os.path.join(ROOT, "mcnet_eval/dcf_test60.json")},
    "ext143": {"pred_dir": os.path.join(ROOT, "mcnet_eval/ext143_preds"),
               "ref_tpl":  os.path.join(ROOT, "data/ucsf_ext/{p}/{p}_tumor_segmentation.nii.gz"),
               "dcf_json": os.path.join(ROOT, "mcnet_eval/dcf_ext143.json")},
}

gt = {}
with open(os.path.join(ROOT, "ucsf493_labels_pools.csv")) as f:
    for row in csv.DictReader(f):
        gt[row["patient_id"]] = (row["pool"], int(row["idh_mutant"]))

def load(path):
    return sitk.ReadImage(path)

def dice_iou(a, b):
    a = a > 0; b = b > 0
    inter = int((a & b).sum()); na = int(a.sum()); nb = int(b.sum())
    dice = 2 * inter / (na + nb) if (na + nb) else (1.0 if na == nb else 0.0)
    union = int((a | b).sum())
    iou = inter / union if union else 1.0
    return dice, iou, na, nb

def dcf_vote(arr):
    fg = arr > 0
    if fg.sum() == 0:
        return None, 0, 0
    lab, n = ndimage.label(fg)
    sizes = ndimage.sum(np.ones_like(lab), lab, range(1, n + 1))
    biggest = int(np.argmax(sizes)) + 1
    votes = arr[lab == biggest]
    n_mut = int((votes == 1).sum()); n_wt = int((votes == 2).sum())
    score = n_mut / (n_mut + n_wt) if (n_mut + n_wt) else None
    return score, n_mut, n_wt

all_rows = {}
for pool, cfgp in POOLS.items():
    rows = []
    pids = sorted(p for p, (pl, _) in gt.items() if pl == pool)
    for pid in pids:
        pf = os.path.join(cfgp["pred_dir"], pid + ".nii.gz")
        rf = cfgp["ref_tpl"].format(p=pid)
        if not os.path.exists(pf):
            rows.append({"pid": pid, "pool": pool, "gt": gt[pid][1], "error": "missing_pred"})
            continue
        pimg = load(pf); parr = sitk.GetArrayFromImage(pimg)
        t0 = time.perf_counter()
        score, n_mut, n_wt = dcf_vote(parr)
        vote_ms = (time.perf_counter() - t0) * 1000
        call = (1 if n_mut > n_wt else 0) if score is not None else None
        row = {"pid": pid, "pool": pool, "gt": gt[pid][1], "call": call,
               "score": score, "n_mut_vox": n_mut, "n_wt_vox": n_wt,
               "vote_ms": round(vote_ms, 2)}
        if call is None:
            row["error"] = "unclassifiable"
        elif os.path.exists(rf):
            rimg = load(rf)
            rarr = sitk.GetArrayFromImage(rimg)
            if rarr.shape != parr.shape:
                rimg = sitk.Resample(rimg, pimg, sitk.Transform(), sitk.sitkNearestNeighbor,
                                     0, rimg.GetPixelID())
                rarr = sitk.GetArrayFromImage(rimg)
                row["ref_resampled"] = True
            d, iou, npred, nref = dice_iou(parr, rarr)
            row.update({"dice": round(d, 4), "iou": round(iou, 4),
                        "pred_vox": npred, "ref_vox": nref})
        else:
            row["error"] = "missing_ref"
        rows.append(row)
    all_rows[pool] = rows

# write CSVs
import pandas as pd
for pool, rows in all_rows.items():
    pd.DataFrame(rows).to_csv(os.path.join(OUT, f"{pool}_audit.csv"), index=False)

summary = {}
for pool, rows in all_rows.items():
    ev = [r for r in rows if "dice" in r and r.get("call") is not None]
    errs = [r for r in ev if r["call"] != r["gt"]]
    corr = [r for r in ev if r["call"] == r["gt"]]
    def stats(rs, k):
        v = [r[k] for r in rs]
        return {"n": len(v), "mean": round(float(np.mean(v)), 4),
                "median": round(float(np.median(v)), 4),
                "min": round(float(np.min(v)), 4), "max": round(float(np.max(v)), 4)} if v else {"n": 0}
    summary[pool] = {
        "n_eval": len(ev),
        "n_errors": len(errs),
        "errors": [{"pid": r["pid"], "gt": r["gt"], "call": r["call"],
                    "score": r["score"], "dice": r.get("dice"),
                    "pred_vox": r.get("pred_vox"), "ref_vox": r.get("ref_vox")} for r in errs],
        "dice_all": stats(ev, "dice"), "dice_errors": stats(errs, "dice"),
        "dice_correct": stats(corr, "dice"), "iou_all": stats(ev, "iou"),
        "vote_ms_median": round(float(np.median([r["vote_ms"] for r in rows if "vote_ms" in r])), 2),
        "unclassifiable": [r["pid"] for r in rows if r.get("error") == "unclassifiable"],
    }
with open(os.path.join(OUT, "audit_summary.json"), "w") as f:
    json.dump(summary, f, indent=1)
print(json.dumps(summary, indent=1))
print("AUDIT-DONE")
