#!/usr/bin/env python
"""Audited DCF + majority-vote eval for Yogananda-style MC-net (Dataset002_IDHMC), fold-0 47 val cases.
Class mapping: 1=IDH_mutant, 2=IDH_wildtype (verified vs Yogananda 2023 convention).
Policy: empty predicted mask -> UNCLASSIFIABLE (counted separately, never forced to a class).
DCF: combine class1+class2 -> largest 3D connected component -> majority voxel vote.
Score for AUROC/AUPRC: mutant voxel fraction within the largest component."""
import os, csv, json, glob, sys
import numpy as np
import SimpleITK as sitk
from scipy import ndimage
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix

root = os.path.expanduser("~/Repos/idh-mcnet")
pred_dir = os.path.join(root, "mcnet_eval/val_preds")
labels_csv = os.path.join(root, "ucsf493_labels_pools.csv")
splits = json.load(open(os.path.join(root, "nnUNet_preprocessed/Dataset002_IDHMC/splits_final.json")))
val_ids = set(splits[0]["val"])

gt = {}
with open(labels_csv) as f:
    for row in csv.DictReader(f):
        if row["patient_id"] in val_ids:
            gt[row["patient_id"]] = int(row["idh_mutant"])

results, unclassifiable, missing_pred = [], [], []
for pid in sorted(gt):
    pf = os.path.join(pred_dir, pid + ".nii.gz")
    if not os.path.exists(pf):
        missing_pred.append(pid); continue
    arr = sitk.GetArrayFromImage(sitk.ReadImage(pf))
    fg = arr > 0
    if fg.sum() == 0:
        unclassifiable.append(pid); continue
    lab, n = ndimage.label(fg)
    if n == 0:
        unclassifiable.append(pid); continue
    sizes = ndimage.sum(np.ones_like(lab), lab, range(1, n + 1))
    biggest = int(np.argmax(sizes)) + 1
    votes = arr[lab == biggest]
    n_mut = int((votes == 1).sum()); n_wt = int((votes == 2).sum())
    if n_mut + n_wt == 0:
        unclassifiable.append(pid); continue
    score = n_mut / (n_mut + n_wt)
    call = 1 if n_mut > n_wt else 0
    results.append({"pid": pid, "gt": gt[pid], "call": call, "score": score,
                    "n_mut_vox": n_mut, "n_wt_vox": n_wt})

y = np.array([r["gt"] for r in results]); c = np.array([r["call"] for r in results])
s = np.array([r["score"] for r in results])
tn, fp, fn, tp = confusion_matrix(y, c, labels=[0, 1]).ravel()
out = {
  "n_val": len(gt), "n_evaluated": len(results),
  "unclassifiable": unclassifiable, "missing_pred": missing_pred,
  "accuracy": float((tp + tn) / max(1, len(results))),
  "sensitivity": float(tp / max(1, tp + fn)), "specificity": float(tn / max(1, tn + fp)),
  "balanced_accuracy": float((tp / max(1, tp + fn) + tn / max(1, tn + fp)) / 2),
  "auroc": float(roc_auc_score(y, s)) if len(set(y)) > 1 else None,
  "auprc": float(average_precision_score(y, s)) if len(set(y)) > 1 else None,
  "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
  "note": "DEVELOPMENT validation on fold-0 47 val only; quarantined VAL-58/TEST-60 untouched."}
print(json.dumps(out, indent=1))
with open(os.path.join(root, "mcnet_eval/dcf_results.json"), "w") as f:
    json.dump({"summary": out, "per_case": results}, f, indent=1)
