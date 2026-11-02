#!/usr/bin/env python
"""DCF + majority-vote eval for one pool (test|ext143), fixed checkpoint, no tuning.
Same policy as dcf_eval.py: empty mask -> UNCLASSIFIABLE, never forced."""
import os, csv, json, sys
import numpy as np
import SimpleITK as sitk
from scipy import ndimage
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix

pool, pred_dir, out_json = sys.argv[1], sys.argv[2], sys.argv[3]
root = os.path.expanduser("~/Repos/idh-mcnet")
gt = {}
with open(os.path.join(root, "ucsf493_labels_pools.csv")) as f:
    for row in csv.DictReader(f):
        if row["pool"] == pool:
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
    sizes = ndimage.sum(np.ones_like(lab), lab, range(1, n + 1))
    biggest = int(np.argmax(sizes)) + 1
    votes = arr[lab == biggest]
    n_mut = int((votes == 1).sum()); n_wt = int((votes == 2).sum())
    if n_mut + n_wt == 0:
        unclassifiable.append(pid); continue
    score = n_mut / (n_mut + n_wt)
    results.append({"pid": pid, "gt": gt[pid], "call": 1 if n_mut > n_wt else 0,
                    "score": score, "n_mut_vox": n_mut, "n_wt_vox": n_wt})

y = np.array([r["gt"] for r in results]); c = np.array([r["call"] for r in results])
s = np.array([r["score"] for r in results])
tn, fp, fn, tp = confusion_matrix(y, c, labels=[0, 1]).ravel()
out = {
  "pool": pool, "n_pool": len(gt), "n_evaluated": len(results),
  "unclassifiable": unclassifiable, "missing_pred": missing_pred,
  "accuracy": float((tp + tn) / max(1, len(results))),
  "sensitivity": float(tp / max(1, tp + fn)) if tp + fn else None,
  "specificity": float(tn / max(1, tn + fp)) if tn + fp else None,
  "balanced_accuracy": float((tp / max(1, tp + fn) + tn / max(1, tn + fp)) / 2) if tp + fn and tn + fp else None,
  "auroc": float(roc_auc_score(y, s)) if len(set(y)) > 1 else None,
  "auprc": float(average_precision_score(y, s)) if len(set(y)) > 1 else None,
  "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
  "checkpoint": "Dataset002 fold_0 checkpoint_best.pth epoch121 sha256 0b034ee40e5c4f9d (FIXED, no retuning)",
}
print(json.dumps(out, indent=1))
with open(out_json, "w") as f:
    json.dump({"summary": out, "per_case": results}, f, indent=1)
