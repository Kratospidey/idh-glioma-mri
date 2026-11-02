#!/usr/bin/env python
"""Overlay figure: reference vs predicted tumor masks on FLAIR for error cases + controls.
Rows: one per case. Cols: FLAIR axial slice at GT tumor center with GT contour (green)
and predicted contour (red), then zoomed crop. Reads audit CSVs; writes PNGs to
error_analysis_20261101/overlays/."""
import os, json
import numpy as np
import pandas as pd
import SimpleITK as sitk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.expanduser("~/Repos/idh-mcnet")
OUT = os.path.join(ROOT, "error_analysis_20261101")
OVL = os.path.join(OUT, "overlays"); os.makedirs(OVL, exist_ok=True)

def flair_path(pid, pool):
    base = "data/ucsf_core350" if pool == "test" else "data/ucsf_ext"
    return os.path.join(ROOT, base, pid, f"{pid}_FLAIR.nii.gz")

def ref_path(pid, pool):
    base = "data/ucsf_core350" if pool == "test" else "data/ucsf_ext"
    return os.path.join(ROOT, base, pid, f"{pid}_tumor_segmentation.nii.gz")

def pred_path(pid, pool):
    d = "test60_preds" if pool == "test" else "ext143_preds"
    return os.path.join(ROOT, "mcnet_eval", d, f"{pid}.nii.gz")

def load_case(pid, pool):
    fl = sitk.GetArrayFromImage(sitk.ReadImage(flair_path(pid, pool))).astype(np.float32)
    ref = sitk.GetArrayFromImage(sitk.ReadImage(ref_path(pid, pool))) > 0
    pr = sitk.GetArrayFromImage(sitk.ReadImage(pred_path(pid, pool))) > 0
    return fl, ref, pr

def render(pid, pool, gt, call, score, dice, fname):
    fl, ref, pr = load_case(pid, pool)
    zz = np.where(ref.any(axis=(1, 2)))[0]
    z = int(zz[len(zz) // 2]) if len(zz) else fl.shape[0] // 2
    img = fl[z]; r = ref[z]; p = pr[z]
    v = img[img > 0]
    vmin, vmax = (np.percentile(v, [1, 99]) if v.size else (0, 1))
    ys, xs = np.where(r | p)
    if len(ys):
        y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
        pad = 25
        y0, y1 = max(0, y0 - pad), min(img.shape[0], y1 + pad)
        x0, x1 = max(0, x0 - pad), min(img.shape[1], x1 + pad)
    else:
        y0, y1, x0, x1 = 0, img.shape[0], 0, img.shape[1]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    for ax, (sl, rr, pp), ttl in zip(axes, [(img, r, p), (img[y0:y1, x0:x1], r[y0:y1, x0:x1], p[y0:y1, x0:x1])],
                                     ["axial FLAIR", "zoom"]):
        ax.imshow(sl, cmap="gray", vmin=vmin, vmax=vmax)
        if rr.any(): ax.contour(rr, levels=[0.5], colors="lime", linewidths=1.4)
        if pp.any(): ax.contour(pp, levels=[0.5], colors="red", linewidths=1.2, linestyles="dashed")
        ax.set_title(ttl); ax.axis("off")
    verdict = "ERROR" if call != gt else "correct"
    fig.suptitle(f"{pid} [{pool}]  gt={'mutant' if gt else 'wildtype'}  MC-net call={'mutant' if call else 'wildtype'}"
                 f"  mut-fraction={score:.2f}  Dice={dice:.2f}  -> {verdict}\n"
                 "green = reference mask, red dashed = MC-net predicted mask", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OVL, fname), dpi=130)
    plt.close(fig)

with open(os.path.join(OUT, "audit_summary.json")) as f:
    summary = json.load(f)

cases = []
for pool in ("test", "ext143"):
    for e in summary[pool]["errors"]:
        cases.append((e["pid"], pool, e["gt"], e["call"], e["score"], e["dice"]))

# controls: 2 correct test60 cases with median dice
ta = pd.read_csv(os.path.join(OUT, "test_audit.csv"))
corr = ta[(ta["call"] == ta["gt"]) & ta["dice"].notna()].sort_values("dice")
med = corr.iloc[len(corr) // 2]
cases.append((med.pid, "test", int(med["gt"]), int(med["call"]), float(med.score), float(med.dice)))

for i, (pid, pool, gt, call, score, dice) in enumerate(cases):
    render(pid, pool, int(gt), int(call), score or 0.0, dice or 0.0, f"overlay_{i:02d}_{pid}.png")
    print("rendered", pid, pool)

# combined grid of error cases (test60 only, for the report figure)
errs_t = [c for c in cases[:-1] if c[1] == "test"]
if errs_t:
    n = len(errs_t)
    fig, axes = plt.subplots(n, 1, figsize=(6, 3.2 * n))
    if n == 1: axes = [axes]
    for ax, (pid, pool, gt, call, score, dice) in zip(axes, errs_t):
        fl, ref, pr = load_case(pid, pool)
        zz = np.where(ref.any(axis=(1, 2)))[0]
        z = int(zz[len(zz) // 2]) if len(zz) else fl.shape[0] // 2
        img = fl[z]; v = img[img > 0]
        vmin, vmax = (np.percentile(v, [1, 99]) if v.size else (0, 1))
        ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
        if ref[z].any(): ax.contour(ref[z], levels=[0.5], colors="lime", linewidths=1.4)
        if pr[z].any(): ax.contour(pr[z], levels=[0.5], colors="red", linewidths=1.2, linestyles="dashed")
        ax.set_title(f"{pid}  gt={'mut' if gt else 'wt'} call={'mut' if call else 'wt'} score={score:.2f} dice={dice:.2f}",
                     fontsize=9)
        ax.axis("off")
    fig.suptitle("MC-net TEST-60 misclassifications (green=reference, red=predicted)", fontsize=11, y=0.998)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_error_overlays.png"), dpi=130)
    plt.close(fig)
print("OVERLAYS-DONE")
