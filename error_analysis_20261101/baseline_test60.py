#!/usr/bin/env python
"""Model A (2.5D EfficientNet-B0 baseline, needs GT mask) inference + timing on TEST-60.
Uses the faithful port code unchanged; reads best.pt; writes only into error_analysis_20261101/.
Threshold 0.5 predeclared (same as dev-47 protocol). No training, no selection."""
import os, sys, csv, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path.home() / "Repos/idh-mcnet"
sys.path.insert(0, str(ROOT / "baseline2p5d"))
import idh_core as C

OUT = ROOT / "error_analysis_20261101"
CACHE = OUT / "cache_a"
CACHE.mkdir(parents=True, exist_ok=True)

# Build test60 manifest (same columns as training manifest)
rows = []
with open(ROOT / "ucsf493_labels_pools.csv") as f:
    for r in csv.DictReader(f):
        if r["pool"] == "test":
            pid = r["patient_id"]
            d = ROOT / "data/ucsf_core350" / pid
            rows.append({
                "patient_id": pid, "idh_label": r["idh_mutant"],
                "t1_path": d / f"{pid}_T1.nii.gz", "t1ce_path": d / f"{pid}_T1c.nii.gz",
                "t2_path": d / f"{pid}_T2.nii.gz", "flair_path": d / f"{pid}_FLAIR.nii.gz",
                "seg_path": d / f"{pid}_tumor_segmentation.nii.gz"})
mdf = pd.DataFrame(rows)
man = OUT / "test60_manifest.csv"
mdf.to_csv(man, index=False)
df = C.load_manifest(str(man))
print(f"test60 manifest: {len(df)} patients; mutant={int(df.idh_label.sum())}")

model, cfg, meta = C.load_checkpoint(ROOT / "baseline2p5d/results/best.pt", device="cuda")
cfg.cache_dir = str(CACHE)
cfg.mixed_precision = False  # match predict_loader semantics (fp32 forward)
model.eval()
print(f"checkpoint epoch={meta.get('epoch')} best_score={meta.get('best_score')}")

ds = C.PatientDataset(df, cfg)

# Phase A: per-case preprocessing + slice selection (cold cache; includes disk)
t_pre = []
for i in range(len(ds)):
    t0 = time.perf_counter()
    x, y, pid = ds[i]
    t_pre.append((pid, time.perf_counter() - t0))

# Phase B: per-case model forward, batch=1, warm cache
preds, t_fwd = [], []
with torch.no_grad():
    for i in range(len(ds)):
        t0 = time.perf_counter()
        x, y, pid = ds[i]
        t_load = time.perf_counter() - t0
        xb = x.unsqueeze(0).to("cuda")
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        logit = float(model(xb).float().cpu().numpy()[0])
        torch.cuda.synchronize()
        t_fw = time.perf_counter() - t0
        prob = float(torch.sigmoid(torch.tensor(logit)).item())
        preds.append({"patient_id": pid, "idh_label": int(y.item()), "probability": prob})
        t_fwd.append({"pid": pid, "load_s": round(t_load, 4), "forward_s": round(t_fw, 4)})

pdf = pd.DataFrame(preds)
pdf.to_csv(OUT / "test60_predictions_modelA.csv", index=False)
m = C.metrics_binary(pdf.idh_label.to_numpy(), pdf.probability.to_numpy(), threshold=0.5)
with open(OUT / "test60_metrics_modelA.json", "w") as f:
    json.dump(m, f, indent=1)

pre = np.array([t for _, t in t_pre]); fw = np.array([t["forward_s"] for t in t_fwd])
ld = np.array([t["load_s"] for t in t_fwd])
timing = {
  "n": len(ds),
  "preprocess_cold_s": {"mean": round(float(pre.mean()), 3), "median": round(float(np.median(pre)), 3), "total": round(float(pre.sum()), 1)},
  "forward_warm_s": {"mean": round(float(fw.mean()), 4), "median": round(float(np.median(fw)), 4), "min": round(float(fw.min()), 4), "max": round(float(fw.max()), 4), "total": round(float(fw.sum()), 2)},
  "load_warm_s": {"mean": round(float(ld.mean()), 4), "median": round(float(np.median(ld)), 4)},
  "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 1e6, 1),
  "note": "Model A requires a supplied GT tumor mask for slice centering; timing excludes mask creation (human/segmenter input).",
}
with open(OUT / "timing_modelA.json", "w") as f:
    json.dump(timing, f, indent=1)
print(json.dumps({"metrics": m, "timing": timing}, indent=1))
print("BASELINE-DONE")
