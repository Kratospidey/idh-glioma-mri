#!/usr/bin/env python
"""MC-net (Model B) inference timing on TEST-60, deployment-faithful.
A) single-case latency: 5 cases, one predictor, predict_from_files per case.
B) batch throughput: all 60 in one predict call (npp=3/nps=3, mirrors run_test_eval.sh).
Writes preds to error_analysis_20261101/timing_preds (existing test60_preds untouched),
then verifies majority-vote call agreement vs dcf_test60.json."""
import os, json, time, glob
import numpy as np
import torch
import SimpleITK as sitk
from scipy import ndimage

ROOT = os.path.expanduser("~/Repos/idh-mcnet")
OUT = os.path.join(ROOT, "error_analysis_20261101")
os.environ.setdefault("nnUNet_raw", os.path.join(ROOT, "nnUNet_raw"))
os.environ.setdefault("nnUNet_preprocessed", os.path.join(ROOT, "nnUNet_preprocessed"))
os.environ.setdefault("nnUNet_results", os.path.join(ROOT, "nnUNet_results"))
os.environ.setdefault("nnUNet_compile", "false")

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

def main():
    MODEL = os.path.join(ROOT, "nnUNet_results/Dataset002_IDHMC/nnUNetTrainer_250epochs_save5__nnUNetPlans__2d")
    IN = os.path.join(ROOT, "mcnet_eval/test60_in")
    pids = sorted({os.path.basename(f).split("_0000")[0] for f in glob.glob(os.path.join(IN, "*_0000.nii.gz"))})
    assert len(pids) == 60, len(pids)

    predictor = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
                                perform_everything_on_device=True, device=torch.device("cuda"),
                                verbose=False, verbose_preprocessing=False, allow_tqdm=False)
    predictor.initialize_from_trained_model_folder(MODEL, use_folds=(0,), checkpoint_name="checkpoint_best.pth")

    def case_files(pid):
        return [os.path.join(IN, f"{pid}_000{i}.nii.gz") for i in range(4)]

    # Phase A: single-case latency, first 5 cases
    single_dir = os.path.join(OUT, "timing_single"); os.makedirs(single_dir, exist_ok=True)
    lat = []
    for pid in pids[:5]:
        torch.cuda.synchronize(); t0 = time.perf_counter()
        predictor.predict_from_files([case_files(pid)], [os.path.join(single_dir, pid + ".nii.gz")],
                                     save_probabilities=False, overwrite=True,
                                     num_processes_preprocessing=3, num_processes_segmentation_export=3)
        torch.cuda.synchronize()
        lat.append({"pid": pid, "seconds": round(time.perf_counter() - t0, 2)})

    # Phase B: batch all 60
    batch_dir = os.path.join(OUT, "timing_preds"); os.makedirs(batch_dir, exist_ok=True)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    predictor.predict_from_files([case_files(p) for p in pids],
                                 [os.path.join(batch_dir, p + ".nii.gz") for p in pids],
                                 save_probabilities=False, overwrite=True,
                                 num_processes_preprocessing=3, num_processes_segmentation_export=3)
    torch.cuda.synchronize()
    batch_total = time.perf_counter() - t0

    # Verify: DCF calls on timing_preds vs recorded dcf_test60.json
    with open(os.path.join(ROOT, "mcnet_eval/dcf_test60.json")) as f:
        ref = {r["pid"]: r["call"] for r in json.load(f)["per_case"]}
    agree, disagree, uncl = 0, [], []
    for pid in pids:
        arr = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(batch_dir, pid + ".nii.gz")))
        fg = arr > 0
        if fg.sum() == 0:
            uncl.append(pid); continue
        lab, n = ndimage.label(fg)
        sizes = ndimage.sum(np.ones_like(lab), lab, range(1, n + 1))
        votes = arr[lab == int(np.argmax(sizes)) + 1]
        call = 1 if (votes == 1).sum() > (votes == 2).sum() else 0
        if call == ref.get(pid):
            agree += 1
        else:
            disagree.append({"pid": pid, "new": call, "old": ref.get(pid)})

    out = {
      "single_case_latency_s": lat,
      "single_case_median_s": round(float(np.median([l["seconds"] for l in lat])), 2),
      "batch_60_total_s": round(batch_total, 1),
      "batch_per_case_s": round(batch_total / 60, 2),
      "call_agreement_vs_recorded": {"agree": agree, "unclassifiable": uncl, "disagree": disagree},
      "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 1e6, 1),
      "note": "Model B self-segments (no GT mask needed). Timing = full 4-seg sliding-window dual-class nnU-Net inference per patient, mirroring on, same trainer/fold/checkpoint as run_test_eval.sh.",
    }
    with open(os.path.join(OUT, "timing_modelB.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))
    print("MCNET-TIMING-DONE")


if __name__ == '__main__':
    main()
