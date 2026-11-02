#!/usr/bin/env python3
"""Run PORT_SPEC.md: a one-epoch smoke test, then the faithful fold-0 retrain.

Run with ../.venv/bin/python -u train_baseline.py. The supplied notebook modules
are imported unchanged. Loader iteration is observed only for timing; model,
preprocessing, augmentation, optimization and early stopping remain theirs.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from urllib.parse import urlparse


HERE = Path(__file__).resolve().parent
# Keep all incidental library caches within the authorized working directory.
os.environ["TORCH_HOME"] = str(HERE / ".cache/torch")
os.environ["MPLCONFIGDIR"] = str(HERE / ".cache/matplotlib")
os.environ["XDG_CACHE_HOME"] = str(HERE / ".cache")
os.environ["MPLBACKEND"] = "Agg"

import pandas as pd
import torch
import torchvision
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

import idh_experiments as experiments
from build_manifest import HERE, PROJECT, RAW_DIR, RESULTS_DIR, SPLITS_JSON, build_manifest
from idh_core import Config, write_json


def source_hashes():
    return {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
            for name in ("PORT_SPEC.md", "idh_core.py", "idh_experiments.py", "cell_18.py",
                         "build_manifest.py", "train_baseline.py")}


def winner_config():
    return Config(
        data_dir=str(RAW_DIR), manifest_csv=str(RESULTS_DIR / "manifest.csv"),
        output_dir=str(RESULTS_DIR), cache_dir=str(HERE / ".cache/preprocessed"),
        experiment_mode="a2", backbone="efficientnet_b0", pretrained_backbone=True,
        lr=1e-4, weight_decay=1e-4, freeze_pretrained_epochs=2,
        input_resolution=112, volume_shape=(112, 112, 112), epochs=20,
        batch_size=2, accumulation_steps=2, mixed_precision=True,
        use_class_weight=True, adjacent_slices=3, locations=3, voxel_spacing=1.5,
        seed=20261033, early_patience=5, device="cuda",
    )


def check_runtime():
    if Path(sys.prefix).resolve() != (PROJECT / ".venv").resolve():
        raise RuntimeError("Use the existing ../.venv/bin/python interpreter")
    if torch.__version__ != "2.8.0+cu128":
        raise RuntimeError(f"Expected unchanged torch 2.8.0+cu128; found {torch.__version__}")
    if torchvision.__version__ != "0.23.0+cu128":
        raise RuntimeError(f"Expected matching torchvision 0.23.0+cu128; found {torchvision.__version__}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; run with GPU device access (outside the sandbox if needed)")
    assert callable(efficientnet_b0)
    if experiments.VARIANTS["a2"] != (False, True, False, False):
        raise RuntimeError("Notebook a2 flags differ from PORT_SPEC.md")
    device = torch.cuda.get_device_properties(0)
    runtime = {"python": sys.version, "executable": sys.executable, "torch": torch.__version__,
               "torchvision": torchvision.__version__, "gpu": device.name,
               "gpu_total_memory_bytes": device.total_memory, "torch_threads": torch.get_num_threads(),
               "imagenet_weights": str(EfficientNet_B0_Weights.DEFAULT),
               "source_sha256": source_hashes(),
               "split_sha256": hashlib.sha256(SPLITS_JSON.read_bytes()).hexdigest()}
    RESULTS_DIR.mkdir(exist_ok=True)
    write_json(runtime, RESULTS_DIR / "runtime.json")
    print(f"Runtime: torch={torch.__version__}, torchvision={torchvision.__version__}, "
          f"GPU={device.name}, VRAM={device.total_memory / 2**30:.3f} GiB", flush=True)
    return runtime


def prepare_imagenet_weights():
    """Reuse an existing DEFAULT weight file when present; cache weights locally."""
    filename = Path(urlparse(EfficientNet_B0_Weights.DEFAULT.url).path).name
    destination = Path(torch.hub.get_dir()) / "checkpoints" / filename
    if not destination.exists():
        old_cache = Path.home() / ".cache/torch/hub/checkpoints" / filename
        if old_cache.is_file():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(old_cache, destination)
            print(f"Copied cached ImageNet weights to {destination}", flush=True)
    # Only torchvision's DEFAULT ImageNet weights may be fetched, never MRI data.
    weights = EfficientNet_B0_Weights.DEFAULT.get_state_dict(progress=True, check_hash=True)
    del weights
    print(f"ImageNet DEFAULT weights ready: {destination}", flush=True)


class EpochObserver:
    """Observe original DataLoader iteration without changing batches or RNG."""
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.rows = []
        self.started = None

    def begin(self):
        torch.cuda.synchronize()
        self.started = time.perf_counter()

    def end(self):
        if self.started is None:  # Final best-checkpoint validation is not an epoch.
            return
        torch.cuda.synchronize()
        seconds = time.perf_counter() - self.started
        peak = torch.cuda.max_memory_allocated()
        row = {"epoch": len(self.rows) + 1, "epoch_seconds": seconds,
               "peak_vram_bytes": peak, "peak_vram_gib": peak / 2**30}
        self.rows.append(row)
        self.started = None
        pd.DataFrame(self.rows).to_csv(self.output_dir / "epoch_times.csv", index=False)
        print(f"TIMING epoch={row['epoch']} train+val={seconds:.2f}s "
              f"peak_VRAM={peak / 2**30:.3f} GiB ({peak} bytes)", flush=True)


class ObservedLoader:
    def __init__(self, loader, name, observer):
        self.loader, self.name, self.observer = loader, name, observer

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        if self.name == "train":
            self.observer.begin()
        yield from self.loader
        if self.name == "val":
            self.observer.end()

    def __getattr__(self, name):
        return getattr(self.loader, name)


def run_stage(parts, cfg, output_dir, stage):
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "best.pt").exists() or (output_dir / "history.csv").exists():
        raise FileExistsError(f"Refusing to overwrite a prior training run in {output_dir}")
    observer = EpochObserver(output_dir)
    original_loaders = experiments.loaders_from_splits

    def observed_loaders(splits, config, balanced=False):
        loaders = original_loaders(splits, config, balanced=balanced)
        return {name: ObservedLoader(loader, name, observer) for name, loader in loaders.items()}

    print(f"START {stage}: {len(parts['train'])} train / {len(parts['val'])} val; "
          f"seed={cfg.seed}, max_epochs={cfg.epochs}", flush=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    experiments.loaders_from_splits = observed_loaders
    try:
        model, destination = experiments.run_training(
            parts, cfg, experiment="a2", experiment_dir=output_dir, evaluate_test=False)
        torch.cuda.synchronize()
        wall_seconds = time.perf_counter() - started
        peak_bytes = torch.cuda.max_memory_allocated()
    finally:
        experiments.loaders_from_splits = original_loaders
    history = pd.read_csv(destination / "history.csv")
    predictions = pd.read_csv(destination / "val_predictions.csv", dtype={"patient_id": str})
    if (len(predictions) != 47 or predictions.patient_id.duplicated().any()
            or predictions.patient_id.tolist() != parts["val"].patient_id.tolist()):
        raise RuntimeError("Validation output does not match the exact 47-patient fold-0 order")
    if predictions.idh_label.tolist() != parts["val"].idh_label.tolist():
        raise RuntimeError("Validation labels changed")
    if not predictions.probability.between(0, 1).all():
        raise RuntimeError("Invalid validation probabilities")
    if len(history) != len(observer.rows):
        raise RuntimeError("Epoch timing/history mismatch")
    metrics = json.loads((destination / "val_metrics.json").read_text())
    epochs = int(history.epoch.iloc[-1])
    best_epoch = int(history.loc[history.is_best, "epoch"].iloc[-1])
    stopped = epochs - best_epoch >= cfg.early_patience
    summary = {
        "stage": stage, "status": "passed" if stage == "smoke" else "completed",
        "n_train": len(parts["train"]), "n_val": len(parts["val"]), "epochs_run": epochs,
        "max_epochs": cfg.epochs, "best_epoch": best_epoch,
        "early_stop_epoch": epochs if stopped else None,
        "val_auroc": metrics["auroc"], "val_balanced_accuracy": metrics["balanced_accuracy"],
        "threshold": cfg.threshold, "peak_vram_bytes": peak_bytes,
        "peak_vram_gib": peak_bytes / 2**30, "wall_time_seconds": wall_seconds,
        "epoch_times_seconds": [row["epoch_seconds"] for row in observer.rows],
        "source_sha256": source_hashes(),
    }
    write_json(summary, output_dir / "summary.json")
    print(f"{stage.upper()} SUMMARY: {json.dumps(summary, sort_keys=True)}", flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return summary


def smoke_parts(parts):
    # First five fold-ordered training patients, ensuring both loss classes occur.
    train = parts["train"]
    indices = list(range(5))
    if train.iloc[indices].idh_label.nunique() < 2:
        missing = 1 - int(train.iloc[0].idh_label)
        indices[-1] = int(train.index[train.idh_label == missing][0])
    return {"train": train.loc[sorted(indices)].reset_index(drop=True), "val": parts["val"].copy()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("all", "smoke", "full"), default="all")
    args = parser.parse_args()
    procedure_started = time.perf_counter()
    runtime = check_runtime()
    parts = build_manifest()
    cfg = winner_config()
    prepare_imagenet_weights()
    smoke_dir = RESULTS_DIR / "smoke"
    if args.stage in ("all", "smoke"):
        small_parts = smoke_parts(parts)
        smoke_dir.mkdir(exist_ok=True)
        write_json({name: frame.patient_id.tolist() for name, frame in small_parts.items()},
                   smoke_dir / "patient_ids.json")
        smoke = run_stage(small_parts, replace(cfg, epochs=1, output_dir=str(smoke_dir)), smoke_dir, "smoke")
        print(f"SMOKE PASSED: epoch time={smoke['epoch_times_seconds'][0]:.2f}s; "
              f"peak VRAM={smoke['peak_vram_gib']:.3f} GiB "
              f"(torch.cuda.max_memory_allocated={smoke['peak_vram_bytes']} bytes)", flush=True)
    if args.stage in ("all", "full"):
        if args.stage == "full":
            smoke = json.loads((smoke_dir / "summary.json").read_text())
            if (smoke.get("status") != "passed" or smoke.get("source_sha256") != source_hashes()
                    or smoke.get("n_train") != 5 or smoke.get("n_val") != 47
                    or smoke.get("epochs_run") != 1):
                raise RuntimeError("Run a passing one-epoch smoke test with these sources first")
        summary = run_stage(parts, cfg, RESULTS_DIR, "full")
        summary["procedure_wall_time_seconds"] = time.perf_counter() - procedure_started
        summary["gpu"] = runtime["gpu"]
        write_json(summary, RESULTS_DIR / "summary.json")
        stopping = (f"early stopping at epoch {summary['early_stop_epoch']}"
                    if summary["early_stop_epoch"] is not None else "20 epochs completed; no early stopping")
        print("\nFINAL SUMMARY", flush=True)
        print(f"Epochs run: {summary['epochs_run']}; best epoch: {summary['best_epoch']}; {stopping}", flush=True)
        print(f"Fold-0 val (n=47): AUROC={summary['val_auroc']:.6f}; "
              f"balanced accuracy={summary['val_balanced_accuracy']:.6f} (threshold=0.5)", flush=True)
        print(f"Peak VRAM: {summary['peak_vram_gib']:.3f} GiB ({summary['peak_vram_bytes']} bytes)", flush=True)
        print(f"Full-run wall time: {summary['wall_time_seconds']:.2f}s; "
              f"procedure wall time: {summary['procedure_wall_time_seconds']:.2f}s", flush=True)
        print(f"Results: {RESULTS_DIR}", flush=True)


if __name__ == "__main__":
    main()
