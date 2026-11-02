"""Build the notebook manifest for the fixed nnU-Net fold-0 development cohort.

Only pool=train rows and the specified 185/47 patient IDs can reach a dataset.
No image data or source labels/splits are modified.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd

from idh_core import load_manifest, save_splits, write_json


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
RAW_DIR = PROJECT / "data/ucsf_core350"
LABELS_CSV = PROJECT / "ucsf493_labels_pools.csv"
SPLITS_JSON = PROJECT / "nnUNet_preprocessed/Dataset002_IDHMC/splits_final.json"
RESULTS_DIR = HERE / "results"
MODALITIES = {"t1": "T1", "t1ce": "T1c", "t2": "T2", "flair": "FLAIR"}


def build_manifest(output_dir: Path = RESULTS_DIR):
    """Return notebook-compatible train/val frames in the original fold order."""
    output_dir = Path(output_dir).resolve()
    if not output_dir.is_relative_to(HERE):
        raise ValueError("Results must stay inside baseline2p5d/")
    fold = json.loads(SPLITS_JSON.read_text())[0]
    if set(fold) != {"train", "val"}:
        raise ValueError("Expected only train and val in fold 0")
    for name, expected in (("train", 185), ("val", 47)):
        ids = fold[name]
        if (len(ids) != expected or len(set(ids)) != expected
                or not all(isinstance(pid, str) and pid.strip() == pid and pid for pid in ids)):
            raise ValueError(f"Invalid fold-0 {name} IDs; expected {expected} unique patients")
    if set(fold["train"]) & set(fold["val"]):
        raise ValueError("Train/val patient overlap")

    # Discard every non-train row before examining its labels or image paths.
    with LABELS_CSV.open(newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"patient_id", "idh_mutant", "pool"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Unexpected labels CSV columns")
        training_pool = [row for row in reader if row["pool"] == "train"]
    labels = {}
    for row in training_pool:
        pid = row["patient_id"]
        if pid in labels or row["idh_mutant"] not in {"0", "1"}:
            raise ValueError(f"Duplicate patient or invalid binary train label: {pid}")
        labels[pid] = int(row["idh_mutant"])
    ordered_ids = fold["train"] + fold["val"]
    if set(ordered_ids) != set(labels):
        raise ValueError("Fold-0 union must exactly equal the 232-patient training pool")

    records = []
    for pid in ordered_ids:
        record = {"patient_id": pid, "idh_label": labels[pid]}
        for modality, suffix in MODALITIES.items():
            record[f"{modality}_path"] = str(RAW_DIR / pid / f"{pid}_{suffix}.nii.gz")
        record["seg_path"] = str(RAW_DIR / pid / f"{pid}_tumor_segmentation.nii.gz")
        # Validate before writing the manifest, including the mandatory raw mask.
        for column, value in record.items():
            if column.endswith("_path") and not Path(value).is_file():
                raise FileNotFoundError(f"{pid}: missing {column}: {value}")
        records.append(record)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.csv"
    pd.DataFrame(records).to_csv(manifest_path, index=False)
    manifest = load_manifest(manifest_path, str(RAW_DIR))
    indexed = manifest.set_index("patient_id", drop=False)
    parts = {name: indexed.loc[fold[name]].reset_index(drop=True) for name in ("train", "val")}
    if any(frame.idh_label.nunique() != 2 for frame in parts.values()):
        raise ValueError("Both classes are required in train and val")
    save_splits(parts, output_dir / "splits")
    write_json(fold, output_dir / "fold0_ids.json")
    for name, ids in fold.items():
        (output_dir / f"{name}_ids.txt").write_text("\n".join(ids) + "\n")
    counts = {name: {str(label): int(count) for label, count in frame.idh_label.value_counts().items()}
              for name, frame in parts.items()}
    write_json({"fold": 0, "source_split": str(SPLITS_JSON), "source_labels": str(LABELS_CSV),
                "source_images": str(RAW_DIR), "allowed_pool": "train", "class_counts": counts,
                "mask": "raw BraTS labels; notebook uses mask > 0 for tumour centering/slices"},
               output_dir / "manifest_audit.json")
    print(f"Manifest verified: train={len(parts['train'])}, val={len(parts['val'])}; "
          f"class counts (0/1)={counts}", flush=True)
    return parts


if __name__ == "__main__":
    build_manifest()
