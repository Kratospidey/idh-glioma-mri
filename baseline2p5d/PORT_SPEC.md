# D6 Port Spec: faithful retrain of old winning baseline (2.5D EfficientNet-B0) on nnU-Net fold-0 split

GOAL: paired same-cohort development comparison - retrain the OLD winning model
(efficientnet_b0_112_lr1e4_aug, variant a2=augmentation-only) on the EXACT 185 fold-0 train patients
and evaluate on the same 47 fold-0 val patients as the MC-net, using the notebook's own code.

SOURCE (faithful, already in this dir): idh_core.py, idh_experiments.py (from his Colab notebook).
cell_18.py shows the winner config. Do not rewrite the pipeline - adapt paths/splits only.

WINNER CONFIG (reproduce exactly): backbone=efficientnet_b0, pretrained_backbone=True (ImageNet,
torchvision DEFAULT), lr=1e-4, weight_decay=1e-4, freeze_pretrained_epochs=2, input_resolution=112,
volume_shape=(112,112,112), variant flags a2=(balance F, augment T, consistency F, adversarial F),
epochs=20 max, batch_size=2, accumulation_steps=2, mixed_precision=True, use_class_weight=True,
adjacent_slices=3, locations=3, voxel_spacing=1.5, seed=20261029+4=20261033.
Early stopping: the notebook's OWN mechanism (rank on val AUROC, tie-break macro-F1 then -loss,
early_patience=5) - pre-declared here; REPORT at which epoch it triggers (or that 20 completed).

DATA (local, NO downloads): raw volumes at ~/Repos/idh-mcnet/data/ucsf_core350/<PID>/<PID>_{T1,T1c,T2,FLAIR}.nii.gz
and mask <PID>_tumor_segmentation.nii.gz (RAW unmodified masks, BraTS labels - use mask>0 as whole tumor,
matching how his pipeline used masks for slice centering only). Labels from
~/Repos/idh-mcnet/ucsf493_labels_pools.csv (pool=="train" only, idh_mutant 0/1).
SPLIT (fixed, not his splitter): 185 train / 47 val patient IDs from
~/Repos/idh-mcnet/nnUNet_preprocessed/Dataset002_IDHMC/splits_final.json fold 0. Save both ID lists
into the results dir. NEVER touch pool val/test/ext143.

ADAPTATIONS ALLOWED (minimal): manifest CSV built locally with the columns load_manifest expects
(check idh_core.load_manifest + PatientDataset for exact path/seg columns); Config paths pointed at
local dirs; colab drive.mount removed; device cuda. Keep model, preprocessing (cached_preprocess,
volume_to_25d slice selection), augmentation, training loop, early stopping UNCHANGED.

ENV: ~/Repos/idh-mcnet/.venv (torch 2.8.0+cu128 DO NOT upgrade/replace torch). pip install only
missing light deps (nibabel, pandas, scikit-learn, tqdm) pinned to notebook ranges; torchvision must
match installed torch (should already be there - verify efficientnet_b0 imports).

PROCEDURE:
1. Write manifest builder + run script train_baseline.py in ~/Repos/idh-mcnet/baseline2p5d/.
2. SMOKE TEST: 1 epoch on 5 train patients, print epoch time + peak VRAM (torch.cuda.max_memory_allocated).
3. If smoke passes, FULL RUN on all 185 train / 47 val, output to ~/Repos/idh-mcnet/baseline2p5d/results/.
4. Save val_predictions.csv (patient_id, idh_label, probability), history.csv, best.pt, settings.
5. Print a summary: epochs run, early-stop epoch if triggered, val AUROC/bal_acc, peak VRAM, wall time.
Do NOT evaluate on anything but the 47 val. Do NOT modify files outside baseline2p5d/ and the venv.
