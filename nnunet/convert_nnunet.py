#!/usr/bin/env python
'''Convert downloaded UCSF core-350 (TRAIN-232 pool ONLY) into nnU-Net raw format.
Enforces PROTOCOL.md: only patients marked train in core350_splitmap.txt are emitted.
Run AFTER dl_core.py completes (CORE-DOWNLOAD-COMPLETE).'''
import csv, json, shutil, sys
from pathlib import Path

ROOT = Path.home() / 'Repos/idh-mcnet'
RAW  = ROOT / 'nnUNet_raw/Dataset001_IDH'
(RAW/'imagesTr').mkdir(parents=True, exist_ok=True)
(RAW/'labelsTr').mkdir(parents=True, exist_ok=True)

split = {}
for line in open(ROOT/'core350_splitmap.txt'):
    pid, role = line.strip().split(',')
    split[pid] = role
train_ids = sorted(p for p, r in split.items() if r == 'train')
print(f'TRAIN pool: {len(train_ids)} patients')

MODS = {'T1':'0000','T1c':'0001','T2':'0002','FLAIR':'0003'}
missing, written = [], 0
for pid in train_ids:
    ok = True
    for mod, ch in MODS.items():
        src = ROOT/f'data/ucsf_core350/{pid}/{pid}_{mod}.nii.gz'
        dst = RAW/f'imagesTr/{pid}_{ch}.nii.gz'
        if src.exists():
            if not dst.exists(): shutil.copy(src, dst)
        else:
            missing.append(str(src)); ok = False
    msrc = ROOT/f'data/ucsf_core350/{pid}/{pid}_tumor_segmentation.nii.gz'
    if msrc.exists():
        dst = RAW/f'labelsTr/{pid}.nii.gz'
        if not dst.exists(): shutil.copy(msrc, dst)
    else:
        missing.append(str(msrc)); ok = False
    if ok: written += 1

json.dump({
  'channel_names': {'0':'T1','1':'T1c','2':'T2','3':'FLAIR'},
  'labels': {'background':0,'tumor':1},
  'numTraining': written,
  'file_ending': '.nii.gz',
  'name': 'IDH', 'description': 'UCSF-PDGM TRAIN-232 pool only (PROTOCOL.md); tumor seg for MC-net pretraining'
}, open(RAW/'dataset.json','w'), indent=2)
print(f'converted {written} patients; missing files: {len(missing)}')
for m in missing[:10]: print(' MISSING', m)
print('CONVERT-DONE')
