# Predicting Glioma IDH Mutation Status from Multisequence MRI

Can an MRI scan tell whether a diffuse glioma carries an **IDH mutation** - a molecular
marker that normally requires tissue testing? This repo holds the code behind two
working models that predict IDH status from routine preoperative MRI (T1, T1ce, T2, FLAIR):

- **Model A** - a mask-guided 2.5D classifier (EfficientNet-B0, 4.18M params) that needs a
  tumor mask at inference.
- **Model B** - a self-segmenting multicontrast network ("MC-net", nnU-Net v2) that finds
  the tumor itself and needs no mask.

![Pipeline: both models](figures/fig_arch.png)

## Class labels

- **Patient level:** `1` = IDH-mutant, `0` = IDH-wildtype.
- **Model B voxel level:** class `1` = IDH-mutant tumor voxels, class `2` = IDH-wildtype
  tumor voxels (the patient's IDH label is pushed onto all of its tumor voxels during
  training). At inference the voxel classes are fused, the largest 3D connected component
  is kept as the whole tumor, and a **majority vote over its voxel classes** becomes the
  patient-level call. The mutant voxel fraction doubles as a probability-like score.

## How each model is trained

| Technique | Model A | Model B | Code |
| --- | --- | --- | --- |
| Pretraining | ImageNet-pretrained EfficientNet-B0 (torchvision) | None - trained from scratch | `baseline2p5d/idh_core.py` |
| Fine-tuning | Backbone frozen for 2 warm-up epochs, then unfrozen | n/a | `baseline2p5d/idh_experiments.py` (`run_training`) |
| Architecture config | Manual comparison of 3 configs; winner: 112px input, lr 1e-4, augmentation-only | Fully automatic nnU-Net v2 configuration (no hyperparameter search) | `baseline2p5d/cell_18.py`, `baseline2p5d/PORT_SPEC.md` |
| Optimization | AdamW, cosine LR schedule, weight decay 1e-4, class-weighted BCE, mixed precision, grad accumulation | nnU-Net default (SGD + poly LR), Dice + CE voxel loss | `baseline2p5d/idh_experiments.py`, nnU-Net v2 |
| Model selection | Early stopping ranked on validation AUROC (patience 5) + validation-only threshold search (operating point 0.81) | Fixed epoch budget, best checkpoint by EMA pseudo-dice (epoch 121) | `baseline2p5d/train_baseline.py`, `nnunet/nnUNetTrainer_Xepochs_save5.py` |
| Input representation | 2.5D: 3 planes x 3 sites x 3 slices = 9 views x 12 channels from a mask-centered 112^3 crop, 1.5 mm spacing | Full-volume 2D nnU-Net over all 4 sequences | `baseline2p5d/idh_core.py`, `nnunet/convert_nnunet.py` |

## Code behind each technique

Every snippet below is copied verbatim from the file named above it (line numbers refer to the files in this repo). Where Model B relies on the nnU-Net v2 library, the snippet is library code, labelled as such, and not part of this repo's own source. Where nothing exists in code, the section says so.

### Model A: 2.5D EfficientNet-B0 (pretrained)

**Pretraining** - `baseline2p5d/idh_core.py` L357-367: load torchvision ImageNet weights, adapt the first conv from 3 to 12 input channels (4 sequences x 3 adjacent slices) by averaging the RGB filters, then add a 1x1 projection that feeds the attention-over-views head.

```python
        elif cfg.backbone == 'efficientnet_b0':
            from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
            backbone=efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT if cfg.pretrained_backbone else None)
            old=backbone.features[0][0]
            new=nn.Conv2d(inp,old.out_channels,kernel_size=old.kernel_size,
                          stride=old.stride,padding=old.padding,bias=False)
            if cfg.pretrained_backbone:
                with torch.no_grad():
                    new.weight.copy_(old.weight.mean(dim=1,keepdim=True).repeat(1,inp,1,1)*(3.0/inp))
            backbone.features[0][0]=new
            self.encoder=nn.Sequential(backbone.features,nn.Conv2d(1280,h,1),nn.SiLU())
```

**Fine-tuning** - `baseline2p5d/idh_experiments.py` L108-113 (freeze for the first 2 epochs, then unfreeze) and L78-80 (keep the pretrained BatchNorm statistics fixed during warm-up).

```python
        if cfg.backbone in ('resnet18','efficientnet_b0') and cfg.pretrained_backbone and cfg.freeze_pretrained_epochs:
            train_encoder=(epoch>cfg.freeze_pretrained_epochs)
            backbone_module=model.encoder[:8] if cfg.backbone=='resnet18' else model.encoder[0]
            for p in backbone_module.parameters():p.requires_grad_(train_encoder)
            if epoch==1:print('Pretrained backbone warm-up: frozen for',cfg.freeze_pretrained_epochs,'epochs')
            if epoch==cfg.freeze_pretrained_epochs+1:print('Unfreezing pretrained ResNet backbone for fine-tuning')
    ...
        if training and cfg.pretrained_backbone and epoch<=cfg.freeze_pretrained_epochs:
            if cfg.backbone=='resnet18': model.encoder[:8].eval()
            elif cfg.backbone=='efficientnet_b0': model.encoder[0].eval()  # keep pretrained BatchNorm statistics fixed during warm-up
```

**Architecture configuration** - `baseline2p5d/train_baseline.py` L46-56 is the winning configuration as run; `baseline2p5d/cell_18.py` L1-18 defines the candidate trials that were compared (the `RUN_*` flags switch which candidates are trained).

```python
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
```

```python
from idh_core import seed_everything
TRIALS=[]
if RUN_SCRATCH:
    TRIALS.append(('cnn_from_scratch_112_lr3e4','baseline',dict(
        backbone='light_cnn',pretrained_backbone=False,lr=3e-4,weight_decay=1e-4,
        freeze_pretrained_epochs=0,input_resolution=112,volume_shape=(112,112,112))))
if RUN_RESNET_112:
    TRIALS.append(('resnet18_112_lr1e4_aug','a2',dict(
        backbone='resnet18',pretrained_backbone=True,lr=1e-4,weight_decay=1e-4,
        freeze_pretrained_epochs=2,input_resolution=112,volume_shape=(112,112,112))))
if RUN_RESNET_96:
    TRIALS.append(('resnet18_96_lr3e5_aug','a2',dict(
        backbone='resnet18',pretrained_backbone=True,lr=3e-5,weight_decay=5e-5,
        freeze_pretrained_epochs=2,input_resolution=96,volume_shape=(96,96,96))))
if RUN_EFFICIENTNET:
    TRIALS.append(('efficientnet_b0_112_lr1e4_aug','a2',dict(
        backbone='efficientnet_b0',pretrained_backbone=True,lr=1e-4,weight_decay=1e-4,
        freeze_pretrained_epochs=2,input_resolution=112,volume_shape=(112,112,112))))
```

**Optimization** - `baseline2p5d/idh_experiments.py` L53-57 (AdamW, cosine schedule, mixed-precision scaler, class-weighted BCE), then the loss and the gradient-accumulation step (L87-89, L98-102).

```python
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg.lr,weight_decay=cfg.weight_decay)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=max(1,cfg.epochs))
    scaler=torch.amp.GradScaler('cuda',enabled=cfg.device.startswith('cuda') and cfg.mixed_precision)
    counts=splits['train'].idh_label.value_counts();weight=(counts.get(0,0)/max(1,counts.get(1,0))) if cfg.use_class_weight else 1.
    pos_weight=torch.tensor(weight,dtype=torch.float32,device=cfg.device)
```

```python
                with autocast_ctx(cfg):
                    logit,z,domains=model(x,grl_lambda=1.0 if training and adv else 0.,return_embedding=True)
                    loss=F.binary_cross_entropy_with_logits(logit,y,pos_weight=pos_weight)
        ...
                if training:
                    scaler.scale(loss/max(1,cfg.accumulation_steps)).backward()
                    if (i+1)%cfg.accumulation_steps==0 or i+1==len(loader):
                        scaler.unscale_(optimizer);nn.utils.clip_grad_norm_(model.parameters(),5.)
                        scaler.step(optimizer);scaler.update();optimizer.zero_grad(set_to_none=True)
```

**Model selection** - `baseline2p5d/idh_experiments.py` L119-127 (rank epochs by validation AUROC, then macro-F1, then loss; stop after 5 epochs without improvement). The operating-point search that gave the 0.81 threshold lives in the Colab notebook (`notebooks/IDH_Clean_Restart_350_Colab.ipynb`, cell 20) and uses validation data only.

```python
        rank=(round(float(score),4),round(float(va['macro_f1']),4),-float(va['loss']))
        improved=rank>best_rank
        if improved:best=score;best_rank=rank;best_ep=epoch
        history.append({'epoch':epoch,**{'train_'+k:v for k,v in tr.items()},**{'val_'+k:v for k,v in va.items()},'is_best':improved})
        save_checkpoint(last,model,cfg,epoch,history,optimizer,scheduler,scaler,best,len(domain_enc.classes_) if domain_enc is not None else 0)
        if improved:save_checkpoint(out/'best.pt',model,cfg,epoch,history,best_score=best,n_domains=len(domain_enc.classes_) if domain_enc is not None else 0)
        pd.DataFrame(history).to_csv(out/'history.csv',index=False)
        print(f'{experiment} {epoch:02d}: train_loss={tr["loss"]:.4f} val_loss={va["loss"]:.4f} val_AUROC={va["auroc"]} best_epoch={best_ep}')
        if epoch-best_ep>=cfg.early_patience:print('Early stopping');break
```

```python
def select_threshold(y,p):
    scores=[]
    for t in np.round(np.arange(.15,.851,.01),3):
        yp=(np.asarray(p)>=t).astype(int)
        scores.append((balanced_accuracy_score(y,yp),f1_score(y,yp,average='macro',zero_division=0),-abs(t-.5),float(t)))
    return max(scores)[-1]
```

**Input representation** - `baseline2p5d/idh_core.py` L264-288: three planes x three positions x three adjacent slices per sequence, centred on the tumor mask, resized to the crop resolution.

```python
def volume_to_25d(vol, cfg, seg=None):
    """[M,X,Y,Z] -> [3*locations,M*adjacent,H,W]; centre chosen by tumour mask if available.
    Strict channel order: for each plane, modalities, then adjacent slices.
    """
    x=torch.as_tensor(vol,dtype=torch.float32)
    if x.ndim!=4 or x.shape[0]!=len(cfg.sequences): raise ValueError('Expected [modalities,X,Y,Z]')
    if cfg.adjacent_slices%2!=1 or cfg.adjacent_slices<1: raise ValueError('adjacent_slices must be odd')
    selected=[]
    for axis in (0,1,2):
        if seg is not None and seg.sum()>0:
            idx=np.where(seg)[axis]; lo=int(np.quantile(idx,0.20)); hi=int(np.quantile(idx,0.80))
        else:
            lo=int(x.shape[axis+1]*.30); hi=int(x.shape[axis+1]*.70)
        centres=np.rint(np.linspace(lo,hi,cfg.locations)).astype(int)
        for centre in centres:
            slices=[]
            for m in range(len(cfg.sequences)):
                for offset in range(-(cfg.adjacent_slices//2),cfg.adjacent_slices//2+1):
                    cut=int(np.clip(centre+offset,0,x.shape[axis+1]-1))
                    plane=x[m].select(axis,cut)
                    slices.append(plane)
            view=torch.stack(slices)[:,None,:,:]
            view=torch.nn.functional.interpolate(view,size=(cfg.input_resolution,cfg.input_resolution),mode='bilinear',align_corners=False)[:,0]
            selected.append(view)
    return torch.stack(selected)
```

### Model B: nnU-Net 2D, dual-class, from scratch

**Pretraining** - none. No pretrained weights are loaded: the network starts from nnU-Net's random initialization and is trained only on the training split. There is no pretraining code to show. The only project-side change to training is the bounded-epoch trainer, `nnunet/nnUNetTrainer_Xepochs_save5.py` (whole file):

```python
from nnunetv2.training.nnUNetTrainer.variants.training_length.nnUNetTrainer_Xepochs import nnUNetTrainer_250epochs


class nnUNetTrainer_250epochs_save5(nnUNetTrainer_250epochs):
    def __init__(self, plans, configuration, fold, dataset_json, device):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.save_every = 5
```

**Fine-tuning** - not applicable. There is no second stage, freezing or unfreezing in Model B; one training run fits the whole network.

**Architecture configuration** - fully automatic, with no hyperparameter search. nnU-Net derives the 2D network from the dataset fingerprint (`nnUNetPlans`): 192 x 160 patches, batch size 106, 1 mm in-plane spacing, per-sequence z-score normalization over 4 channels. The class labels come from `nnunet/dataset002_IDHMC_dataset.json` (copy of the Dataset002 `dataset.json`; background 0, IDH_mutant 1, IDH_wildtype 2). `nnunet/convert_nnunet.py` shows how images and masks are written in nnU-Net raw format and which patients are allowed in; it is the single-class tumor variant (`Dataset001_IDH`). The step that stamps the 1/2 patient label onto each tumor mask for Dataset002 was run interactively and is not saved as a script, so it is not included here.

```python
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
```

**Optimization** - nnU-Net v2 defaults (library code, nnunetv2 2.8.1, Apache-2.0, `training/nnUNetTrainer/nnUNetTrainer.py`; `nnUNetTrainer_250epochs` only changes the epoch count to 250): SGD with Nesterov momentum 0.99 and a poly learning-rate decay, with Dice + cross-entropy voxel loss.

```python
        self.initial_lr = 1e-2
        self.weight_decay = 3e-5
        self.oversample_foreground_percent = 0.33
    ...
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
                                    momentum=0.99, nesterov=True)
        lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)
        return optimizer, lr_scheduler
    ...
    def _build_loss(self):
        if self.label_manager.has_regions:
            loss = DC_and_BCE_loss({},
                                   {'batch_dice': self.configuration_manager.batch_dice,
                                    'do_bg': True, 'smooth': 1e-5, 'ddp': self.is_ddp},
                                   use_ignore_label=self.label_manager.ignore_label is not None,
                                   dice_class=MemoryEfficientSoftDiceLoss)
        else:
            loss = DC_and_CE_loss({'batch_dice': self.configuration_manager.batch_dice,
                                   'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp}, {}, weight_ce=1, weight_dice=1,
                                  ignore_label=self.label_manager.ignore_label, dice_class=MemoryEfficientSoftDiceLoss)
```

**Model selection** - nnU-Net library code, same file: `checkpoint_best.pth` is saved whenever the EMA pseudo-Dice improves. The reported checkpoint is epoch 121 of the `nnUNetTrainer_250epochs_save5` run, loaded with `-chk checkpoint_best.pth` in `mcnet_eval/run_test_eval.sh`. There is no early stopping, only a fixed epoch budget.

```python

        # handle 'best' checkpointing. ema_fg_dice is computed by the logger and can be accessed like this
        if self._best_ema is None or self.logger.get_value('ema_fg_dice', step=-1) > self._best_ema:
            self._best_ema = self.logger.get_value('ema_fg_dice', step=-1)
            self.print_to_log_file(f"Yayy! New best EMA pseudo Dice: {np.round(self._best_ema, decimals=4)}")
            self.save_checkpoint(join(self.output_folder, 'checkpoint_best.pth'))
```

**Input representation and patient-level call** - Model B reads all 4 sequences as channels of a full-volume 2D nnU-Net (`nnunet/convert_nnunet.py`). `mcnet_eval/dcf_eval.py` L30-45 turns the predicted voxel classes into a patient call: keep the largest connected tumor component and take a majority vote over its mutant (1) and wildtype (2) voxels.

```python
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
```

## Code map

```
baseline2p5d/            Model A
  idh_core.py              dataset, 2.5D sampling, MRI25DNet (attention over 9 views)
  idh_experiments.py       training runner: freeze/unfreeze, cosine LR, early stopping
  train_baseline.py        local retrain driver on the fixed fold-0 split
  build_manifest.py        manifest builder for local retraining
  cell_18.py               the winning configuration, as run
  PORT_SPEC.md             spec for the faithful same-split retrain
  results/                 aggregate metrics, training history, runtimes
nnunet/
  convert_nnunet.py        writes nnU-Net raw images + whole-tumor masks (single-class Dataset001 variant)
  dataset002_IDHMC_dataset.json  Model B label map: 0 background, 1 IDH_mutant, 2 IDH_wildtype
  nnUNetTrainer_Xepochs_save5.py  trainer variant: bounded epochs, checkpoint every 5
mcnet_eval/              Model B evaluation
  dcf_eval.py              dual-class fusion + majority-vote patient call
  dcf_eval_pool.py         pooled evaluation driver
  run_test_eval.sh         held-out test evaluation entry point
  dcf_*.json               aggregate results (dev / test / external pools)
error_analysis_20261101/
  error_audit.py           per-case error audit + Dice vs reference masks
  mcnet_timing.py          inference-cost measurement (batch and single-case)
  baseline_test60.py       Model A held-out test evaluation
  overlays.py              error overlay rendering
  *.json                   aggregate audit + timing results
notebooks/
  IDH_Clean_Restart_350_Colab.ipynb  the original Model A pipeline, self-contained (Colab)
figures/                   paper figures (fig_arch via make_arch.py)
```

## Outputs and how to read them

- **Model A** outputs a probability `P(IDH-mutant)`; the validation-fitted operating point
  is **0.81**. Aggregate metrics live in `baseline2p5d/results/`.
- **Model B** outputs a dual-class voxel map plus the patient call (majority vote) and the
  mutant voxel fraction. Aggregate metrics live in `mcnet_eval/`; segmentation quality
  (Dice against reference masks) and inference timing in `error_analysis_20261101/`.
- No patient-level rows, raw scans, or model weights are published here - only code,
  reproduction schemas and aggregate outputs.

## Results (internal evaluation, UCSF-PDGM cohort)

| Evaluation | Model A | Model B |
| --- | --- | --- |
| Paired dev, 47 patients (identical split, both models) | 85.1% acc, AUROC 0.901 | 87.2% acc, AUROC 0.896 |
| Held-out test, 60 patients | 85.0% acc, AUROC 0.895 | **90.0% acc, AUROC 0.932** (sens 88.2%, spec 90.7%) |
| Unseen wildtype-only pool, 143 patients | - | 93.7% specificity (134/143) |
| Whole-tumor segmentation (test-60 audit) | - | median Dice 0.942 |
| Inference cost | ~1.0 s/case (dominated by preprocessing; needs a supplied tumor mask) | ~3.2 s single case, ~0.99 s/case in batch (no mask needed) |

Read these with the caveats:

- The paired dev comparison agrees on 44/47 patients (McNemar exact p = 1.0): with only 3
  discordant pairs the test has no power, so this is a **failure to detect a difference**,
  not proof of equivalence.
- The 60-patient test was run under unmatched protocols (Model A from its original Colab
  run with its own validation-fitted threshold; Model B from a fixed checkpoint with no
  retuning), and the split carries historical exposure from earlier untracked experiments.
  It is internal evidence, **not external validation**.
- Model B's practical advantage is removing the mask dependency - not a proven accuracy or
  speed superiority. Model A is faster per case but useless without a tumor mask; Model B
  runs end-to-end on raw sequences.
- The 143-case external pool is wildtype-only, so it measures specificity alone; mutant
  sensitivity on never-seen data remains unmeasured.

## Recordings

Terminal recordings of the project code running on the project laptop (NVIDIA RTX 5070 Laptop GPU, 8 GB). Three are live runs, captured as pixels of the terminal window while the output streamed (the LIVE timer is on screen). Patient IDs and local paths are masked before they are displayed. The matching text logs sit next to each video. Outputs of these runs were written to a scratch folder, so the original result files were not touched.

| Recording | What it shows | Length |
| --- | --- | --- |
| [Model A training](recordings/model_a_training_live.mp4) (live) | Retrain of the 2.5D EfficientNet-B0 on the fixed 185/47 split: 11 epochs, best epoch 6, validation AUROC 0.901, 123 s of training | 3:05 |
| [Model A test metrics](recordings/model_a_test_metrics_live.mp4) (live) | Saved `best.pt` scored on the 60 held-out test patients: accuracy 0.85, sensitivity 0.706, specificity 0.907, AUROC 0.910 | 1:20 |
| [Model B test metrics](recordings/model_b_test_metrics_live.mp4) (live) | Saved nnU-Net checkpoint (epoch 121) predicting the 60 test patients, then the fusion and majority-vote metrics: accuracy 0.90, sensitivity 0.882, specificity 0.907, AUROC 0.932 | 1:26 |
| [Model B training](recordings/model_b_training_historical_log.mp4) (**historical, not live**) | Excerpts of the real training logs and the `progress.png` curves from the 2026-10-30 run. A full Model B training run takes about 7 hours (about 101 s per epoch), so it was not re-run for the video | 0:19 |

## Data

All patients come from the public **UCSF-PDGM** cohort (CC-BY; please attribute in any
reuse): 350 core cases split patient-disjoint into TRAIN-232 (67 mutant / 165 wildtype),
VAL-58 (selection/threshold only) and TEST-60 (held out), plus 143 additional
wildtype-only cases kept locked as a specificity probe. Raw masks carry BraTS-style labels
{1, 2, 4}; Model B binarizes them (any voxel > 0) and relabels the whole tumor by the
patient's IDH status for dual-class supervision. Patient IDs, manifests with per-patient
rows, raw data, preprocessed arrays and trained weights are intentionally excluded from
this repo.

## Findings

1. A small ImageNet transfer model with staged fine-tuning already reaches ~85-90%
   patient-level accuracy on this cohort - cheap to train (about two minutes on an
   RTX 5070 laptop GPU, 0.31 GB VRAM).
2. The self-segmenting variant matches it on the identical split while dropping the mask
   requirement, and stays specific (93.7%) on 143 never-seen wildtype patients.
3. Segmentation quality is not the failure mode: misclassified cases still had good tumor
   overlap (median Dice 0.92 among errors), so errors come from the IDH signal, not from
   finding the tumor.

## Future work

- **External validation with mutants:** curated TCGA-LGG external cohort (public, via IDC)
  to measure mutant sensitivity on never-seen patients; EGD dataset access requested.
- **Declared early stopping** for every future run (monitored metric, patience 25,
  min delta 0.005) - adopted after a fixed-budget run kept training ~126 epochs past its
  best checkpoint.
- Sub-region (BraTS-style) segmentation instead of whole-tumor binarization.
- Matched-protocol re-evaluation of both models on a fully untouched split.

## Companion paper

IEEE-format paper (Deep Learning course project, NMIMS):
*Patient-Level Prediction of Glioma IDH Mutation Status from Multisequence MRI:
A Mask-Guided 2.5D Classifier Versus a Self-Segmenting Multicontrast Network* -
Param Singh Makwana, Shrijesh Yadav, Deven Yewale, Masooda Modak.

Research code, not a medical device. Not for diagnosis.
