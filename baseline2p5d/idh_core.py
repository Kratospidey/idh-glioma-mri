"""Patient-level IDH classification core for research, not for clinical use.
Input: a VERIFIED patient manifest. Class 1 = IDH-mutant, 0 = IDH-wildtype.
"""
from __future__ import annotations
import os, json, random, hashlib, warnings, math
from pathlib import Path
from dataclasses import dataclass, asdict, replace
from typing import Optional
import numpy as np
import pandas as pd
from scipy import ndimage
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score, confusion_matrix, brier_score_loss,
    roc_curve, precision_recall_curve)

SEQUENCES = ('t1','t1ce','t2','flair')
PATH_FIELDS = tuple(f'{m}_path' for m in SEQUENCES)
LABEL_MAP = {'idh-mutant':1, 'mutant':1, 'mutated':1, 'idh-mutated':1, '1':1,
             'idh-wildtype':0, 'idh-wild-type':0, 'wildtype':0, 'wild-type':0, '0':0}

@dataclass
class Config:
    data_dir: str = '/content/drive/MyDrive/IDH_Audit/dataset'
    manifest_csv: str = '/content/drive/MyDrive/IDH_Audit/manifest.csv'
    output_dir: str = '/content/drive/MyDrive/IDH_Audit/results'
    cache_dir: str = '/content/idh_preprocessed' # Colab local disk is much faster
    experiment_mode: str = 'baseline'       # baseline, a1, a2, a3, a4, combined, all
    domain_field: str = 'vendor'           # use site if vendor not available
    sequences: tuple = SEQUENCES
    voxel_spacing: float = 1.5
    volume_shape: tuple = (96,96,96)
    input_resolution: int = 96
    adjacent_slices: int = 3
    locations: int = 3
    batch_size: int = 2
    epochs: int = 20
    lr: float = 3e-4
    weight_decay: float = 1e-4
    seed: int = 123
    test_fraction: float = 0.20
    val_fraction: float = 0.20 # fraction of ALL patients
    mixed_precision: bool = True
    accumulation_steps: int = 2
    workers: int = 2
    early_patience: int = 5
    dropout: float = 0.2
    hidden_dim: int = 128
    backbone: str = 'light_cnn'     # light_cnn or resnet18
    pretrained_backbone: bool = False # ImageNet weights ONLY when opted in
    freeze_pretrained_epochs: int = 0 # freeze ResNet ImageNet backbone before fine-tuning
    use_class_weight: bool = True
    use_n4_bias_correction: bool = False  # For raw MRI only; often redundant for UCSF-PDGM
    assume_coregistered: bool = True      # DO NOT use this with unaligned raw scans
    augment_strength: float = 0.4
    consistency_lambda: float = 0.5
    consistency_warmup_epochs: int = 3
    adversarial_lambda: float = 0.1
    threshold: float = 0.5
    run_explanations: bool = False
    run_lodo: bool = False
    external_manifest_csv: str = ''
    resume: bool = False
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def write_json(obj, path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    def encode(x):
        if isinstance(x, (np.integer, np.floating)): return x.item()
        if isinstance(x,np.ndarray): return x.tolist()
        if isinstance(x,Path): return str(x)
        if isinstance(x,float) and not np.isfinite(x): return None
        raise TypeError(f'Not JSON serialisable: {type(x)}')
    with open(path,'w') as f: json.dump(obj,f,indent=2,default=encode,allow_nan=False)

def load_manifest(csv_path, data_dir='', require_labels=True, sequences=SEQUENCES):
    p=Path(csv_path)
    if not p.is_file(): raise FileNotFoundError(f'Missing patient manifest: {p}. See DATASET_GUIDE.md')
    df=pd.read_csv(p, dtype={'patient_id':str}).fillna('')
    required=['patient_id']+(["idh_label"] if require_labels else [])+[f'{s}_path' for s in sequences]
    missing=set(required)-set(df.columns)
    if missing: raise ValueError(f'Manifest is missing required columns: {sorted(missing)}')
    if df.empty: raise ValueError('Manifest is empty')
    df['patient_id']=df.patient_id.astype(str).str.strip()
    if (df.patient_id=='').any(): raise ValueError('Empty patient_id found')
    if df.patient_id.duplicated().any():
        raise ValueError('Duplicate patient_id rows, including follow-up visits. Resolve to one independent patient first: '+
                         ', '.join(df.loc[df.patient_id.duplicated(),'patient_id'].head().tolist()))
    if require_labels:
        labels=df.idh_label.astype(str).str.strip().str.lower()
        bad=sorted(set(labels)-set(LABEL_MAP))
        if bad: raise ValueError('Unsupported/uncertain IDH labels; only use molecular ground truth: '+str(bad[:10]))
        df['idh_label']=labels.map(LABEL_MAP).astype(int)
    elif 'idh_label' in df:
        df=df.drop(columns='idh_label')
    root=Path(data_dir or p.parent).expanduser().resolve()
    for mod in sequences:
        col=f'{mod}_path'; arr=[]
        for raw in df[col]:
            name=str(raw).strip()
            if not name: raise ValueError(f'Missing required {col}; patient: {df.iloc[len(arr)].patient_id}')
            candidate=Path(name).expanduser()
            candidate=(candidate if candidate.is_absolute() else root/candidate).resolve()
            if not candidate.is_file(): raise FileNotFoundError(f'{df.iloc[len(arr)].patient_id}: {candidate}')
            if not (str(candidate).lower().endswith(('.nii','.nii.gz'))):
                raise ValueError(f'Expected NIfTI (.nii/.nii.gz): {candidate}')
            arr.append(str(candidate))
        df[col]=arr
    if 'seg_path' in df:
        resolved=[]
        for raw in df.seg_path:
            if not str(raw).strip(): resolved.append(''); continue
            f=Path(str(raw)); f=f if f.is_absolute() else root/f
            if not f.is_file(): raise FileNotFoundError(f'Optional segmentation path invalid: {f}')
            resolved.append(str(f.resolve()))
        df['seg_path']=resolved
    # Reject accidental self-reuse of a single series as two distinct MRI sequences.
    for _,row in df.iterrows():
        seq_paths=[str(row[f'{name}_path']) for name in sequences]
        if len(set(seq_paths))!=len(seq_paths):
            raise ValueError(f'{row.patient_id}: same MRI file assigned to multiple sequence fields')
    # If identical image file is assigned to multiple patient IDs, stop.
    for c in (f'{s}_path' for s in sequences):
        if df[c].duplicated().any(): raise ValueError(f'Cross-patient reuse of {c}; inspect duplicates')
    return df.reset_index(drop=True)

def check_file_hashes(df, columns=PATH_FIELDS):
    """Optional expensive audit for exact identical files (not perceptual near duplicates)."""
    records=[]; seen={}
    for r in df.itertuples(index=False):
        for col in columns:
            if col not in df: continue
            p=Path(getattr(r,col)); h=hashlib.sha256()
            with p.open('rb') as stream:
                while True:
                    block=stream.read(2**20)
                    if not block: break
                    h.update(block)
            digest=h.hexdigest()
            if digest in seen and seen[digest][0]!=r.patient_id:
                records.append({'patient_a':seen[digest][0],'patient_b':r.patient_id,'path_a':seen[digest][1],'path_b':str(p)})
            seen[digest]=(r.patient_id,str(p))
    return pd.DataFrame(records)

def patient_split(df, config):
    if 'idh_label' not in df: raise ValueError('Cannot split for training without molecular labels')
    if df.idh_label.nunique()!=2: raise ValueError('Dataset must include both IDH classes')
    if config.val_fraction+config.test_fraction>=0.8: raise ValueError('Development fraction too small')
    counts=df.idh_label.value_counts()
    if counts.min()<3: raise ValueError('Too few patients in at least one class for 3 independent splits')
    try:
        dev,test=train_test_split(df,test_size=config.test_fraction,stratify=df.idh_label,random_state=config.seed)
        relative_val=config.val_fraction/(1-config.test_fraction)
        train,val=train_test_split(dev,test_size=relative_val,stratify=dev.idh_label,random_state=config.seed+1)
    except ValueError as ex:
        raise ValueError('Class sizes insufficient for stratified train/val/test splitting: '+str(ex)) from ex
    parts={'train':train.reset_index(drop=True),'val':val.reset_index(drop=True),'test':test.reset_index(drop=True)}
    assert sum(len(d) for d in parts.values())==len(df)
    assert all(parts[a].patient_id.isin(parts[b].patient_id).sum()==0 for a in parts for b in parts if a!=b)
    return parts

def save_splits(splits,path):
    p=Path(path); p.mkdir(parents=True,exist_ok=True)
    for k,d in splits.items(): d.to_csv(p/f'{k}_manifest.csv',index=False)

def _load_nifti(path):
    import nibabel as nib
    image=nib.as_closest_canonical(nib.load(str(path)))
    if len(image.shape)!=3: raise ValueError(f'Expected 3D MRI: {path} shape={image.shape}')
    return image

def _n4_correct(img):
    import SimpleITK as sitk
    import nibabel as nib
    x=np.asarray(img.get_fdata(dtype=np.float32), dtype=np.float32)
    m=(x>0).astype(np.uint8)
    if m.sum()==0: raise ValueError('Empty brain mask for N4 correction')
    itk=sitk.GetImageFromArray(x); mask=sitk.GetImageFromArray(m)
    corrected=sitk.N4BiasFieldCorrectionImageFilter().Execute(itk,mask)
    return nib.Nifti1Image(sitk.GetArrayFromImage(corrected),img.affine)

def _centre_crop_pad(vol,centre,shape):
    result=np.zeros(tuple(shape),dtype=np.float32)
    in_slices=[]; out_slices=[]
    for d,(sz,c,s) in enumerate(zip(vol.shape,centre,shape)):
        st=int(round(c))-s//2; end=st+s; lo=max(0,st); hi=min(sz,end)
        if hi<=lo: continue
        in_slices.append(slice(lo,hi)); out_slices.append(slice(lo-st,hi-st))
    result[tuple(out_slices)]=vol[tuple(in_slices)]
    return result

def preprocess_patient(row, config):
    """Orientation standardise, affine-resample into a common physical grid, mask, crop, normalise.
    Assumes source modalities have been registered or are in aligned physical coordinates.
    Simply resampling DOES NOT perform deformable/rigid registration of unaligned scans.
    """
    import nibabel as nib
    from nibabel.processing import resample_to_output, resample_from_to
    if not config.assume_coregistered:
        raise ValueError('Unregistered MRI input: register T1/T1ce/T2/FLAIR externally first (e.g., ANTs), QC, then set assume_coregistered=True.')
    reference_name='t1ce' if 't1ce' in config.sequences else config.sequences[0]
    original={name:_load_nifti(row[f'{name}_path']) for name in config.sequences}
    if config.use_n4_bias_correction:
        original={k:_n4_correct(v) for k,v in original.items()}
    ref=resample_to_output(original[reference_name],voxel_sizes=(config.voxel_spacing,)*3,order=1)
    tgt=(ref.shape,ref.affine)
    aligned={name: np.asarray((ref if name==reference_name else
                       resample_from_to(vol,tgt,order=1)).get_fdata(dtype=np.float32),dtype=np.float32)
                       for name,vol in original.items()}
    if any(not np.isfinite(x).all() for x in aligned.values()): raise ValueError('NaN/Inf in image')
    masks=np.stack([np.abs(x)>1e-6 for x in aligned.values()]).any(0)
    if masks.sum()<500: raise ValueError(f'Empty/near-empty MRI for patient {row["patient_id"]}')
    coords=np.argwhere(masks); centre=np.median(coords,axis=0).astype(int)
    seg=None
    if 'seg_path' in row.index and isinstance(row['seg_path'],str) and row['seg_path'].strip():
        seg_image=_load_nifti(row['seg_path'])
        seg=resample_from_to(seg_image,tgt,order=0).get_fdata(dtype=np.float32)>0
        if seg.sum()>0: centre=np.median(np.argwhere(seg),axis=0).astype(int)
    arr=[]
    for name in config.sequences:
        vol=_centre_crop_pad(aligned[name],centre,config.volume_shape)
        mask=(np.abs(vol)>1e-6)
        if mask.sum()<100: raise ValueError(f'Empty aligned {name}, check registration and affine: {row["patient_id"]}')
        vals=vol[mask]; low,high=np.percentile(vals,[0.5,99.5]); vol=np.clip(vol,low,high)
        vals=vol[mask]; vol[mask]=(vals-vals.mean())/(vals.std()+1e-6)
        vol[~mask]=0
        arr.append(np.clip(vol,-5,5).astype(np.float32))
    seg_out=_centre_crop_pad(seg.astype(np.float32),centre,config.volume_shape)>0 if seg is not None else None
    # Anatomical alignment QC: image masks must overlap; low overlap often means an affine/registration error.
    overlap_masks=[np.abs(a)>1e-6 for a in arr]
    ref_mask=overlap_masks[0]
    overlap=[float((ref_mask & m).sum() / (ref_mask | m).sum()) for m in overlap_masks[1:]]
    if overlap and min(overlap)<0.20:
        raise ValueError(f'Sequence alignment failed QC (foreground IoU={overlap}), patient={row["patient_id"]}')
    return np.stack(arr).astype(np.float32),seg_out

def _cache_key(row,cfg):
    identity={'id':str(row.patient_id),'vol_shape':cfg.volume_shape,'spacing':cfg.voxel_spacing,
              'sequences':cfg.sequences,'n4':cfg.use_n4_bias_correction,'version':'optionB_v4'}
    for k in [*PATH_FIELDS,'seg_path']:
        if k in row.index and str(row[k]):
            p=Path(str(row[k])); identity[k]=[str(p),p.stat().st_size,p.stat().st_mtime_ns]
    return hashlib.sha256(json.dumps(identity,sort_keys=True,default=str).encode()).hexdigest()[:20]

def cached_preprocess(row,cfg):
    root=Path(cfg.cache_dir); root.mkdir(parents=True,exist_ok=True)
    dest=root/(_cache_key(row,cfg)+'.npz')
    if not dest.exists():
        x,seg=preprocess_patient(row,cfg)
        tmp=dest.with_suffix('.temp.npz')
        np.savez_compressed(tmp, x=x.astype(np.float16),seg=(seg.astype(np.uint8) if seg is not None else np.zeros((0,),dtype=np.uint8)))
        os.replace(tmp,dest)
    with np.load(dest) as item:
        x=item['x'].astype(np.float32); seg=item['seg'].astype(bool)
        return x,seg if seg.size else None

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

class PatientDataset(Dataset):
    def __init__(self,df,cfg,include_labels=True):
        self.df=df.reset_index(drop=True); self.cfg=cfg; self.include_labels=include_labels
    def __len__(self): return len(self.df)
    def __getitem__(self,i):
        row=self.df.iloc[i]
        # Cache only deterministic, unaugmented views. Training augmentations remain stochastic.
        views=Path(self.cfg.cache_dir)/'views'
        views.mkdir(parents=True,exist_ok=True)
        key=hashlib.sha256((str(_cache_key(row,self.cfg))+str(self.cfg.sequences)+
             f'{self.cfg.locations}_{self.cfg.adjacent_slices}_{self.cfg.input_resolution}').encode()).hexdigest()[:18]
        file=views/(str(row.patient_id)+'_'+key+'.npy')
        if not file.exists():
            v,seg=cached_preprocess(row,self.cfg)
            x=volume_to_25d(v,self.cfg,seg)
            tmp=file.with_suffix('.tmp.npy')
            np.save(tmp,x.numpy().astype(np.float16))
            os.replace(tmp,file)
        x=torch.from_numpy(np.load(file).astype(np.float32))
        label=float(row.idh_label) if self.include_labels else -1.0
        return x,torch.tensor(label,dtype=torch.float32),str(row.patient_id)

def loaders_from_splits(splits,cfg,balanced=False):
    datasets={k:PatientDataset(v,cfg) for k,v in splits.items()}
    sampler=None
    if balanced:
        df=splits['train']; col=cfg.domain_field
        if col not in df or not df[col].astype(str).str.strip().all():
            raise ValueError(f'Balanced sampling requested, but {col} unavailable')
        weights=1.0/df.groupby([col,'idh_label'])['patient_id'].transform('count').to_numpy()
        weights=weights/weights.mean()
        sampler=WeightedRandomSampler(torch.tensor(weights,dtype=torch.double),num_samples=len(df),replacement=True)
    return {k:DataLoader(ds,batch_size=cfg.batch_size,shuffle=k=='train' and sampler is None,
                     sampler=sampler if k=='train' else None,num_workers=cfg.workers,
                     pin_memory=cfg.device.startswith('cuda'),persistent_workers=cfg.workers>0)
              for k,ds in datasets.items()}

class ConvBlock(nn.Module):
    def __init__(self,c1,c2):
        super().__init__(); self.layers=nn.Sequential(nn.Conv2d(c1,c2,3,padding=1,bias=False),
            nn.BatchNorm2d(c2),nn.SiLU(),nn.Conv2d(c2,c2,3,padding=1,bias=False),
            nn.BatchNorm2d(c2),nn.SiLU(),nn.MaxPool2d(2))
    def forward(self,x): return self.layers(x)

class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,weight): ctx.weight=weight; return x.view_as(x)
    @staticmethod
    def backward(ctx,grad): return -ctx.weight*grad,None

class MRI25DNet(nn.Module):
    def __init__(self,cfg,n_domains=0):
        super().__init__(); inp=len(cfg.sequences)*cfg.adjacent_slices; h=cfg.hidden_dim
        self.locations=cfg.locations; self.n_views=3*cfg.locations
        if cfg.backbone == 'light_cnn':
            self.encoder=nn.Sequential(ConvBlock(inp,24),ConvBlock(24,48),ConvBlock(48,96),
                                      nn.Conv2d(96,h,3,padding=1),nn.SiLU())
        elif cfg.backbone == 'resnet18':
            from torchvision.models import resnet18, ResNet18_Weights
            backbone=resnet18(weights=ResNet18_Weights.DEFAULT if cfg.pretrained_backbone else None)
            old=backbone.conv1
            backbone.conv1=nn.Conv2d(inp,old.out_channels,kernel_size=old.kernel_size,
                                    stride=old.stride,padding=old.padding,bias=False)
            if cfg.pretrained_backbone:
                with torch.no_grad():
                    backbone.conv1.weight.copy_(old.weight.mean(1,keepdim=True).repeat(1,inp,1,1)*(3.0/inp))
            self.encoder=nn.Sequential(*list(backbone.children())[:-2],nn.Conv2d(512,h,1),nn.SiLU())
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
        else: raise ValueError('backbone must be light_cnn, resnet18 or efficientnet_b0')
        self.pool=nn.AdaptiveAvgPool2d(1)
        self.view_embedding=nn.Embedding(3,h)
        self.attention=nn.Sequential(nn.Linear(h,h//2),nn.Tanh(),nn.Linear(h//2,1))
        self.classifier=nn.Sequential(nn.LayerNorm(h),nn.Dropout(cfg.dropout),nn.Linear(h,1))
        self.domain_head=nn.Sequential(nn.LayerNorm(h),nn.Linear(h,h//2),nn.ReLU(),nn.Linear(h//2,n_domains)) if n_domains>1 else None
    def forward(self,x,grl_lambda=0.0,return_embedding=False):
        if x.ndim!=5 or x.shape[1]!=self.n_views: raise ValueError(f'Expected input [B,{self.n_views},C,H,W], got {tuple(x.shape)}')
        b,n,c,h,w=x.shape
        maps=self.encoder(x.reshape(b*n,c,h,w))
        v=self.pool(maps).flatten(1).reshape(b,n,-1)
        plane_ids=torch.arange(n,device=x.device)//self.locations
        v=v+self.view_embedding(plane_ids)[None]
        weights=torch.softmax(self.attention(v),dim=1)
        z=(v*weights).sum(1)
        logits=self.classifier(z).squeeze(1)
        domain_logits=self.domain_head(GradientReversal.apply(z,grl_lambda)) if self.domain_head is not None else None
        return (logits,z,domain_logits) if return_embedding else logits

def n_params(model): return sum(p.numel() for p in model.parameters() if p.requires_grad)

def perturb(x,kind='random',strength=0.4,seed=None):
    """Paired acquisition-like perturbations; fixed seed optional for reproducible evaluation.
    Avoid spatial translation/rotation: anatomy and modality alignment are retained.
    """
    if strength<=0: return x.clone()
    out=x.clone(); b,n,c,h,w=x.shape
    if kind=='random':
        kinds=['noise','blur','resolution','bias','gamma','scale']
        if seed is not None:
            gen=np.random.default_rng(seed); kind=kinds[int(gen.integers(len(kinds)))]
        else: kind=random.choice(kinds)
    if kind=='noise':
        if seed is not None:
            rng=torch.Generator(device=x.device).manual_seed(int(seed)); eps=torch.randn(x.shape,device=x.device,dtype=x.dtype,generator=rng)
        else: eps=torch.randn_like(x)
        out=x+(0.18*strength)*eps
    elif kind=='blur':
        # Separable Gaussian blur, no kornia dependency; independent per image channel.
        sigma=0.5+1.1*strength; k=2*int(math.ceil(3*sigma))+1
        t=torch.arange(k,device=x.device,dtype=x.dtype)-k//2; kernel=torch.exp(-t*t/(2*sigma*sigma)); kernel=kernel/kernel.sum()
        z=x.reshape(b*n,c,h,w)
        z=torch.nn.functional.conv2d(z, kernel.view(1,1,k,1).expand(c,1,k,1),padding=(k//2,0),groups=c)
        z=torch.nn.functional.conv2d(z, kernel.view(1,1,1,k).expand(c,1,1,k),padding=(0,k//2),groups=c)
        out=z.reshape_as(x)
    elif kind=='resolution':
        factor=max(0.45,1-0.55*strength); low=(max(8,int(h*factor)),max(8,int(w*factor)))
        z=x.reshape(b*n,c,h,w)
        z=torch.nn.functional.interpolate(z,size=low,mode='area')
        out=torch.nn.functional.interpolate(z,size=(h,w),mode='bilinear',align_corners=False).reshape_as(x)
    elif kind=='bias':
        a=torch.linspace(-1,1,h,device=x.device,dtype=x.dtype)
        v=torch.linspace(-1,1,w,device=x.device,dtype=x.dtype)
        field=1+0.35*strength*(a[:,None]+v[None,:])/2
        out=x*field[None,None,None]
    elif kind=='gamma':
        mag=0.45*strength; gamma=1+mag
        # Sign-preserving response on normalised MRI intensities (not RGB gamma).
        out=torch.sign(x)*torch.abs(x).clamp(min=1e-5).pow(gamma)
    elif kind=='scale': out=x*(1+0.25*strength)
    else: raise ValueError(f'Unknown perturbation type {kind}')
    return out.clamp(-6,6)

def ece_binary(y,p,n_bins=10):
    y=np.asarray(y).astype(int); p=np.asarray(p)
    conf=np.maximum(p,1-p); correct=((p>=0.5)==y)
    bins=np.linspace(0.5,1,n_bins+1); result=0.0
    for i in range(n_bins):
        sub=(conf>=bins[i]) & (conf<bins[i+1] if i<n_bins-1 else conf<=bins[i+1])
        if sub.any(): result+=sub.mean()*abs(correct[sub].mean()-conf[sub].mean())
    return float(result)

def metrics_binary(y,prob,threshold=.5):
    y=np.asarray(y).astype(int); p=np.asarray(prob,dtype=float)
    if len(y)==0: raise ValueError('Cannot evaluate an empty test set')
    pred=(p>=threshold).astype(int); cm=confusion_matrix(y,pred,labels=[0,1]); tn,fp,fn,tp=cm.ravel()
    out={'n':len(y),'n_wildtype':int((y==0).sum()),'n_mutant':int((y==1).sum()),
         'accuracy':float(accuracy_score(y,pred)),'balanced_accuracy':float(balanced_accuracy_score(y,pred)),
         'precision':float(precision_score(y,pred,zero_division=0)),
         'recall':float(recall_score(y,pred,zero_division=0)),
         'sensitivity':float(tp/(tp+fn)) if tp+fn else None,
         'specificity':float(tn/(tn+fp)) if tn+fp else None,
         'macro_f1':float(f1_score(y,pred,average='macro',labels=[0,1],zero_division=0)),
         'auroc':float(roc_auc_score(y,p)) if np.unique(y).size==2 else None,
         'auprc':float(average_precision_score(y,p)) if np.unique(y).size==2 else None,
         'brier':float(brier_score_loss(y,p)),'ece':ece_binary(y,p),
         'confusion_matrix':cm.tolist()}
    return out

def bootstrap_ci(y,p,metric='auroc',n_boot=500,seed=123):
    y=np.asarray(y);p=np.asarray(p);rng=np.random.default_rng(seed); vals=[]
    if len(y)<5: return [None,None]
    for _ in range(n_boot):
        idx=rng.integers(0,len(y),len(y)); v=metrics_binary(y[idx],p[idx]).get(metric)
        if v is not None: vals.append(v)
    return [float(v) for v in np.percentile(vals,[2.5,97.5])] if len(vals)>10 else [None,None]

def domain_metrics(predictions,col='vendor',threshold=.5):
    if col not in predictions or predictions[col].replace('',np.nan).isna().all(): return pd.DataFrame()
    rows=[]
    for domain,group in predictions.groupby(col,dropna=False):
        if not str(domain).strip(): continue
        v=metrics_binary(group.idh_label,group.probability,threshold)
        rows.append({col:domain,**{k:value for k,value in v.items() if k!='confusion_matrix'}})
    return pd.DataFrame(rows)

def autocast_ctx(cfg):
    return torch.autocast(device_type='cuda',dtype=torch.float16,enabled=cfg.device.startswith('cuda') and cfg.mixed_precision)

@torch.no_grad()
def predict_loader(model,loader,cfg):
    model.eval(); rows=[]
    for x,y,pids in loader:
        logits=model(x.to(cfg.device)).float().cpu().numpy()
        for pid,target,logit in zip(pids,y.numpy(),logits):
            rows.append({'patient_id':str(pid),'idh_label':int(target) if target>=0 else -1,
                         'logit':float(logit),'probability':float(torch.sigmoid(torch.tensor(float(logit))).item())})
    return pd.DataFrame(rows)

def with_metadata(pred, original):
    if pred.empty: return pred
    meta=original.drop(columns=[c for c in ['idh_label'] if c in original],errors='ignore')
    # avoid merging MRI paths into result CSV by default
    meta=meta[['patient_id']+[c for c in ['site','vendor','field_strength','cohort'] if c in meta]]
    return pred.merge(meta,on='patient_id',how='left',validate='one_to_one')

def save_checkpoint(path,model,cfg,epoch,history,optimizer=None,scheduler=None,scaler=None,best_score=None,n_domains=0):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    info={'model':model.state_dict(),'config':asdict(cfg),'epoch':epoch,'history':history,'best_score':best_score,
          'n_domains':n_domains,'optimizer':optimizer.state_dict() if optimizer is not None else None,
          'scheduler':scheduler.state_dict() if scheduler is not None else None,
          'scaler':scaler.state_dict() if scaler is not None else None}
    tmp=str(path)+'.tmp';torch.save(info,tmp);os.replace(tmp,path)

def load_checkpoint(path,device='cpu'):
    d=torch.load(path,map_location='cpu',weights_only=False)
    cfg=Config(**d['config']);cfg.device=device
    construction_cfg=replace(cfg,pretrained_backbone=False) if cfg.pretrained_backbone else cfg
    model=MRI25DNet(construction_cfg,n_domains=d.get('n_domains',0))
    model.load_state_dict(d['model']);return model.to(device),cfg,d
