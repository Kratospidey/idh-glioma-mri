"""Research experiments for MRI acquisition shortcut auditing. Import idh_core first."""
from __future__ import annotations
import math, json, gc, warnings
from pathlib import Path
from dataclasses import replace, asdict
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, r2_score, mean_absolute_error
from scipy.stats import chi2_contingency, fisher_exact, pointbiserialr
import matplotlib.pyplot as plt
from idh_core import (Config, MRI25DNet, PatientDataset, cached_preprocess, loaders_from_splits,
    seed_everything, perturb, n_params, predict_loader, metrics_binary, with_metadata,
    write_json, save_checkpoint, load_checkpoint, domain_metrics, bootstrap_ci, autocast_ctx,
    volume_to_25d, patient_split, save_splits, SEQUENCES)

VARIANTS={'baseline':(False,False,False,False), 'a1':(True,False,False,False),
          'a2':(False,True,False,False), 'a3':(False,False,True,False),
          'a4':(False,False,False,True), 'combined':(True,True,True,True)}

def domain_encoder(train_df,cfg,enabled):
    if not enabled:return None
    if cfg.domain_field not in train_df or not train_df[cfg.domain_field].astype(str).str.strip().all():
        raise ValueError(f'Adversarial model needs COMPLETE {cfg.domain_field} labels for all training patients')
    enc=LabelEncoder().fit(train_df[cfg.domain_field].astype(str))
    if len(enc.classes_)<2: raise ValueError(f'Adversarial head requires >=2 {cfg.domain_field} categories')
    return enc

# Specialised train function (with explicit metadata argument) avoids global state in notebook.
def run_training(splits,cfg,experiment='baseline',experiment_dir=None,evaluate_test=True):
    if experiment not in VARIANTS: raise ValueError(f'Unknown experiment: {experiment}')
    seed_everything(cfg.seed)
    balanced,aug,cons,adv=flags=VARIANTS[experiment]
    domain_enc=domain_encoder(splits['train'],cfg,adv)
    if balanced and (cfg.domain_field not in splits['train'] or
        not splits['train'][cfg.domain_field].astype(str).str.strip().all()):
        raise ValueError(f'Balanced sampling requires {cfg.domain_field} for all training patients')
    out=Path(experiment_dir or Path(cfg.output_dir)/experiment)
    out.mkdir(parents=True,exist_ok=True)
    save_splits(splits,out/'splits')
    write_json({'experiment':experiment,'flags':dict(zip(('balance','augment','consistency','adversarial'),flags)),
                'config':asdict(cfg),'domain_categories':domain_enc.classes_.tolist() if domain_enc is not None else []},out/'settings.json')
    loaders=loaders_from_splits(splits,cfg,balanced=balanced)
    model=MRI25DNet(cfg,n_domains=len(domain_enc.classes_) if domain_enc is not None else 0).to(cfg.device)
    print(f'{experiment}: {n_params(model):,} trainable parameters')
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg.lr,weight_decay=cfg.weight_decay)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=max(1,cfg.epochs))
    scaler=torch.amp.GradScaler('cuda',enabled=cfg.device.startswith('cuda') and cfg.mixed_precision)
    counts=splits['train'].idh_label.value_counts();weight=(counts.get(0,0)/max(1,counts.get(1,0))) if cfg.use_class_weight else 1.
    pos_weight=torch.tensor(weight,dtype=torch.float32,device=cfg.device)
    history=[];best=-float('inf');best_ep=0;start=1
    best_rank=(-float('inf'),-float('inf'),-float('inf'))
    last=out/'last.pt'
    if cfg.resume and last.exists():
        loaded=torch.load(last,map_location='cpu',weights_only=False)
        if loaded['config'].get('seed')!=cfg.seed or loaded.get('n_domains',0)!=(len(domain_enc.classes_) if domain_enc is not None else 0):
            raise ValueError('Resume configuration differs in random seed or domain count')
        model.load_state_dict(loaded['model']);optimizer.load_state_dict(loaded['optimizer'])
        scheduler.load_state_dict(loaded['scheduler']);scaler.load_state_dict(loaded['scaler'])
        start=loaded['epoch']+1;history=loaded['history'];best=loaded['best_score'] or -float('inf')
        best_ep=max([int(r['epoch']) for r in history if r.get('is_best')],default=0)
        prior=[(round(float(r['val_auroc'] if r['val_auroc'] is not None else r['val_macro_f1']),4),
                round(float(r['val_macro_f1']),4),-float(r['val_loss']))
               for r in history if r.get('is_best')]
        if prior:best_rank=max(prior)
        print('Resuming',experiment,'from epoch',start)
    # Use a patient->domain dict within this run, never a module-global mutable variable.
    train_meta=splits['train'].set_index('patient_id')[[cfg.domain_field]].to_dict('index') if domain_enc is not None else {}
    def train_epoch(loader,training=True,epoch=1):
        model.train(training)
        if training and cfg.pretrained_backbone and epoch<=cfg.freeze_pretrained_epochs:
            if cfg.backbone=='resnet18': model.encoder[:8].eval()
            elif cfg.backbone=='efficientnet_b0': model.encoder[0].eval()  # keep pretrained BatchNorm statistics fixed during warm-up
        if training:optimizer.zero_grad(set_to_none=True)
        total=0.; num=0; labels=[]; probs=[]
        for i,(x,y,ids) in enumerate(loader):
            x=x.to(cfg.device);y=y.to(cfg.device)
            with torch.set_grad_enabled(training):
                if training and aug:x=perturb(x,'random',cfg.augment_strength)
                with autocast_ctx(cfg):
                    logit,z,domains=model(x,grl_lambda=1.0 if training and adv else 0.,return_embedding=True)
                    loss=F.binary_cross_entropy_with_logits(logit,y,pos_weight=pos_weight)
                    if training and cons:
                        p2=model(perturb(x,'random',cfg.augment_strength))
                        lambda_now=cfg.consistency_lambda*min(1,epoch/max(1,cfg.consistency_warmup_epochs))
                        loss=loss+lambda_now*F.mse_loss(torch.sigmoid(p2.float()),torch.sigmoid(logit.float()).detach())
                    if training and adv:
                        labels_d=domain_enc.transform([train_meta[pid][cfg.domain_field] for pid in ids])
                        dl=torch.tensor(labels_d,dtype=torch.long,device=cfg.device)
                        loss=loss+cfg.adversarial_lambda*F.cross_entropy(domains,dl)
                if training:
                    scaler.scale(loss/max(1,cfg.accumulation_steps)).backward()
                    if (i+1)%cfg.accumulation_steps==0 or i+1==len(loader):
                        scaler.unscale_(optimizer);nn.utils.clip_grad_norm_(model.parameters(),5.)
                        scaler.step(optimizer);scaler.update();optimizer.zero_grad(set_to_none=True)
            num+=len(y);total+=float(loss.detach().item())*len(y)
            labels.extend(y.detach().int().cpu().tolist());probs.extend(torch.sigmoid(logit.detach().float()).cpu().tolist())
        m=metrics_binary(labels,probs,cfg.threshold)
        return {'loss':total/max(num,1),'auroc':m['auroc'],'macro_f1':m['macro_f1'],'accuracy':m['accuracy']}
    for epoch in range(start,cfg.epochs+1):
        if cfg.backbone in ('resnet18','efficientnet_b0') and cfg.pretrained_backbone and cfg.freeze_pretrained_epochs:
            train_encoder=(epoch>cfg.freeze_pretrained_epochs)
            backbone_module=model.encoder[:8] if cfg.backbone=='resnet18' else model.encoder[0]
            for p in backbone_module.parameters():p.requires_grad_(train_encoder)
            if epoch==1:print('Pretrained backbone warm-up: frozen for',cfg.freeze_pretrained_epochs,'epochs')
            if epoch==cfg.freeze_pretrained_epochs+1:print('Unfreezing pretrained ResNet backbone for fine-tuning')
        tr=train_epoch(loaders['train'],True,epoch)
        va=train_epoch(loaders['val'],False,epoch)
        scheduler.step();score=va['auroc'] if va['auroc'] is not None else va['macro_f1']
        # AUROC primary; for ties prefer validation macro-F1, then lower validation loss.
        # Fixes the observed epoch-1 checkpoint despite later perfect validation F1.
        rank=(round(float(score),4),round(float(va['macro_f1']),4),-float(va['loss']))
        improved=rank>best_rank
        if improved:best=score;best_rank=rank;best_ep=epoch
        history.append({'epoch':epoch,**{'train_'+k:v for k,v in tr.items()},**{'val_'+k:v for k,v in va.items()},'is_best':improved})
        save_checkpoint(last,model,cfg,epoch,history,optimizer,scheduler,scaler,best,len(domain_enc.classes_) if domain_enc is not None else 0)
        if improved:save_checkpoint(out/'best.pt',model,cfg,epoch,history,best_score=best,n_domains=len(domain_enc.classes_) if domain_enc is not None else 0)
        pd.DataFrame(history).to_csv(out/'history.csv',index=False)
        print(f'{experiment} {epoch:02d}: train_loss={tr["loss"]:.4f} val_loss={va["loss"]:.4f} val_AUROC={va["auroc"]} best_epoch={best_ep}')
        if epoch-best_ep>=cfg.early_patience:print('Early stopping');break
    if not (out/'best.pt').exists(): raise RuntimeError('No valid checkpoint produced')
    model,_,_=load_checkpoint(out/'best.pt',cfg.device)
    for split_name in (('val','test') if evaluate_test else ('val',)):
        pred=with_metadata(predict_loader(model,loaders[split_name],cfg),splits[split_name]);pred.to_csv(out/f'{split_name}_predictions.csv',index=False)
        m=metrics_binary(pred.idh_label,pred.probability,cfg.threshold)
        m['auroc_bootstrap_ci95']=bootstrap_ci(pred.idh_label,pred.probability,seed=cfg.seed)
        dm=domain_metrics(pred,cfg.domain_field,cfg.threshold)
        if not dm.empty:
            dm.to_csv(out/f'{split_name}_domains.csv',index=False)
            a=dm.auroc.dropna();m['worst_domain_auroc']=float(a.min()) if len(a) else None
        else:m['worst_domain_auroc']=None
        write_json(m,out/f'{split_name}_metrics.json')
    del loaders;gc.collect()
    if cfg.device.startswith('cuda'):torch.cuda.empty_cache()
    return model,out

@torch.no_grad()
def extract_embeddings(model,loader,cfg):
    model.eval();features=[];ids=[];labels=[]
    for x,y,p in loader:
        _,z,_=model(x.to(cfg.device),return_embedding=True)
        features.append(z.detach().float().cpu().numpy());ids+=list(p);labels+=y.int().tolist()
    return np.concatenate(features),ids,np.asarray(labels)

def audit_metadata(df,out):
    """Association audit: association != causation; only measured metadata fields."""
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    pd.crosstab(df.idh_label,columns='patients').to_csv(out/'label_counts.csv')
    results=[]
    for col in ('site','vendor','field_strength','voxel_spacing'):
        if col not in df: continue
        clean=df[[col,'idh_label']].copy().replace('',np.nan).dropna()
        if len(clean)<5:continue
        x=pd.to_numeric(clean[col],errors='coerce')
        # field strength is categorical by default (1.5, 3T).
        if col=='voxel_spacing' and x.notna().sum()>5 and x.nunique()>5:
            rho,pv=pointbiserialr(clean.loc[x.notna(),'idh_label'],x.dropna())
            results.append({'field':col,'type':'point_biserial','statistic':float(rho),'p':float(pv),'n':len(x.dropna())})
            continue
        tab=pd.crosstab(clean[col].astype(str),clean.idh_label).reindex(columns=[0,1],fill_value=0)
        tab.to_csv(out/f'{col}_contingency.csv')
        if tab.shape[0]>1:
            if tab.shape==(2,2) and (tab.values<5).any():stat,pv=fisher_exact(tab.values);method='fisher_exact'
            else:
                stat,pv,_,_=chi2_contingency(tab.values);method='chi2'
            results.append({'field':col,'type':method,'statistic':float(stat),'p':float(pv),'n':len(clean)})
        ax=tab.div(tab.sum(axis=1),axis=0).plot(kind='bar',stacked=True,figsize=(8,4))
        ax.set_title(f'IDH distribution by {col}');ax.set_ylabel('proportion');ax.legend(['wildtype','mutant']);plt.tight_layout()
        plt.savefig(out/f'{col}_distribution.png',dpi=160);plt.close()
    result=pd.DataFrame(results);result.to_csv(out/'association_statistics.csv',index=False)
    return result

def image_features(vol):
    """Preprocessed image-derived pixels (not header metadata). Coarse spatial + intensity stats."""
    features=[]
    for v in vol:
        z=torch.tensor(v[None,None],dtype=torch.float32)
        down=F.adaptive_avg_pool3d(z,(6,6,6)).flatten().numpy()
        nz=v[v!=0];features.extend(down.tolist())
        features.extend([float(np.mean(nz)),float(np.std(nz)),*np.percentile(nz,[10,25,50,75,90]).tolist()])
    return np.asarray(features,dtype=np.float32)

def probe_acquisition(model,splits,cfg,out,field='vendor'):
    """Patient-disjoint frozen linear probes for image-derived features & embeddings.
    Probe labels are available ONLY where actual metadata is present. Uses training patients
    to fit and validation/test patients as held-out. No IDH-model fine-tuning.
    """
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if any(field not in d for d in splits.values()):
        write_json({'status':'not_performed','reason':f'No {field} metadata'},out/'probe_status.json');return pd.DataFrame()
    usable={k:d[d[field].astype(str).str.strip().ne('')].copy() for k,d in splits.items()}
    train_domains=usable['train'][field].astype(str)
    if train_domains.nunique()<2:
        write_json({'status':'not_performed','reason':f'Only {train_domains.nunique()} usable training {field} domains'},out/'probe_status.json');return pd.DataFrame()
    outputs=[]
    for feature_type in ('pixels','embedding'):
        xs={};ys={};ids={}
        for split,df in usable.items():
            if not len(df):continue
            if feature_type=='pixels':
                arr=np.stack([image_features(cached_preprocess(row,cfg)[0]) for _,row in df.iterrows()]);pid=df.patient_id.tolist()
            else:
                dl=torch.utils.data.DataLoader(PatientDataset(df,cfg),batch_size=cfg.batch_size,shuffle=False,num_workers=cfg.workers)
                arr,pid,_=extract_embeddings(model,dl,cfg)
            xs[split]=arr;ys[split]=df[field].astype(str).to_numpy();ids[split]=pid
        if 'train' not in xs:continue
        # regression for measured, continuous quantitative nuisance variables
        vals=pd.to_numeric(pd.Series(ys['train']),errors='coerce')
        is_continuous=(field in ('voxel_spacing','echo_time','repetition_time') and vals.notna().all() and vals.nunique()>5)
        if is_continuous:
            probe=make_pipeline(StandardScaler(),Ridge(alpha=10.0))
            probe.fit(xs['train'],ys['train'].astype(float))
        else:
            probe=make_pipeline(StandardScaler(),LogisticRegression(max_iter=500,class_weight='balanced'))
            probe.fit(xs['train'],ys['train'])
        for split in ('val','test'):
            if split not in xs:continue
            if is_continuous:
                yi=ys[split].astype(float);p=probe.predict(xs[split]);score={'mae':mean_absolute_error(yi,p),'r2':r2_score(yi,p) if len(yi)>1 else None}
            else:
                yi=ys[split]; p=probe.predict(xs[split]);major=train_domains.value_counts().idxmax()
                score={'accuracy':accuracy_score(yi,p),'macro_f1':f1_score(yi,p,average='macro',zero_division=0),
                       'majority_accuracy':float(np.mean(yi==major)),
                       'confusion_matrix':confusion_matrix(yi,p,labels=sorted(set(yi)|set(p))).tolist()}
            outputs.append({'features':feature_type,'target':field,'partition':split,'n':len(yi),**score})
            if not is_continuous:
                comparisons=pd.DataFrame({'patient_id':ids[split], 'true_domain':yi, 'predicted_domain':p})
                comparisons.to_csv(out/f'{feature_type}_{split}_{field}_predictions.csv',index=False)
                labs=sorted(set(yi)|set(p)); mat=confusion_matrix(yi,p,labels=labs)
                fig,ax=plt.subplots(figsize=(max(5,len(labs)),max(4,len(labs)*.6)))
                ax.imshow(mat,cmap='Blues')
                ax.set(xticks=range(len(labs)),yticks=range(len(labs)),xticklabels=labs,yticklabels=labs,
                       xlabel='Predicted acquisition domain',ylabel='True acquisition domain',
                       title=f'{feature_type} {field} probe ({split})')
                plt.setp(ax.get_xticklabels(),rotation=45,ha='right')
                for k in range(len(labs)):
                    for j in range(len(labs)):
                        ax.text(j,k,str(mat[k,j]),ha='center',va='center',fontsize=9)
                fig.tight_layout();fig.savefig(out/f'{feature_type}_{split}_{field}_confusion.png',dpi=160);plt.close(fig)
        # Probe coefficients/normaliser fitted strictly on train (no test data in preprocessing)
    tab=pd.DataFrame(outputs);tab.to_csv(out/'probes.csv',index=False)
    return tab

def counterfactual_audit(model,test_df,cfg,out,kinds=('noise','blur','resolution','bias','gamma','scale'),strengths=(0.25,0.5)):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    loader=torch.utils.data.DataLoader(PatientDataset(test_df,cfg),batch_size=cfg.batch_size,shuffle=False,num_workers=cfg.workers)
    model.eval();original=predict_loader(model,loader,cfg)
    all_preds=[];all_metrics=[]
    original_metrics=metrics_binary(original.idh_label,original.probability,cfg.threshold)
    for kind in kinds:
        for strength in strengths:
            pred=[]
            with torch.no_grad():
                for x,y,pid in loader:
                    x=x.to(cfg.device)
                    # Fixed perturbation for reproducible paired comparisons within a run.
                    xp=torch.stack([perturb(x[i:i+1],kind,strength,seed=cfg.seed+index)[0]
                         for i,index in enumerate([sum(bytearray(k.encode())) for k in pid])])
                    prob=torch.sigmoid(model(xp).float()).cpu().numpy()
                    pred.extend(zip(pid,prob))
            after=pd.DataFrame(pred,columns=['patient_id','perturbed_probability'])
            merged=original.merge(after,on='patient_id',validate='one_to_one')
            m=metrics_binary(merged.idh_label,merged.perturbed_probability,cfg.threshold)
            m.update({'perturbation':kind,'strength':strength,
                      'probability_drift':float(np.abs(merged.probability-merged.perturbed_probability).mean()),
                      'prediction_disagreement':float(((merged.probability>=cfg.threshold)!=(merged.perturbed_probability>=cfg.threshold)).mean()),
                      'label_flip_rate':float(((merged.probability>=cfg.threshold)!=(merged.perturbed_probability>=cfg.threshold)).mean()),
                      'auroc_change':m['auroc']-original_metrics['auroc'] if m['auroc'] is not None and original_metrics['auroc'] is not None else None,
                      'macro_f1_change':m['macro_f1']-original_metrics['macro_f1'],
                      'ece_change':m['ece']-original_metrics['ece'],
                      'brier_change':m['brier']-original_metrics['brier'],
                      'sensitivity_change':(m['sensitivity']-original_metrics['sensitivity']) if m['sensitivity'] is not None and original_metrics['sensitivity'] is not None else None,
                      'specificity_change':(m['specificity']-original_metrics['specificity']) if m['specificity'] is not None and original_metrics['specificity'] is not None else None})
            all_metrics.append({k:v for k,v in m.items() if k!='confusion_matrix'})
            merged['perturbation']=kind;merged['strength']=strength
            all_preds.append(merged)
    summary=pd.DataFrame(all_metrics);detail=pd.concat(all_preds,ignore_index=True)
    summary.to_csv(out/'perturbation_metrics.csv',index=False);detail.to_csv(out/'paired_predictions.csv',index=False)
    for name in ('probability_drift','label_flip_rate','auroc','macro_f1','ece'):
        if name not in summary:continue
        ax=summary.pivot(index='perturbation',columns='strength',values=name).plot(kind='bar',figsize=(8,4))
        ax.set_title(name+' under acquisition-like changes');ax.set_ylabel(name);plt.tight_layout()
        plt.savefig(out/(name+'_perturbations.png'),dpi=180);plt.close()
    return summary

def calibrate_temperature(val_predictions,test_predictions,out=None):
    v=val_predictions.copy();t=test_predictions.copy()
    if v.idh_label.nunique()<2:raise ValueError('Temperature scaling needs both classes in validation set')
    logits=torch.tensor(v.logit.to_numpy(),dtype=torch.float64)
    labels=torch.tensor(v.idh_label.to_numpy(),dtype=torch.float64)
    logtemp=torch.nn.Parameter(torch.zeros([],dtype=torch.float64))
    opt=torch.optim.LBFGS([logtemp],lr=.1,max_iter=100,line_search_fn='strong_wolfe')
    def closure():
        opt.zero_grad();loss=F.binary_cross_entropy_with_logits(logits/torch.exp(logtemp),labels)
        loss.backward();return loss
    opt.step(closure)
    temperature=float(torch.exp(logtemp.detach()).clamp(min=.05,max=20.).item())
    for frame in (v,t):
        frame['calibrated_probability']=1/(1+np.exp(-np.clip(frame.logit.to_numpy()/temperature,-60,60)))
    before=metrics_binary(t.idh_label,t.probability);after=metrics_binary(t.idh_label,t.calibrated_probability)
    if out:
        out=Path(out);out.mkdir(parents=True,exist_ok=True)
        write_json({'temperature':temperature,'test_uncalibrated':before,'test_calibrated':after},out/'calibration_metrics.json')
        t.to_csv(out/'calibrated_test_predictions.csv',index=False)
        plot_reliability(t,out/'reliability_diagram.png')
    return temperature,t

def plot_reliability(df,path,n_bins=10):
    fig,ax=plt.subplots(figsize=(5.5,5))
    for key,name in [('probability','Raw'),('calibrated_probability','Calibrated')]:
        if key not in df:continue
        p=np.asarray(df[key]);y=np.asarray(df.idh_label)
        bins=np.linspace(0,1,n_bins+1);cent=[];fractions=[]
        for j in range(n_bins):
            mask=(p>=bins[j])&(p<(bins[j+1] if j<n_bins-1 else bins[j+1]+.0001))
            if mask.any():cent.append(p[mask].mean());fractions.append(y[mask].mean())
        ax.plot(cent,fractions,'o-',label=name)
    ax.plot([0,1],[0,1],'--',label='Perfect calibration');ax.set(xlabel='Mean predicted mutant probability',ylabel='Observed mutant frequency',title='Reliability diagram');ax.legend();fig.tight_layout()
    fig.savefig(path,dpi=180);plt.close(fig)

def plot_eval(pred,out):
    from sklearn.metrics import roc_curve,precision_recall_curve,ConfusionMatrixDisplay
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    m=metrics_binary(pred.idh_label,pred.probability)
    fig,ax=plt.subplots(figsize=(5,4));ConfusionMatrixDisplay(np.asarray(m['confusion_matrix']),display_labels=['wildtype','mutant']).plot(ax=ax,cmap='Blues',colorbar=False)
    fig.tight_layout();fig.savefig(out/'confusion.png',dpi=180);plt.close(fig)
    if pred.idh_label.nunique()==2:
        fpr,tpr,_=roc_curve(pred.idh_label,pred.probability)
        precision,recall,_=precision_recall_curve(pred.idh_label,pred.probability)
        fig,ax=plt.subplots(figsize=(5,4));ax.plot(fpr,tpr,label=f'AUROC {m["auroc"]:.3f}');ax.plot([0,1],[0,1],'--');ax.set(xlabel='FPR',ylabel='TPR');ax.legend();fig.tight_layout();fig.savefig(out/'roc.png',dpi=180);plt.close(fig)
        fig,ax=plt.subplots(figsize=(5,4));ax.plot(recall,precision);ax.set(xlabel='Recall',ylabel='Precision',title=f'AUPRC {m["auprc"]:.3f}');fig.tight_layout();fig.savefig(out/'pr.png',dpi=180);plt.close(fig)

def leave_one_domain_out(df,cfg,out,field='vendor',minimum_domain_patients=10):
    """Full independent retraining for each held-out domain, no domain used in development."""
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if field not in df or df[field].astype(str).str.strip().eq('').any() or df[field].nunique()<2:
        write_json({'status':'not_performed','reason':f'Insufficient complete {field} domains'},out/'lodo_status.json');return pd.DataFrame()
    all_results=[]
    for domain in sorted(df[field].astype(str).unique()):
        test=df[df[field].astype(str)==domain].copy();rest=df[df[field].astype(str)!=domain].copy()
        if len(test)<minimum_domain_patients or test.idh_label.nunique()<2 or rest.idh_label.nunique()<2:
            print(f'Skipping LODO {field}={domain}: insufficient held-out class coverage/patients');continue
        try:
            tr,va=train_test_split(rest,test_size=cfg.val_fraction/(1-cfg.test_fraction),stratify=rest.idh_label,random_state=cfg.seed)
        except ValueError:
            print('Skipping domain',domain,': insufficient training/validation class counts');continue
        splits={'train':tr,'val':va,'test':test}
        if any(x.idh_label.nunique()!=2 for x in splits.values()):continue
        try:
            model,dest=run_training(splits,cfg,'baseline',out/('holdout_'+str(domain).replace('/','_')))
            m=json.loads((dest/'test_metrics.json').read_text())
            all_results.append({'holdout_domain':domain,'n':len(test),**{k:v for k,v in m.items() if k not in ('confusion_matrix','auroc_bootstrap_ci95')}})
            del model;gc.collect()
        except (ValueError,RuntimeError) as ex:
            print('LODO failed for',domain,str(ex));continue
    result=pd.DataFrame(all_results)
    if not result.empty:result.to_csv(out/'lodo_results.csv',index=False)
    else:write_json({'status':'not_performed','reason':'No eligible domains'},out/'lodo_status.json')
    return result

def evaluate_external(checkpoint,external_df,cfg,out,development_df=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if development_df is not None and set(development_df.patient_id)&set(external_df.patient_id):
        raise ValueError('External cohort patient IDs overlap with development cohort')
    model,trained_cfg,_=load_checkpoint(checkpoint,cfg.device)
    # preprocessing and architecture from frozen checkpoint, ONLY cache/output/device replaced
    trained_cfg.cache_dir=cfg.cache_dir;trained_cfg.device=cfg.device
    dl=torch.utils.data.DataLoader(PatientDataset(external_df,trained_cfg),batch_size=cfg.batch_size,num_workers=cfg.workers)
    p=with_metadata(predict_loader(model,dl,trained_cfg),external_df)
    p.to_csv(out/'external_predictions.csv',index=False)
    m=metrics_binary(p.idh_label,p.probability,trained_cfg.threshold)
    write_json(m,out/'external_metrics.json');plot_eval(p,out)
    return p,m

def plot_training(history_csv,output):
    hist=pd.read_csv(history_csv); fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,k in zip(axes,('loss','auroc')):
        for p in ('train','val'):ax.plot(hist.epoch,hist[f'{p}_{k}'],label=p)
        ax.set(xlabel='Epoch',ylabel=k);ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(output,dpi=170);plt.close(fig)
