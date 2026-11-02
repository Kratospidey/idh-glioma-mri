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
assert TRIALS,'Enable at least one training experiment in Section 3'
for i,(name,variant,change) in enumerate(TRIALS,1):
    dest=OUTPUT/'experiments'/name
    if (dest/'best.pt').is_file() and (dest/'val_predictions.csv').is_file():
        print('REUSING COMPLETED EXPERIMENT:',name); continue
    print(f'[{i}/{len(TRIALS)}] STARTING:',name)
    trial=replace(cfg,**change,seed=SEED+i)
    model,_=run_training(parts,trial,experiment=variant,experiment_dir=dest,evaluate_test=False)
    del model; gc.collect(); torch.cuda.empty_cache()
    if (dest/'val_predictions.csv').exists():
        (dest/'last.pt').unlink(missing_ok=True) # keep best.pt; save Drive space
print('All selected training runs complete. Final test still locked.')
