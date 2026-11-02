#!/bin/bash
set -e
cd ~/Repos/idh-mcnet
source .venv/bin/activate
export nnUNet_raw=$PWD/nnUNet_raw nnUNet_preprocessed=$PWD/nnUNet_preprocessed nnUNet_results=$PWD/nnUNet_results nnUNet_compile=false
echo "START test60 $(date)"
nnUNetv2_predict -i mcnet_eval/test60_in -o mcnet_eval/test60_preds -d Dataset002_IDHMC -c 2d -f 0 -tr nnUNetTrainer_250epochs_save5 -p nnUNetPlans -chk checkpoint_best.pth -npp 3 -nps 3
echo "START ext143 $(date)"
nnUNetv2_predict -i mcnet_eval/ext143_in -o mcnet_eval/ext143_preds -d Dataset002_IDHMC -c 2d -f 0 -tr nnUNetTrainer_250epochs_save5 -p nnUNetPlans -chk checkpoint_best.pth -npp 3 -nps 3
echo "START dcf test60 $(date)"
python mcnet_eval/dcf_eval_pool.py test mcnet_eval/test60_preds mcnet_eval/dcf_test60.json
echo "START dcf ext143 $(date)"
python mcnet_eval/dcf_eval_pool.py ext143 mcnet_eval/ext143_preds mcnet_eval/dcf_ext143.json
echo "ALL-DONE $(date)"
