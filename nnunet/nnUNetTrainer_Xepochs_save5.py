from nnunetv2.training.nnUNetTrainer.variants.training_length.nnUNetTrainer_Xepochs import nnUNetTrainer_250epochs


class nnUNetTrainer_250epochs_save5(nnUNetTrainer_250epochs):
    def __init__(self, plans, configuration, fold, dataset_json, device):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.save_every = 5
