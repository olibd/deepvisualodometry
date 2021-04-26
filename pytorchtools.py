import os
import pathlib
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer

from Common.Helpers import cuda_is_available

"""
Code modified from https://github.com/Bjarten/early-stopping-pytorch
"""
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, destination_path="checkpoint"):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.destination_path = destination_path

    def __call__(self, val_loss, model, optimizer):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer)
            self.counter = 0

    def save_checkpoint(self, val_loss, model: nn.Module, optimizer: Optimizer):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

        if not os.path.isdir(self.destination_path):
            pathlib.Path(self.destination_path).mkdir(parents=True, exist_ok=True)

        torch.save(model.state_dict(), self.destination_path + "_model.checkpoint")
        torch.save(optimizer.state_dict(), self.destination_path + "_optimizer.checkpoint")
        self.val_loss_min = val_loss

    def load_model_checkpoint(self) -> Any:
        device = "cpu"
        if cuda_is_available():
            device = "cuda"
        return torch.load(self.destination_path + "_model.checkpoint", map_location=torch.device(device))

    def load_optimizer_checkpoint(self) -> Any:
        device = "cpu"
        if cuda_is_available():
            device = "cuda"
        return torch.load(self.destination_path + "_optimizer.checkpoint", map_location=torch.device(device))
