import numpy as np
import torch
import os

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, save_path='./', patience=7, verbose=False, delta=0):
        """
        Args:
            save_path : 模型保存文件夹
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.save_path = save_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.delta = delta

        self.vacc_mx = 0.0
        self.vlss_mn = np.inf
        self.early_stop = False
        
        
    def __call__(self, val_loss, val_acc, model):
        
        if val_acc >= self.vacc_mx or val_loss <= self.vlss_mn:
            if val_acc >= self.vacc_mx and val_loss <= self.vlss_mn:
                self.save_checkpoint(model)
            
            self.vlss_mn = min(self.vlss_mn,val_loss)
            self.vacc_mx = max(self.vacc_mx,val_acc)
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, model):
        '''Saves model when validation loss decrease.'''
        path = os.path.join(self.save_path, 'best_model.pth')
        torch.save(model.state_dict(), path)	

