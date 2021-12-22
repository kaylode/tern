import torch
import torch.nn as nn
import os
from datetime import datetime
from tools.configs import config_from_dict

class Checkpoint():
    """
    Checkpoint for saving model state
    :param save_per_epoch: (int)
    :param path: (string)
    """
    def __init__(self, save_per_iter = 1000, path = None):
        self.path = path
        self.save_per_iter = save_per_iter
        # Create folder
        if self.path is None:
            self.path = os.path.join('weights',datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        
    def save(self, model, save_mode='last', **kwargs):
        """
        Save model and optimizer weights
        :param model: Pytorch model with state dict
        """
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        model_path = "_".join([model.model_name,save_mode])
        
        epoch = int(kwargs['epoch']) if 'epoch' in kwargs else 0
        iters = int(kwargs['iters']) if 'iters' in kwargs else 0
        best_value = float(kwargs['best_value']) if 'best_value' in kwargs else 0
        class_names = kwargs['class_names'] if 'class_names' in kwargs else None
        config = kwargs['config'] if 'config' in kwargs else None
        config_dict = config.to_dict()
        weights = {
            'model': model.model.state_dict(),
            'optimizer': model.optimizer.state_dict(),
            'epoch': epoch,
            'iters': iters,
            'best_value': best_value,
            'class_names': class_names,
            'config': config_dict,
        }

        if model.scaler is not None:
            weights[model.scaler.state_dict_key] = model.scaler.state_dict()

        torch.save(weights, os.path.join(self.path,model_path)+".pth")
    
def load_checkpoint(model, path):
    """
    Load trained model checkpoint
    :param model: (nn.Module)
    :param path: (string) checkpoint path
    """
    state = torch.load(path)
    current_lr = None
    if model.optimizer is not None:
        for param_group in model.optimizer.param_groups:
            if 'lr' in param_group.keys():
                current_lr = param_group['lr']
                break

    try:
        model.model.load_state_dict(state["model"])
        if model.optimizer is not None:
            model.optimizer.load_state_dict(state["optimizer"])
        if model.scaler is not None:
            model.scaler.load_state_dict(state[model.scaler.state_dict_key])
    except RuntimeError as e:
        try:
            ret = model.model.load_state_dict(state["model"], strict=False)
        except RuntimeError as e:
            print(f'[Warning] Ignoring {e}')
            print('Load pretrained weights')

    if current_lr is not None and model.optimizer is not None:
        for param_group in model.optimizer.param_groups:
            param_group['lr'] = current_lr
        print(f'Set learning rate to {current_lr}')
    print("Loaded Successfully!")

def get_epoch_iters(path):
    """
    Get epoch and iter from weight path
    """
    state = torch.load(path)
    epoch_idx = int(state['epoch']) if 'epoch' in state.keys() else 0
    iter_idx = int(state['iters']) if 'iters' in state.keys() else 0
    best_value = float(state['best_value']) if 'best_value' in state.keys() else 0.0

    return epoch_idx, iter_idx, best_value

def get_config(path, ignore_keys=[]):
    """
    Load config from weight
    """
    state = torch.load(path)
    config_dict = state['config'] if 'config' in state.keys() else None
    if config_dict is None:
        return None
    config = config_from_dict(config_dict, ignore_keys)
    return config