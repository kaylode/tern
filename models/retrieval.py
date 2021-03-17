from .base_model import BaseModel
import torch
import torchvision.models as models
import torch.nn as nn
from tqdm import tqdm


import sys
sys.path.append('..')

class Retrieval(BaseModel):
    def __init__(self, model, n_classes, **kwargs):
        super(Retrieval, self).__init__(**kwargs)
        self.model = model
        self.model_name = self.model.name
        if self.optimizer is not None:
            self.optimizer = self.optimizer(self.parameters(), lr= self.lr)
            self.set_optimizer_params()
        self.n_classes = n_classes

        if self.freeze:
            for params in self.model.parameters():
                params.requires_grad = False

        if self.device:
            self.model.to(self.device)
        
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch):
        img_feats, txt_feats = self.model(batch, self.device)
        output = self.criterion(img_feats, txt_feats)

        loss_dict = {k:v.item() for k,v in output.items()}
        loss = output['T']
        return loss, loss_dict

    def inference_step(self, batch):
        img_feats, txt_feats = self.model(batch, self.device)
        img_feats = img_feats.detach().cpu().numpy()
        txt_feats = txt_feats.detach().cpu().numpy()
        return img_feats, txt_feats  

    def evaluate_step(self, batch):
        img_feats, txt_feats = self.model(batch, self.device)
        output = self.criterion(img_feats, txt_feats)

        loss_dict = {k:v.item() for k,v in output.items()}
        loss = output['T']

        #self.update_metrics(model=self)
        #self.update_metrics(outputs = outputs, targets = targets)

        return loss, loss_dict

    