from .base_model import BaseModel
import torch
import torchvision.models as models
import torch.nn as nn
from tqdm import tqdm


import sys
sys.path.append('..')

class Retrieval(BaseModel):
    def __init__(self, model, **kwargs):
        super(Retrieval, self).__init__(**kwargs)
        self.model = model
        self.model_name = "efficient_bert"
        if self.optimizer is not None:
            self.optimizer = self.optimizer(self.parameters(), lr= self.lr)
            self.set_optimizer_params()

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

        self.update_metrics(model=self)
        #self.update_metrics(outputs = outputs, targets = targets)

        return loss, loss_dict

class Retrieval2(BaseModel):
    def __init__(self, model, **kwargs):
        super(Retrieval2, self).__init__(**kwargs)
        self.model = model
        self.model_name = "efficientnet_b2"
        if self.optimizer is not None:
            self.optimizer = self.optimizer(self.parameters(), lr= self.lr)
            self.set_optimizer_params()

        if self.freeze:
            for params in self.model.parameters():
                params.requires_grad = False

        if self.device:
            self.model.to(self.device)
        
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch):
        feats = self.model(batch, self.device)
        lbls = batch['lbls'].to(self.device)
        output = self.criterion(feats, lbls)

        loss_dict = {k:v.item() for k,v in output.items()}
        loss = output['T']
        return loss, loss_dict

    def inference_step(self, batch):
        feats = self.model(batch, self.device)
        feats = feats.detach().cpu().numpy()
        return feats, feats.copy() # For evaluation metric requires 2 features 

    def evaluate_step(self, batch):
        feats = self.model(batch, self.device)
        
        lbls = batch['lbls'].to(self.device)
        output = self.criterion(feats, lbls)

        loss_dict = {k:v.item() for k,v in output.items()}
        loss = output['T']

        self.update_metrics(model=self)
        #self.update_metrics(outputs = outputs, targets = targets)

        return loss, loss_dict