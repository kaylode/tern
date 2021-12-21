import numpy as np
from .base_model import BaseModel

import sys
sys.path.append('..')

class Retriever(BaseModel):
    def __init__(self, model, **kwargs):
        super(Retriever, self).__init__(**kwargs)
        self.model = model
        self.model_name = self.model.name
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
        outputs_1, outputs_2 = self.model.forward(batch, self.device)
        loss = self.criterion(outputs_1, outputs_2)
        loss_dict = {k:v.item() for k,v in loss.items()}
        return loss['T'], loss_dict

    def inference_step(self, batch):
        outputs_1, outputs_2 = self.model.forward(batch, self.device)
        return outputs_1.cpu().detach().numpy(), outputs_2.cpu().detach().numpy()

    def evaluate_step(self, batch):
        outputs_1, outputs_2 = self.model.forward(batch, self.device)
        loss = self.criterion(outputs_1, outputs_2)
        loss_dict = {k:v.item() for k,v in loss.items()}

        self.update_metrics(model=self)
        return loss['T'], loss_dict

    def get_visual_embeddings(self, batch):
        outputs_v = self.model.visual_forward(batch, self.device)
        return outputs_v.cpu().detach().numpy().astype(np.float32)

    def get_lang_embeddings(self, batch):
        outputs_l = self.model.lang_forward(batch, self.device)
        return outputs_l.cpu().detach().numpy().astype(np.float32)