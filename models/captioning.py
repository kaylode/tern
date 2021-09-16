from .base_model import BaseModel

import sys
sys.path.append('..')

class Captioning(BaseModel):
    def __init__(self, model, **kwargs):
        super(Captioning, self).__init__(**kwargs)
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
        
        src_inputs = batch['feats'].to(self.device)
        loc_src_inputs = batch['loc_feats'].to(self.device)
        lang_src_inputs = batch['lang_feats'].to(self.device)

        outputs_l, outputs_v = self.model(
            visual_inputs=src_inputs, 
            spatial_inputs=loc_src_inputs, 
            lang_inputs=lang_src_inputs)

        loss = self.criterion(outputs_l, outputs_v)

        loss_dict = {'T': loss.item()}
        return loss, loss_dict

    def inference_step(self, batch):
        
        src_inputs = batch['feats'].to(self.device)
        loc_src_inputs = batch['loc_feats'].to(self.device)
        lang_src_inputs = batch['lang_feats'].to(self.device)

        outputs_l, outputs_v = self.model(
            visual_inputs=src_inputs, 
            spatial_inputs=loc_src_inputs, 
            lang_inputs=lang_src_inputs)

        return outputs_v.cpu().detach().numpy(), outputs_l.cpu().detach().numpy()

    def evaluate_step(self, batch):

        src_inputs = batch['feats'].to(self.device)
        loc_src_inputs = batch['loc_feats'].to(self.device)
        lang_src_inputs = batch['lang_feats'].to(self.device)

        outputs_l, outputs_v = self.model(
            visual_inputs=src_inputs, 
            spatial_inputs=loc_src_inputs, 
            lang_inputs=lang_src_inputs)

        loss = self.criterion(outputs_l, outputs_v)

        loss_dict = {'T': loss.item()}

        self.update_metrics(model=self)
        return loss, loss_dict

