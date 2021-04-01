import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch

from .extractors import _get_cnn_basemodel, _get_bert_basemodel

def get_model(args, config):
    net = Extractor(
        cnn_base_model=config.image_extractor,
        bert_base_model=config.text_extractor,
        out_dim=512,
        freeze_layers=[0,1,2,3,4,5],
        freeze_cnn=args.freeze_cnn)
    
    net = nn.DataParallel(net)
    
    return net

class Extractor(nn.Module):
    def __init__(self, cnn_base_model, bert_base_model, out_dim, freeze_layers=None, freeze_cnn=False):
        super(Extractor, self).__init__()    
        
        # Text extractor
        self.text_extractor = _get_bert_basemodel(bert_base_model,freeze_layers)
        self.image_extractor = _get_cnn_basemodel(cnn_base_model)

        if freeze_cnn:
            self.image_extractor.freeze()

        self.embedding_imgs = nn.Sequential(
            nn.Linear(self.image_extractor.feature_dim, self.image_extractor.feature_dim),
            nn.ReLU(),
            nn.Linear(self.image_extractor.feature_dim, out_dim))

        self.embedding_text = nn.Sequential(
            nn.Linear(self.text_extractor.feature_dim, self.text_extractor.feature_dim),
            nn.ReLU(),
            nn.Linear(self.text_extractor.feature_dim, out_dim))

    def image_encoder(self, img):
        x = self.image_extractor(img)
        x = self.embedding_imgs(x)
        return x

    def text_encoder(self, encoded_inputs):
        x = self.text_extractor(encoded_inputs)
        x = self.embedding_text(x)
        return x

    def forward(self, batch, device):
        image = batch['imgs'].to(device)
        encoded_text = batch['txts']
        encoded_text = {k:v.to(device) for k,v in encoded_text.items()}
        img_feats = self.image_encoder(image)
        txt_feats = self.text_encoder(encoded_text)

        return img_feats, txt_feats