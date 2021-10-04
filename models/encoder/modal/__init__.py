from .tern import TERN

def get_cross_modal(config):
    if config.name == 'tern':
        return TERN(config, precomp_bert=True)