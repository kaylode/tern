from .tern import TERN

def get_cross_modal(config):
    if config['name'] == 'TERN':
        return TERN(config, precomp_bert=True)