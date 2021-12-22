import torch
import numpy as np
import gdown
from PIL import Image
import matplotlib.pyplot as plt 

def download_weights(id_or_url, cached=None, md5=None, quiet=False):
    if id_or_url.startswith('http'):
        url = id_or_url
    else:
        url = 'https://drive.google.com/uc?id={}'.format(id_or_url)

    return gdown.cached_download(url=url, path=cached, md5=md5, quiet=quiet)


weight_url = {}

def download_pretrained_weights(name, cached=None):
    return download_weights(weight_url[name], cached)
    
def draw_image_caption(image, text, image_name=None, figsize=(10,10)):

    plt.close('all')
    fig = plt.figure(figsize=figsize)

    if isinstance(image, torch.Tensor):
        image = image.numpy().squeeze().transpose((1,2,0))

    # Display the image
    plt.imshow(image)
    plt.axis('off')
    fig.text(.5, .05, text, ha='center')

    if image_name:
        plt.savefig(image_name, bbox_inches='tight')

    return fig

def draw_retrieval_results(query, top_k_relevant, gt_path=None, save_filename=None, figsize=(10,10)):
    plt.close('all')
    fig=plt.figure(figsize=figsize)

    columns = len(top_k_relevant) + 1
    for i, (image, score) in enumerate(top_k_relevant):
        img = Image.open(image)
        fig.add_subplot(1, columns, i+1)
        plt.imshow(img)
        plt.title(str(score))
        plt.tight_layout()
        plt.axis('off')


    # Plot ground truth
    img = Image.open(gt_path)
    fig.add_subplot(1, columns, columns)
    plt.imshow(img)
    plt.title('Ground Truth')
    plt.tight_layout()
    plt.axis('off')
    

    if save_filename is not None:
        plt.savefig(save_filename)

    fig.suptitle(query)
    return fig
