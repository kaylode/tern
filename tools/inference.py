import os
import cv2
import torch
import faiss
import argparse
import numpy as np
from tqdm import tqdm
from .configs import Config
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from modules.models import EncoderBERT, TERN
from datasets.utils import make_feature_batch


parser = argparse.ArgumentParser('Retrieve Text-To-Images')
parser.add_argument('--weight', type=str, help='weight path file')
parser.add_argument('--image_dir', type=str, help='path to image directory')
parser.add_argument('--ann_path', type=str, help='path to annotation file')
parser.add_argument('--feat_dir', type=str, help='path to bottom up numpy feature directory ')

def make_gallery_embeddings(ann_path, feat_dir, model, device):

    coco = COCO(ann_path)
    imgIds = coco.getImgIds()
    img_infos = coco.loadImgs(imgIds)

    gallery_embeddings = []
    npy_paths = []
    npy_loc_paths = []

    for info in img_infos:
        image_name = info['file_name']
        npy_name = image_name[:-4]+'.npz'

        npy_path = os.path.join(feat_dir, 'data_att', npy_name)
        npy_loc_path = os.path.join(feat_dir, 'data_box', npy_name)

        npy_loc_paths.append(npy_loc_path)
        npy_paths.append(npy_path)

    batch_feats = []
    batch_loc_feats = []
    for idx, (npy_path, npy_loc_path) in enumerate(tqdm(zip(npy_paths, npy_loc_paths))):
        npy_feat = np.load(npy_path, mmap_mode='r')['feat']
        npy_loc_feat = np.load(npy_loc_path, mmap_mode='r')['feat']

        feats = torch.from_numpy(npy_feat).float()
        loc_feats = torch.from_numpy(npy_loc_feat).float()

        batch_feats.append(feats)
        batch_loc_feats.append(loc_feats)

        if len(batch_feats) == 32 or idx == len(npy_paths) - 1:

            feats = torch.stack(batch_feats).to(device)
            loc_feats = torch.stack(batch_loc_feats).to(device)

            with torch.no_grad():
                embedding = model.visual_forward({
                    "feats": feats, 
                    "loc_feats": loc_feats}, 
                    device)
            embedding = embedding.cpu().detach().numpy()
            gallery_embeddings.append(embedding)

            batch_feats = []
            batch_loc_feats = []

    gallery_embeddings = np.concatenate(gallery_embeddings, axis=0)
    return img_infos, gallery_embeddings

def make_query_embeddings(text, model, device):
    with torch.no_grad():
        feat = model.lang_forward({
            'texts': text
        }, device)

    feat = feat.cpu().numpy()
    return feat

def faiss_search(faiss_pool, query_embedding, gallery_embedding, top_k=10):
    faiss_pool.reset()
    faiss_pool.add(gallery_embedding)
    top_k_scores, top_k_indexes = faiss_pool.search(query_embedding, k=top_k)
    return top_k_indexes, top_k_scores

def show_retrieval(image_dir, infos, top_k_indexes, top_k_scores):
    fig = plt.figure(figsize=(30,10))
    for i, (index, score) in enumerate(zip(top_k_indexes[0], top_k_scores[0])):
        image_info = infos[index]
        image_path = os.path.join(image_dir, image_info['file_name'])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        fig.add_subplot(2, 5, i+1)
        plt.imshow(image)
        plt.title(score)
        plt.axis('off')
    
    plt.show()

def main(args, config):
    device = torch.device('cuda')

    config.model.update({'precomp_bert': False})

    model = TERN(config.model)

    model.load_state_dict(torch.load(args.weight)['model'])
    model = model.to(device)
    model.eval()

    res = faiss.StandardGpuResources()  # use a single GPU
    faiss_pool = faiss.IndexFlatIP(config.model['args']['d_embed'])
    faiss_pool = faiss.index_cpu_to_gpu(res, 0, faiss_pool)

    query_embeddings = make_query_embeddings(args.text, model, device)
    img_infos, gallery_embeddings = make_gallery_embeddings(args.ann_path, args.feat_dir, model, device)

    top_k_indexes, top_k_scores = faiss_search(faiss_pool, query_embeddings, gallery_embeddings, top_k=10)
    show_retrieval(args.image_dir, img_infos, top_k_indexes, top_k_scores)

if __name__ == '__main__':
    
    args = parser.parse_args()
    config = Config("./tools/configs/yaml/tern.yaml")

    main(args, config)