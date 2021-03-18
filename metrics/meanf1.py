# from . import TemplateMetric
import os
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances
import torch
import torch.nn as nn
from tqdm import tqdm

# Source: https://en.wikipedia.org/wiki/F-score
"""
F1 = TP / (TP + 1/2 (FP + FN))
"""

def get_dist_func(name):
    if name == 'cosine':
        return cosine_distances
    elif name == 'euclide':
        return euclidean_distances
    else:
        raise NotImplementedError

class MeanF1Score():
    def __init__(self, queries_loader, gallery_loader, top_k=10):
        self.queries_loader = queries_loader
        self.gallery_loader = gallery_loader
        self.queries = []
        self.gallery = []
        self.lbl_queries = []
        self.lbl_gallery = []
        self.query_post_ids = []
        self.gallery_post_ids = []
        self.lbl_gallery_count_dict = {}
        self.dist_func = get_dist_func('cosine')
        self.score_results = []
        self.top_k_results = []
        self.top_k_results_post_ids = {}
        self.top_k = top_k 
    
    def reset(self):
        self.model = None
        self.queries = []
        self.gallery = []
        self.lbl_queries = []
        self.lbl_gallery = []
        self.query_post_ids = []
        self.gallery_post_ids = []
        self.top_k_results_post_ids = {}
        self.top_k_results = []
        self.score_results = []

    def update(self, model):
        self.model = model
        self.model.eval()

    def compute_gallery(self):
        with torch.no_grad():
            with tqdm(total=len(self.gallery_loader)) as pbar:
                for idx, batch in enumerate(self.gallery_loader):
                    labels = batch['lbls']
                    post_id = batch['post_ids']
                    img_feats, txt_feats = self.model.inference_step(batch)
                    
                    batch_size = img_feats.shape[0]
                    for i in range(batch_size):
                        concat_feats = np.concatenate([img_feats[i], txt_feats[i]], axis=-1)
                        self.gallery.append(concat_feats)
                        self.lbl_gallery.append(labels[i])
                        self.gallery_post_ids.append(post_id[i])
                        if labels[i] not in self.lbl_gallery_count_dict.keys():
                            self.lbl_gallery_count_dict[labels[i]] = 1
                        else:
                            self.lbl_gallery_count_dict[labels[i]] += 1
                    pbar.update(1)

    def compute_queries(self):
        with torch.no_grad():
            with tqdm(total=len(self.gallery_loader)) as pbar:
                for idx, batch in enumerate(self.queries_loader):
                    labels = batch['lbls']
                    post_ids = batch['post_ids']
                    img_feats, txt_feats = self.model.inference_step(batch)
                    
                    batch_size = img_feats.shape[0]
                    for i in range(batch_size):
                        concat_feats = np.concatenate([img_feats[i], txt_feats[i]], axis=-1)
                        self.queries.append(concat_feats)
                        self.lbl_queries.append(labels[i])
                        self.query_post_ids.append(post_ids[i])

                    pbar.update(1)

    def compute(self):
        self.compute_gallery()
        self.compute_queries()
        dist_mat = self.dist_func(self.queries, self.gallery)

        for idx, row in enumerate(dist_mat):
            object_dist_score = dist_mat[idx]
            top_k_indexes = object_dist_score.argsort()[:self.top_k]
            top_k_scores = object_dist_score[top_k_indexes]
            self.top_k_results.append({
                'indexes': top_k_indexes,
                'scores': top_k_scores
            })
        
        # Compute F1 score for each row
        for idx, item in enumerate(self.top_k_results):
            top_k_indexes = self.top_k_results[idx]['indexes']
            top_k_scores = self.top_k_results[idx]['scores']
            query_obj_lbl = self.lbl_queries[idx]
            TP = 0
            FP = 0
            FN = 0
            top_k_post_ids = {
                'indexes': [],
                'scores': []
            }
            for retrieved_obj_idx, retrieved_obj_score in zip(top_k_indexes,top_k_scores):
                retrieved_obj_lbl = self.lbl_gallery[retrieved_obj_idx]
                top_k_post_ids['indexes'].append(self.gallery_post_ids[retrieved_obj_idx])
                top_k_post_ids['scores'].append(retrieved_obj_score)
                
                
                if retrieved_obj_lbl == query_obj_lbl:
                    TP += 1
                else:
                    FN += 1

            self.top_k_results_post_ids[self.query_post_ids[idx]] = top_k_post_ids

            # False positive: all grouth truth that have same label with query
            # FP = total objects of that class - total retrieved objects of that class
            FP = self.lbl_gallery_count_dict[query_obj_lbl] - TP

            f1_score = TP / (TP + 0.5 * (FP + FN))

            self.score_results.append(f1_score)

        
        return np.mean(self.score_results)


    def value(self):
        result = self.compute()
        if not os.path.exists('./results'):
            os.mkdir('./results')
        np.save('./results/query_results.txt', np.array(self.top_k_results_post_ids))
        return {
            'mean-f1': np.round(float(result), 3)
        }

    def __str__(self):
        return f'Mean F1: {self.value()}'

    def __len__(self):
        return len(self.queries_loader)
        
if __name__ == '__main__':
    from utils.getter import *
    config = Config('/home/pmkhoi/source/shopee-matching/configs/config.yaml')
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu') 
    trainset, valset, trainloader, valloader = get_dataset_and_dataloader(config)
    net = get_model(None, config)
    model = Retrieval(
            model = net,
            device = device)


    metric = MeanF1Score(valloader, trainloader)
    metric.update(model)
    metric.compute()
    