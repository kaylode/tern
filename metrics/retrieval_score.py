# from . import TemplateMetric
import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances
import torch
import torch.nn as nn
import torch.utils.data as data
from tqdm import tqdm

try:
    import faiss
    USE_FAISS=True
except ImportError as e:
    USE_FAISS=False

def get_dist_func(name):
    if name == 'cosine':
        return cosine_distances # 1 - similarity
    elif name == 'euclide':
        return euclidean_distances
    else:
        raise NotImplementedError

def get_retrieval_embedding(img_feat, txt_feat, embedding_type):
    if embedding_type == 'img':
        return img_feat
    elif embedding_type == 'txt':
        return txt_feat
    else:
        return np.concatenate([img_feat, txt_feat], axis=-1)

def save_results(query_results):
    if not os.path.exists('./results/'):
        os.mkdir('./results/')
    np.save('./results/query_results.npy', query_results, allow_pickle=True)

def get_top_k(object_dist_score, top_k=5, max_distance=2.0):
    """
    Input: Array of distance of each item in the gallery to the query
    Return: top k objects's indexes and scores
    """
    # Sort item by distance and get top-k
    top_k_indexes = object_dist_score.argsort()[:top_k]
    top_k_scores = object_dist_score[top_k_indexes]

    # Keep only item with near distance
    if max_distance is not None:
        keep_indexes = top_k_scores <= max_distance
        top_k_indexes = top_k_indexes[keep_indexes]
        top_k_scores = top_k_scores[keep_indexes]

    return top_k_indexes, top_k_scores


"""
All retrieval metrics
- Mean average precision
- First tier
- Second tier
- Dice score
- Nearest neighbor
"""

# Modified version of https://github.com/kaylode/3d-mesh-retrieval/blob/74e3a888da825a89e68643c1746e612626d248ef/MeshNet/utils/evaluate_distmat.py#L24
def nearest_neighbor(target_labels, retrieved_labels):
    return int(retrieved_labels[0] in target_labels)

def first_tier(target_labels, retrieved_labels):
    n_relevant_objs = sum([1 if i in target_labels else 0 for i in retrieved_labels])
    retrieved_1st_tier = retrieved_labels[:n_relevant_objs+1]
    return np.mean([1 if i in target_labels else 0 for i in retrieved_1st_tier])

def second_tier(target_labels, retrieved_labels):
    n_relevant_objs = sum([1 if i in target_labels else 0 for i in retrieved_labels])
    retrieved_2nd_tier = retrieved_labels[:2*n_relevant_objs+1]
    return np.mean([1 if i in target_labels else 0 for i in retrieved_2nd_tier])

def mean_average_precision(target_labels, retrieved_labels):
    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(retrieved_labels):
        if p in target_labels and p not in retrieved_labels[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)
    return score

def dice_score(target_labels, retrieved_labels):
    # F1 score: https://www.kaggle.com/cdeotte/part-2-rapids-tfidfvectorizer-cv-0-700?scriptVersionId=0
    n = len(np.intersect1d(target_labels,retrieved_labels)) # Number of corrects
    score = 2*n / (len(target_labels)+len(retrieved_labels))
    return score

metrics_mapping = {
    'FT': first_tier,
    'ST': second_tier,
    'NN': nearest_neighbor,
    'MAP': mean_average_precision,
    'F1': dice_score,
}

class RetrievalScore():    
    def __init__(self, 
            queries_set, 
            gallery_set=None, 
            metric_names=['FT', "ST", "MAP", "NN", "F1"],
            max_distance = 1.3,
            top_k=10,
            save_results=True):

        self.metric_names = metric_names
        self.queries_loader = data.DataLoader(
            queries_set,
            batch_size=256, 
            shuffle = True, 
            collate_fn=queries_set.collate_fn, 
            num_workers= 2,
            pin_memory=True
        )

        self.gallery_loader = data.DataLoader(
            gallery_set,
            batch_size=256, 
            shuffle = True, 
            collate_fn=gallery_set.collate_fn, 
            num_workers= 2,
            pin_memory=True
        ) if gallery_set is not None else None
        
        self.top_k = top_k                  # Query top k candidates
        self.max_distance = max_distance    # Query candidates with distances lower than threshold
        self.save_results = save_results    # Save for vizualization
        self.queries_embedding_type = 'txt'
        self.gallery_embedding_type = 'img'

        self.queries_embedding = [] 
        self.gallery_embedding = []
        self.queries_post_ids = []
        self.gallery_post_ids = []
        self.targets_post_ids = []

        # Distance function
        self.dist_func = get_dist_func('cosine')

        if self.save_results:
            self.post_results_dict = {}

        if USE_FAISS:
            res = faiss.StandardGpuResources()  # use a single GPU
            self.faiss_pool = faiss.IndexFlatL2(1024)
            self.faiss_pool = faiss.index_cpu_to_gpu(res, 0, self.faiss_pool)

    def reset(self):
        self.queries_embedding = [] 
        self.gallery_embedding = []
        self.queries_post_ids = []
        self.gallery_post_ids = []
        self.targets_post_ids = []
        if self.save_results:
            self.post_results_dict = {}

        if USE_FAISS:
            self.faiss_pool.reset()

    def update(self, model):
        self.model = model
        self.model.eval()

    def compute_queries(self):
        for idx, batch in enumerate(tqdm(self.queries_loader)):
            post_ids = batch['ann_ids']
            target_ids = batch['image_ids']
            img_feats, txt_feats = self.model.inference_step(batch)
         
            # Get embedding of each item in batch
            batch_size = img_feats.shape[0]
            for i in range(batch_size):
                feat = get_retrieval_embedding(
                    img_feats[i], 
                    txt_feats[i], 
                    embedding_type=self.queries_embedding_type)

                self.queries_embedding.append(feat)
                self.queries_post_ids.append(post_ids[i])
                self.targets_post_ids.append(target_ids[i])
        self.queries_embedding = np.array(self.queries_embedding)
        self.targets_post_ids = np.array(self.targets_post_ids)
        self.queries_post_ids = np.array(self.queries_post_ids)

    def compute_gallery(self):
        for idx, batch in enumerate(tqdm(self.gallery_loader)):
            post_ids = batch['image_ids']
            img_feats, txt_feats = self.model.inference_step(batch)

            # Get embedding of each item in batch
            batch_size = img_feats.shape[0]
            for i in range(batch_size):
                feat = get_retrieval_embedding(
                    img_feats[i], 
                    txt_feats[i], 
                    embedding_type=self.gallery_embedding_type)

                self.gallery_embedding.append(feat)
                self.gallery_post_ids.append(post_ids[i])
        self.gallery_embedding = np.array(self.gallery_embedding)
        self.gallery_post_ids = np.array(self.gallery_post_ids)

    def compute_default(self):
        """
        Compute score for each metric and return 
        """
        total_scores = {
            i: [] for i in self.metric_names
        }

        # Compute distance matrice for queries and gallery
        print("Calculating distance matrice...")
        dist_mat = self.dist_func(self.queries_embedding, self.gallery_embedding)

        # np.savetxt("./results/dist_mat.txt",dist_mat)
        for idx, row in enumerate(dist_mat):
            object_dist_score = dist_mat[idx]
            top_k_indexes, top_k_scores = get_top_k(
                object_dist_score,
                top_k=self.top_k,
                max_distance=self.max_distance
            )

            current_post_id = self.queries_post_ids[idx] # query caption id
            target_post_ids = self.targets_post_ids[idx] # target image id

            pred_post_ids = self.gallery_post_ids[top_k_indexes] # gallery image id
            pred_post_ids = pred_post_ids.tolist()

            if self.save_results:
                self.post_results_dict[current_post_id] = {
                    'image_ids': pred_post_ids,
                    'target_ids': target_post_ids,
                    'scores': top_k_scores 
                }
            
            for metric_name in self.metric_names:
                metric_fn = metrics_mapping[metric_name]
                score = metric_fn(target_post_ids, pred_post_ids)
                total_scores[metric_name].append(score)

        return total_scores


    def compute_faiss(self):
        """
        Compute score for each metric and return using faiss
        """
        total_scores = {
            i: [] for i in self.metric_names
        }

        self.faiss_pool.add(self.gallery_embedding)
        top_k_scores_all, top_k_indexes_all = self.faiss_pool.search(self.queries_embedding, k=self.top_k)
        
        
        for idx, (top_k_scores, top_k_indexes) in enumerate(zip(top_k_scores_all, top_k_indexes_all)):
          
            current_post_id = self.queries_post_ids[idx]
            target_post_ids = self.targets_post_ids[idx]

            pred_post_ids = self.gallery_post_ids[top_k_indexes]
            pred_post_ids = pred_post_ids.tolist()

            if self.save_results:
                self.post_results_dict[current_post_id] = {
                    'image_ids': pred_post_ids,
                    'target_ids': target_post_ids,
                    'scores': top_k_scores 
                }
            
            for metric_name in self.metric_names:
                metric_fn = metrics_mapping[metric_name]
                score = metric_fn([target_post_ids], pred_post_ids)
                total_scores[metric_name].append(score)

        return total_scores

    def compute(self):
        print("Extracting features...")
        with torch.no_grad():
            self.compute_queries()
            if self.gallery_loader is not None:
                self.compute_gallery()
            else:
                self.gallery_embedding = self.queries_embedding.copy()
                self.gallery_post_ids = self.queries_post_ids.copy()
                
        if USE_FAISS:
            total_scores = self.compute_faiss()
        else:
            total_scores = self.compute_default()

        # Save results for visualization later
        if self.save_results:
            print("Saving retrieval results...")
            save_results(self.post_results_dict)
          
        result_dict={
            k:np.mean(v) for k,v in total_scores.items()
        }

        return result_dict
            
    def value(self):
        result_dict = self.compute()

        result_dict = {
            k:np.round(float(v), 5) for k,v in result_dict.items()
        }

        return result_dict

    def __str__(self):
        return str(self.value())