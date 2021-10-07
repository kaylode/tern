import os
import numpy as np
from tqdm import tqdm

import torch
import torch.utils.data as data
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances

from .f1 import F1Metric
from .map import MAPMetric
from .recall import RecallMetric
from .neighbor import NearestNeighborMetric
from .tier import FirstTierMetric, SecondTierMetric

# Use Faiss for faster retrieval: https://github.com/facebookresearch/faiss
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

metrics_mapping = {
    'FT': FirstTierMetric(),
    'ST': SecondTierMetric(),
    'NN': NearestNeighborMetric(),
    'MAP@10': MAPMetric(k=10),
    'F1@10': F1Metric(k=10),
    'R@1': RecallMetric(k=10),
    'R@5': RecallMetric(k=5),
    'R@10': RecallMetric(k=10),
}

class RetrievalScore():    
    def __init__(self, 
            queries_set, 
            gallery_set=None, 
            dimension=1024,
            metric_names=['FT', "ST", "MAP", "NN", "F1", "R@10"],
            max_distance = None,
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
        self.queries_ids = []
        self.gallery_ids = []
        self.targets_ids = []

        # Distance function
        self.dist_func = get_dist_func('cosine')

        if self.save_results:
            self.results_dict = {}

        if USE_FAISS:
            res = faiss.StandardGpuResources()  # use a single GPU
            self.faiss_pool = faiss.IndexFlatL2(dimension)
            self.faiss_pool = faiss.index_cpu_to_gpu(res, 0, self.faiss_pool)

    def reset(self):
        self.queries_embedding = [] 
        self.gallery_embedding = []
        self.queries_ids = []
        self.gallery_ids = []
        self.targets_ids = []
        if self.save_results:
            self.results_dict = {}

        if USE_FAISS:
            self.faiss_pool.reset()

    def update(self, model):
        self.model = model
        self.model.eval()

    def compute_queries(self):
        for idx, batch in enumerate(tqdm(self.queries_loader)):
            ids = batch['ann_ids']
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
                self.queries_ids.append(ids[i])
                self.targets_ids.append(target_ids[i])
        self.queries_embedding = np.array(self.queries_embedding)
        self.targets_ids = np.array(self.targets_ids)
        self.queries_ids = np.array(self.queries_ids)

    def compute_gallery(self):
        for idx, batch in enumerate(tqdm(self.gallery_loader)):
            ids = batch['image_ids']
            img_feats, txt_feats = self.model.inference_step(batch)

            # Get embedding of each item in batch
            batch_size = img_feats.shape[0]
            for i in range(batch_size):
                feat = get_retrieval_embedding(
                    img_feats[i], 
                    txt_feats[i], 
                    embedding_type=self.gallery_embedding_type)

                self.gallery_embedding.append(feat)
                self.gallery_ids.append(ids[i])
        self.gallery_embedding = np.array(self.gallery_embedding)
        self.gallery_ids = np.array(self.gallery_ids)

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

            current_id = self.queries_ids[idx] # query caption id
            target_ids = self.targets_ids[idx] # target image id

            pred_ids = self.gallery_ids[top_k_indexes] # gallery image id
            pred_ids = pred_ids.tolist()

            if self.save_results:
                self.results_dict[current_id] = {
                    'image_ids': pred_ids,
                    'target_ids': target_ids,
                    'scores': top_k_scores 
                }
            
            for metric_name in self.metric_names:
                metric_fn = metrics_mapping[metric_name]
                score = metric_fn(target_ids, pred_ids)
                total_scores[metric_name].append(score)

        return total_scores


    def compute_faiss(self):
        """
        Compute score for each metric and return using faiss
        """

        self.faiss_pool.add(self.gallery_embedding)
        top_k_scores_all, top_k_indexes_all = self.faiss_pool.search(self.queries_embedding, k=self.top_k)
        
        
        for idx, (top_k_scores, top_k_indexes) in enumerate(zip(top_k_scores_all, top_k_indexes_all)):
          
            current_id = self.queries_ids[idx]
            target_ids = self.targets_ids[idx]

            pred_ids = self.gallery_ids[top_k_indexes]
            pred_ids = pred_ids.tolist()

            if self.save_results:
                self.results_dict[current_id] = {
                    'image_ids': pred_ids,
                    'target_ids': target_ids,
                    'scores': top_k_scores 
                }
            
            for metric_name in self.metric_names:
                metric_fn = metrics_mapping[metric_name]
                metric_fn.update(pred_ids, [target_ids])
                

    def compute(self):
        print("Extracting features...")
        with torch.no_grad():
            self.compute_queries()
            if self.gallery_loader is not None:
                self.compute_gallery()
            else:
                self.gallery_embedding = self.queries_embedding.copy()
                self.gallery_ids = self.queries_ids.copy()
                
        if USE_FAISS:
            self.compute_faiss()
        else:
            total_scores = self.compute_default()

        # Save results for visualization later
        if self.save_results:
            print("Saving retrieval results...")
            save_results(self.results_dict)
        
        result_dict = {}
        for metric_name in self.metric_names:
            metric_fn = metrics_mapping[metric_name]
            result_dict.update(metric_fn.value())
        
        return result_dict
            
    def value(self):
        result_dict = self.compute()
        return result_dict

    def __str__(self):
        return str(self.value())