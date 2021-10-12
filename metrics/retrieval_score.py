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

def save_results(query_results, outname):
    if not os.path.exists('./results/'):
        os.mkdir('./results/')
    np.save(f'./results/{outname}.npy', query_results, allow_pickle=True)

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
    'R@1': RecallMetric(k=1),
    'R@5': RecallMetric(k=5),
    'R@10': RecallMetric(k=10),
}

class RetrievalScore():    
    def __init__(self, 
            image_set, 
            text_set, 
            dimension=1024,
            metric_names=['FT', "ST", "MAP", "NN", "F1", "R@10"],
            max_distance = None,
            top_k=10,
            save_results=True):

        self.metric_names = metric_names
        self.image_loader = data.DataLoader(
            image_set,
            batch_size=256, 
            shuffle = False, 
            collate_fn=image_set.collate_fn, 
            num_workers= 2,
            pin_memory=True
        )

        self.text_loader = data.DataLoader(
            text_set,
            batch_size=256, 
            shuffle = False, 
            collate_fn=text_set.collate_fn, 
            num_workers= 2,
            pin_memory=True
        )
        
        self.top_k = top_k                  # Query top k candidates
        self.max_distance = max_distance    # Query candidates with distances lower than threshold
        self.save_results = save_results    # Save for vizualization

        self.image_embedding = [] 
        self.text_embedding = []

        # For image-to-text retrieval
        self.image_ids = []
        self.text_target_ids = []

        # For text-to-image retrieval
        self.text_ids = []
        self.image_target_ids = []

        if self.save_results:
            self.results_dict = {}

        if USE_FAISS:
            res = faiss.StandardGpuResources()  # use a single GPU
            self.faiss_pool = faiss.IndexFlatIP(dimension)
            self.faiss_pool = faiss.index_cpu_to_gpu(res, 0, self.faiss_pool)
        else:
            # Distance function
            self.dist_func = get_dist_func('cosine')

    def reset(self):
        self.image_embedding = [] 
        self.text_embedding = []
        
        # For image-to-text retrieval
        self.image_ids = []
        self.text_target_ids = []

        # For text-to-image retrieval
        self.text_ids = []
        self.image_target_ids = []

        if self.save_results:
            self.results_dict = {}

        if USE_FAISS:
            self.faiss_pool.reset()

        for metric_name in self.metric_names:
            metric_fn = metrics_mapping[metric_name]
            metric_fn.reset()        

    def update(self, model):
        self.model = model
        self.model.eval()

    def compute_images(self):
        for idx, batch in enumerate(tqdm(self.image_loader)):
            image_ids = batch['ids']
            text_target_ids = batch['text_ids']
            feats = self.model.get_visual_embeddings(batch)
         
            # Get embedding of each item in batch
            batch_size = feats.shape[0]
            for i in range(batch_size):
                feat = feats[i]
                self.image_embedding.append(feat)
                self.image_ids.append(image_ids[i])
                self.text_target_ids.append(text_target_ids[i])
                
        self.image_embedding = np.array(self.image_embedding)
        self.text_target_ids = np.array(self.text_target_ids)
        self.image_ids = np.array(self.image_ids)

    def compute_texts(self):
        for idx, batch in enumerate(tqdm(self.text_loader)):
            text_ids = batch['ids']
            image_target_ids = batch['image_ids']
            feats = self.model.get_lang_embeddings(batch)

            # Get embedding of each item in batch
            batch_size = feats.shape[0]
            for i in range(batch_size):
                feat = feats[i]
                self.text_embedding.append(feat)
                self.text_ids.append(text_ids[i])
                self.image_target_ids.append(image_target_ids[i])

        self.text_embedding = np.array(self.text_embedding)
        self.image_target_ids = np.array(self.image_target_ids)
        self.text_ids = np.array(self.text_ids)

    def compute_default(self, queries_embedding, gallery_embedding, queries_ids, targets_ids, gallery_ids):
        """
        Compute score for each metric and return 
        """

        # Compute distance matrice for queries and gallery
        print("Calculating distance matrice...")
        dist_mat = self.dist_func(queries_embedding, gallery_embedding)

        # np.savetxt("./results/dist_mat.txt",dist_mat)
        for idx, row in enumerate(dist_mat):
            object_dist_score = dist_mat[idx]
            top_k_indexes, top_k_scores = get_top_k(
                object_dist_score,
                top_k=self.top_k,
                max_distance=self.max_distance
            )

            current_id = queries_ids[idx] # query id
            target_ids = targets_ids[idx] # target id

            pred_ids = gallery_ids[top_k_indexes] # gallery id
            pred_ids = pred_ids.tolist()

            if self.save_results:
                self.results_dict[current_id] = {
                    'pred_ids': pred_ids,
                    'target_ids': target_ids,
                    'scores': top_k_scores 
                }
            
            for metric_name in self.metric_names:
                metric_fn = metrics_mapping[metric_name]
                metric_fn.update(pred_ids, target_ids)

    def compute_faiss(self, queries_embedding, gallery_embedding, queries_ids, targets_ids, gallery_ids):
        """
        Compute score for each metric and return using faiss
        """

        self.faiss_pool.reset()
        self.faiss_pool.add(gallery_embedding)
        top_k_scores_all, top_k_indexes_all = self.faiss_pool.search(queries_embedding, k=self.top_k)
        
        for idx, (top_k_scores, top_k_indexes) in enumerate(zip(top_k_scores_all, top_k_indexes_all)):
          
            current_id = queries_ids[idx]
            target_ids = targets_ids[idx]
            if not isinstance(target_ids, np.ndarray):
                target_ids = np.array([target_ids])

            pred_ids = gallery_ids[top_k_indexes]
            pred_ids = pred_ids.tolist()

            if self.save_results:
                self.results_dict[current_id] = {
                    'pred_ids': pred_ids,
                    'target_ids': target_ids,
                    'scores': top_k_scores 
                }
            
            for metric_name in self.metric_names:
                metric_fn = metrics_mapping[metric_name]
                metric_fn.update(pred_ids, target_ids)
                

    def _compute_score(
        self, 
        queries_embedding, 
        gallery_embedding, 
        queries_ids, 
        targets_ids, 
        gallery_ids,
        outname):

        self.results_dict = {}
        if USE_FAISS:
            self.compute_faiss(
                queries_embedding, 
                gallery_embedding, 
                queries_ids, 
                targets_ids, 
                gallery_ids)
        else:
            self.compute_default(
                queries_embedding, 
                gallery_embedding, 
                queries_ids, 
                targets_ids, 
                gallery_ids)

        # Save results for visualization later
        if self.save_results:
            print("Saving retrieval results...")
            save_results(self.results_dict, outname)

        result_dict = {}
        for metric_name in self.metric_names:
            metric_fn = metrics_mapping[metric_name]
            result_dict.update(metric_fn.value())

        return result_dict

    def compute(self):
        print("Extracting features...")
        self.reset()
        with torch.no_grad():
            self.compute_images()
            self.compute_texts()
            
        i2t_dict = self._compute_score(
            self.image_embedding, self.text_embedding, 
            self.image_ids, self.text_target_ids, self.text_ids, outname='i2t_results')

        t2i_dict = self._compute_score(
            self.text_embedding, self.image_embedding, 
            self.text_ids , self.image_target_ids, self.image_ids, outname='t2i_results')

        result_dict = {"i2t/" + k : v for k,v in i2t_dict.items()}
        result_dict.update({"t2i/" + k : v for k,v in t2i_dict.items()})
        
        return result_dict
    
    def value(self):
        result_dict = self.compute()
        return result_dict

    def __str__(self):
        return str(self.value())