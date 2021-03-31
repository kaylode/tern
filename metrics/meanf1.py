# from . import TemplateMetric
import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances
import torch
import torch.nn as nn
from tqdm import tqdm

USE_PHASH=True

if USE_PHASH:
    info_df = pd.read_csv('./data/shopee-matching/annotations/train_clean2.csv')
    postid_phash_mapping = {k:v for k,v in zip(info_df['posting_id'], info_df['image_phash'])}

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

def phash_trick(post_id, pred_post_ids=[]):
    phash_id = postid_phash_mapping[post_id] # phash of current post
    post_ids_same_phash = info_df[info_df['image_phash'] == phash_id].posting_id.tolist() # all posts with same phash
    return post_ids_same_phash #+ list(set(pred_post_ids) - set(post_ids_same_phash))

class MeanF1Score():    
    def __init__(self, 
            queries_loader, 
            gallery_loader=None, 
            retrieval_pairing='txt-to-img', 
            max_distance = 0.001,
            top_k=10):

        self.queries_loader = queries_loader
        self.gallery_loader = gallery_loader if gallery_loader is not None else queries_loader
        
        self.top_k = top_k                  # Query top k candidates
        self.max_distance = max_distance    # Query candidates with distances lower than threshold

        self.queries_embedding = [] 
        self.gallery_embedding = []
        self.queries_post_ids = []
        self.gallery_post_ids = []
        self.targets_post_ids = []
 

        self.retrieval_pairing = retrieval_pairing
        self.queries_embedding_type = retrieval_pairing.split('-')[0]
        self.gallery_embedding_type = retrieval_pairing.split('-')[2]

        # Distance function
        self.dist_func = get_dist_func('cosine')
    
    def update(self, model):
        self.model = model
        self.model.eval()

    def compute_queries(self):
        for idx, batch in enumerate(tqdm(self.queries_loader)):
            post_ids = batch['post_ids']
            target_ids = batch['targets']
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

    def compute_gallery(self):
        for idx, batch in enumerate(tqdm(self.gallery_loader)):
            post_ids = batch['post_ids']
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
        self.gallery_post_ids = np.array(self.gallery_post_ids)

    def compute(self):
        print("Extracting features...")
        self.compute_queries()
        self.compute_gallery()

        # Compute distance matrice for queries and gallery
        print("Calculating distance matrice...")
        dist_mat = self.dist_func(self.queries_embedding, self.gallery_embedding)
        #np.savetxt("./results/dist_mat.txt",dist_mat)

        total_scores = []
        for idx, row in enumerate(dist_mat):
            object_dist_score = dist_mat[idx]

            # Sort item by distance and get top-k
            top_k_indexes = object_dist_score.argsort()[:self.top_k]
            top_k_scores = object_dist_score[top_k_indexes]

            # Keep only item with near distance
            if self.max_distance is not None:
                keep_indexes = top_k_scores <= self.max_distance
                top_k_indexes = top_k_indexes[keep_indexes]
                top_k_scores = top_k_scores[keep_indexes]

            current_post_id = self.queries_post_ids[idx]
            target_post_ids = self.targets_post_ids[idx]

            pred_post_ids = self.gallery_post_ids[top_k_indexes]
            pred_post_ids = pred_post_ids.tolist()

            # Add post with same phash into prediction list
            if USE_PHASH:
                pred_post_ids = phash_trick(current_post_id, pred_post_ids)

            # F1 score: https://www.kaggle.com/cdeotte/part-2-rapids-tfidfvectorizer-cv-0-700?scriptVersionId=0
            n = len(np.intersect1d(target_post_ids,pred_post_ids)) # Number of corrects
            score = 2*n / (len(target_post_ids)+len(pred_post_ids))
            
            total_scores.append(score)

        return np.mean(total_scores)
            
    def value(self):
        result = self.compute()
        return {
            'mean-f1': np.round(float(result), 3)
        }

    def __str__(self):
        return str(self.value())
        
if __name__ == '__main__':
    from utils.getter import *
    config = Config('/home/pmkhoi/source/shopee-matching/configs/config.yaml')
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu') 
    trainset, valset, trainloader, valloader = get_dataset_and_dataloader(config)
    net = get_model(None, config)
    model = Retrieval(
            model = net,
            device = device)


    metric = MeanF1Score(valloader, valloader)
    metric.update(model)
    with torch.no_grad():
        print(metric)
    