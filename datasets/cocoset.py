import os
import cv2
import torch
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from augmentations.transforms import get_resize_augmentation, get_augmentation, Denormalize

from pycocotools.coco import COCO
from torch.utils.data import Dataset, DataLoader

from .utils import create_masks, make_feature_batch
from utils.utils import draw_image_caption

class CocoDataset(Dataset):
    """
    Coco dataset
    """
    def __init__(self, 
            root_dir, ann_path, 
            tokenizer, image_size=[512,512], 
            keep_ratio=False,
            type='train'):

        self.patch_size = 16
        self.root_dir = root_dir
        self.ann_path = ann_path
        self.image_size = image_size

        self.tokenizer = tokenizer
        self.transforms = A.Compose([
            get_resize_augmentation(image_size, keep_ratio=keep_ratio),
            get_augmentation(_type=type)
        ])

        self.coco = COCO(ann_path)
        self.image_ids = self.coco.getImgIds()

    def __len__(self):
        return len(self.image_ids)

    def load_image(self, image_index):
        image_info = self.coco.loadImgs(self.image_ids[image_index])[0]
        image_path = os.path.join(self.root_dir, image_info['file_name'])
        return image_path

    def load_annotations(self, image_index, return_all=False):
        # get ground truth annotations
        ann_id = self.coco.getAnnIds(imgIds=self.image_ids[image_index])

        if not return_all:
            ann_id = random.choice(ann_id)
            anns = self.coco.loadAnns(ann_id)[0]['caption']
        else:
            anns = self.coco.loadAnns(ann_id)
            anns = [i['caption'] for i in anns]
        return anns, ann_id

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        image_path = self.load_image(index)
        text, ann_id = self.load_annotations(index)

        return {
            'image_id': image_id,
            'ann_id': ann_id,
            'image_path': image_path,
            'text': text,
        }

    def load_augment(self, image_path):
        ori_img = cv2.imread(image_path)
        ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
        image = ori_img.astype(np.float32)
        image /= 255.0
        image = self.transforms(image=image)['image']
        return image, ori_img

    def collate_fn(self, batch):
        
        image_paths = [s['image_path'] for s in batch]
        image_ids = [s['image_id'] for s in batch]
        ann_ids = [s['ann_id'] for s in batch]
        
        image_names = []
        ori_imgs = []
        for image_path in image_paths:
            image_names.append(os.path.basename(image_path))

        imgs = []
        for image_path in image_paths:
            image, ori_img = self.load_augment(image_path)
            imgs.append(image)
            ori_imgs.append(ori_img)
        feats = torch.stack(imgs)

        texts = [s['text'] for s in batch]
        
        tokens = self.tokenizer(texts, truncation=True)
        tokens = [np.array(i) for i in tokens['input_ids']]

        texts_inp = make_feature_batch(
            tokens, pad_token=self.tokenizer.pad_token_id)
                
        texts_inp = texts_inp.squeeze(-1)

        return {
            'image_ids': image_ids,
            'ann_ids': ann_ids,
            'image_names': image_names,
            'ori_imgs': ori_imgs,
            'imgs': feats,
            'tgt_texts_raw': texts,
            'texts': texts_inp.long(),
        }


    def visualize_item(self, index = None, figsize=(15,15)):
        """
        Visualize an image with its captions by index
        """

        if index is None:
            index = random.randint(0,len(self.coco.imgs)-1)
        image_path = self.load_image(index)
        image_name = os.path.basename(image_path)
        image, _ = self.load_augment(image_path)

        texts = self.load_annotations(index, return_all=True)
        
        normalize = False
        if self.transforms is not None:
            for x in self.transforms.transforms[1]:
                if isinstance(x, A.Normalize):
                    normalize = True
                    denormalize = Denormalize(mean=x.mean, std=x.std)

        # Denormalize and reverse-tensorize
        if normalize:
            image = denormalize(img = image)

        self.visualize(image, texts, figsize = figsize, img_name= image_name)

    def visualize(self, img, texts, figsize=(15,15), img_name=None):
        """
        Visualize an image with its captions
        """

        text = []
        for i, t in enumerate(texts):
            text.append(f"{i+1}. {t}")
        text = "\n".join(text)
        fig = draw_image_caption(img, text, figsize=figsize)

        if img_name is not None:
            plt.title(img_name)
        plt.show()

    def count_dict(self, types = 1):
        """
        Count text length frequencies
        """
        cnt_dict = {}
        if types == 1: # Text length Frequencies
            for image_id in range(len(self.image_ids)):
                texts = self.load_annotations(image_id, return_all=True)
                for text in texts:
                    text_length = len(text)
                    if text_length not in cnt_dict.keys():
                        cnt_dict[text_length] = 0
                    cnt_dict[text_length] += 1
        
        return cnt_dict

    def plot(self, figsize = (8,8), types = ["length"]):
        """
        Plot distribution
        """
        ax = plt.figure(figsize = figsize)
        num_plots = len(types)
        plot_idx = 1

        if "length" in types:
            ax.add_subplot(num_plots, 1, plot_idx)
            plot_idx +=1
            cnt_dict = self.count_dict(types = 1)
            plt.title("Total texts: "+ str(sum(list(cnt_dict.values()))))
            bar1 = plt.bar(list(cnt_dict.keys()), list(cnt_dict.values()), color=[np.random.rand(3,) for i in range(len(cnt_dict.keys()))])
            for rect in bar1:
                height = rect.get_height()
                plt.text(rect.get_x() + rect.get_width()/2.0, height, '%d' % int(height), ha='center', va='bottom')
        
        plt.show()

    def __str__(self): 
        s1 = "Number of images: " + str(len(self.image_ids)) + '\n'
        s2 = "Number of texts: " + str(len(self.coco.getAnnIds())) + '\n'
        return s1 + s2

class NumpyFeatureDataset(Dataset):
    """
    Coco dataset
    """
    def __init__(self, root_dir, ann_path, feat_dir, text_dir):

        self.root_dir = root_dir
        self.ann_path = ann_path
        self.feat_dir = feat_dir
        self.text_dir = text_dir
        self.coco = COCO(ann_path)
        self.image_ids = self.coco.getImgIds()

    def __len__(self):
        return len(self.image_ids)

    def load_image(self, image_index):
        image_info = self.coco.loadImgs(self.image_ids[image_index])[0]
        image_path = os.path.join(self.root_dir, image_info['file_name'])
        return image_path

    def load_image_by_id(self, image_id):
        image_infos = self.coco.loadImgs(image_id)
        image_path = [os.path.join(self.root_dir, i['file_name']) for i in image_infos]
        return image_path

    def load_annotations_by_id(self, ann_id):
        anns = self.coco.loadAnns(ann_id)
        anns = [i['caption'] for i in anns]
        return anns

    def load_numpy(self, image_index):
        image_info = self.coco.loadImgs(self.image_ids[image_index])[0]
        npy_path = os.path.join(self.feat_dir, 'data_att', image_info['file_name'][:-4]+'.npz')
        npy_loc_path = os.path.join(self.feat_dir, 'data_box', image_info['file_name'][:-4]+'.npz')
        return npy_path, npy_loc_path

    def load_annotations(self, image_index, return_all=False):
        # get ground truth annotations
        ann_id = self.coco.getAnnIds(imgIds=self.image_ids[image_index])

        if not return_all:
            ann_id = random.choice(ann_id)
            anns = self.coco.loadAnns(ann_id)[0]['caption']
            language_path = os.path.join(self.text_dir, f"{ann_id}.npz")
            return anns, language_path, ann_id
        else:
            anns = self.coco.loadAnns(ann_id)
            anns = [i['caption'] for i in anns]
            return anns, ann_id

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        image_path = self.load_image(index)
        npy_path, npy_loc_path = self.load_numpy(index)
        text, language_path, ann_id = self.load_annotations(index)

        return {
            'image_id': image_id,
            'ann_id': ann_id,
            'npy_path': npy_path,
            'language_path': language_path,
            "npy_loc_path": npy_loc_path,
            'image_path': image_path,
            'text': text,
        }

    def collate_fn(self, batch):
        
        image_paths = [s['image_path'] for s in batch]
        npy_paths = [s['npy_path'] for s in batch]
        npy_loc_paths = [s['npy_loc_path'] for s in batch]
        language_paths = [s['language_path'] for s in batch]
        image_ids = [s['image_id'] for s in batch]
        ann_ids = [s['ann_id'] for s in batch]
        texts = [s['text'] for s in batch]
        
        image_names = []
        ori_imgs = []
        for image_path in image_paths:
            image_names.append(os.path.basename(image_path))

        for image_path in image_paths:
            ori_img = cv2.imread(image_path)
            ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
            ori_imgs.append(ori_img)
        
        npy_feats = []
        npy_loc_feats = []
        npy_lang_feats = []
        for npy_path, npy_loc_path, language_path in zip(npy_paths, npy_loc_paths, language_paths):
            npy_feat = np.load(npy_path, mmap_mode='r')['feat']
            npy_loc_feat = np.load(npy_loc_path, mmap_mode='r')['feat']
            npy_lang_feat = np.load(language_path, mmap_mode='r')['feat']
            npy_feats.append(npy_feat)
            npy_loc_feats.append(npy_loc_feat)
            npy_lang_feats.append(npy_lang_feat)

        npy_feats = np.stack(npy_feats, axis=0)
        npy_loc_feats = np.stack(npy_loc_feats, axis=0)

        feats = torch.from_numpy(npy_feats).float()
        loc_feats = torch.from_numpy(npy_loc_feats).float()

        lang_feats = make_feature_batch(npy_lang_feats, pad_token=0)
        lang_feats = lang_feats.float()

        return {
            'image_ids': image_ids,
            'ann_ids': ann_ids,
            'image_names': image_names,
            'ori_imgs': ori_imgs,
            'feats': feats,
            'loc_feats': loc_feats,
            'lang_feats': lang_feats,
            'tgt_texts_raw': texts,
        }

    def __str__(self): 
        s1 = "Number of images: " + str(len(self.image_ids)) + '\n'
        s2 = "Number of texts: " + str(len(self.coco.getAnnIds())) + '\n'
        return s1 + s2
