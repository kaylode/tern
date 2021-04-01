import torch
import torch.utils.data as data

import os
import cv2
import pandas as pd
import numpy as np


class RetrievalDataset(data.Dataset):
    def __init__(self, config, root, csv_in, transforms=None, text_transforms=None, tokenizer=None):
        self.config = config
        self.df = pd.read_csv(csv_in)
        self.root = root
        self.transforms = transforms
        self.text_transforms = text_transforms
        self.tokenizer = tokenizer
        self.load_data()

    def load_data(self):
        self.fns = []
        annotations = [
            row
            for row in zip(
                self.df["image"],
                self.df["cleaned_title"],
                self.df['label_group'],
                self.df['posting_id'],
                self.df['target']) # string of 
        ]

        # Labels of each data point, this is for BatchSampler
        self.labels = []

        self.classes = sorted(self.df['label_group'].unique().tolist())
        self.classes_to_idx = {k:v for v,k in enumerate(self.classes)}
        self.num_classes = len(self.classes)

        self.txt_data = []

        for ann in annotations:
            image_name, title, label, post_id, str_targets = ann
            image_path = os.path.join(self.root, image_name)

            target_string = str_targets.replace(" ", ",")
            target_list = eval(target_string)

            self.txt_data.append(title)

            self.fns.append([image_path, title, label, post_id, target_list])
            self.labels.append(self.classes_to_idx[label])

    def __getitem__(self, index):
        image_path, title, label, post_id, target_list = self.fns[index]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0

        if self.transforms is not None:
            item = self.transforms(image=image)
            image = item['image']

        if self.text_transforms is not None:
            title = self.text_transforms(text=title)

        return {
            'image': image, 
            'post_id': post_id,
            'text': title,
            'label':label,
            'target': target_list
        }

    def collate_fn(self, batch):
        imgs = torch.stack([i['image'] for i in batch])
        post_ids = [i['post_id'] for i in batch]
        txts = [i['text'] for i in batch]
        lbls = [i['label'] for i in batch]
        targets = [i['target'] for i in batch]

        encoded_inputs = self.tokenizer(
            txts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True)

        return {
            'imgs': imgs,
            'txts': encoded_inputs,
            'lbls': lbls,
            'post_ids': post_ids,
            'targets': targets
        }

    def __len__(self):
        return len(self.fns)

    def __str__(self): 
        s1 = "Number of samples: " + str(len(self.fns))
        return s1