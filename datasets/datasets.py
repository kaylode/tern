import torch
import torch.utils.data as data

import os
import cv2
import pandas as pd
import numpy as np


class RetrievalDataset(data.Dataset):
    def __init__(self, config, root, csv_in, transforms=None, tokenizer=None):
        self.config = config
        self.df = pd.read_csv(csv_in)
        self.root = root
        self.transforms = transforms
        self.tokenizer = tokenizer
        self.load_data()

    def load_data(self):
        self.fns = []
        annotations = [
            row
            for row in zip(self.df["image"], self.df["cleaned_title"])
        ]

        for ann in annotations:
            image_name, title = ann
            image_path = os.path.join(self.root, image_name)
            self.fns.append([image_path, title])

    def __getitem__(self, index):
        image_path, title = self.fns[index]

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0

        if self.transforms is not None:
            item = self.transforms(image=image)
            image = item['image']

        return {
            'image': image, 
            'text': title
        }

    def collate_fn(self, batch):
        imgs = torch.stack([i['image'] for i in batch])
        txts = [i['text'] for i in batch]

        encoded_inputs = self.tokenizer(
            txts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True)

        return {
            'imgs': imgs,
            'txts': encoded_inputs
        }

    def __len__(self):
        return len(self.fns)