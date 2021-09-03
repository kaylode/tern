import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler
from torchtext.legacy.data import BucketIterator

from .cocoset import NumpyFeatureDataset



class NumpyFeatureLoader(BucketIterator):
    """
    Use BucketIterator to make texts of same length into batch
    """
    def __init__(self, 
                batch_size,
                root_dir,
                npy_dir, 
                ann_path, 
                tokenizer,
                device,
                **kwargs):
       
        self.dataset = NumpyFeatureDataset(
            root_dir, ann_path, tokenizer, npy_dir)

        self.tokenizer = self.dataset.tokenizer
        self.collate_fn = self.dataset.collate_fn
        
        super(NumpyFeatureLoader, self).__init__(
            self.dataset,
            batch_size=batch_size,
            device=device, 
            sort_key=lambda x: len(x['text']),
            repeat=True, # Repeat the iterator for multiple epochs.
            sort=False,  # Sort all examples in data using `sort_key`.
            shuffle=True, # Shuffle data on each epoch run.
            sort_within_batch=True) # Use `sort_key` to sort examples in each batch.

class RawNumpyFeatureLoader(DataLoader):
    """
    Use DataLoader to make texts into batch
    """
    def __init__(self, 
                batch_size,
                root_dir,
                npy_dir, 
                ann_path, 
                tokenizer,
                **kwargs):
       
        self.dataset = NumpyFeatureDataset(
            root_dir, ann_path, tokenizer, npy_dir)

        self.tokenizer = self.dataset.tokenizer
        self.collate_fn = self.dataset.collate_fn
        
        super(RawNumpyFeatureLoader, self).__init__(
            self.dataset,
            batch_size=batch_size,
            num_workers=2,
            pin_memory=True,
            collate_fn=self.collate_fn)

class SameGroupSampler(Sampler):
    """
    Sample two items with same label group
    """
    def __init__(self, csv_in ,dataset):
        super().__init__(dataset)
        
        df = pd.read_csv(csv_in)
        # Create a dictionary of posting_id -> index in dataset
        self.index_to_position = dict(zip(df.posting_id, range(len(df))))
        
        # Create a Series of label_group -> set(posting_id)
        self.label_group = df.reset_index().groupby('label_group')['posting_id'].apply(set).map(sorted).map(np.array)

    def __len__(self):
        return len(self.label_group)
        
    def __iter__(self):
        for _ in range(len(self)):
            # Sample one label_group
            label_group_sample = self.label_group.sample(1).iloc[0]
            
            # Check if contains only one group label
            if len(label_group_sample) < 2:
                sample1 = label_group_sample[0]
                yield self.index_to_position[sample1]
            else:
                # Sample two posting_id's
                sample1, sample2 = np.random.choice(label_group_sample, 2, replace=False)

                yield self.index_to_position[sample1]
                yield self.index_to_position[sample2]            