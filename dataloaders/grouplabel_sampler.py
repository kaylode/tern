import pandas as pd
import numpy as np
from torch.utils.data.sampler import Sampler

"""
Sample two items with same label group
"""

class SameGroupSampler(Sampler):
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