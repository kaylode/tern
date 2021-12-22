from torch.utils.data import DataLoader
from torchtext.legacy.data import BucketIterator
from .dataset import NumpyFeatureDataset

class NumpyFeatureLoader(BucketIterator):
    """
    Use BucketIterator to make texts of same length into batch
    """
    def __init__(self, 
                batch_size,
                root_dir,
                ann_path, 
                feat_dir, 
                text_dir, 
                device,
                **kwargs):
       
        self.dataset = NumpyFeatureDataset(
            root_dir, ann_path,  feat_dir, text_dir)

        self.collate_fn = self.dataset.collate_fn
        
        super(NumpyFeatureLoader, self).__init__(
            self.dataset,
            batch_size=batch_size,
            device=device, 
            sort_key=lambda x: len(x['text']),
            repeat=True, # Repeat the iterator for multiple epochs.
            sort=False,  # Sort all examples in data using `sort_key`.
            shuffle=True, # Shuffle data on each epoch run.
            sort_within_batch=True,
            **kwargs) # Use `sort_key` to sort examples in each batch.

class RawNumpyFeatureLoader(DataLoader):
    """
    Use DataLoader to make texts into batch
    """
    def __init__(self, 
                batch_size,
                root_dir,
                ann_path, 
                feat_dir, 
                text_dir, 
                **kwargs):
       
        self.dataset = NumpyFeatureDataset(
            root_dir, ann_path,  feat_dir, text_dir)

        self.collate_fn = self.dataset.collate_fn
        
        super(RawNumpyFeatureLoader, self).__init__(
            self.dataset,
            batch_size=batch_size,
            pin_memory=True,
            collate_fn=self.collate_fn,
            **kwargs)       
