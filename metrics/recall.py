import numpy as np
from .base_metric import TemplateMetric

class RecallMetric(TemplateMetric):
    """
    Nearest neighbor
    """
    def __init__(self, k, decimals = 4):
        self.k = k
        self.decimals = decimals
        self.scores = []
        
    def compute(self):
        mean_score = np.mean(self.scores)
        mean_score = np.round(float(mean_score), decimals=self.decimals)
        return mean_score

    def update(self, output, target):
        retrieved_labels = output[:self.k]
        n_targets = len(target)                                        # Number of targets
        n_relevant_objs = len(np.intersect1d(target,retrieved_labels)) # Number of corrects
        score = n_relevant_objs*1.0 / n_targets
        self.scores.append(score)

    def reset(self):
        self.scores = []

    def value(self):
        score = self.compute()
        return {f"R@{self.k}" : score}

    def __str__(self):
        return f'{self.value()}'

    def __len__(self):
        return len(self.scores)