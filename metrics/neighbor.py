import numpy as np
from .base_metric import TemplateMetric

class NearestNeighborMetric(TemplateMetric):
    """
    Nearest neighbor
    """
    def __init__(self, decimals = 4):
        self.decimals = decimals
        self.scores = []
        
    def compute(self):
        mean_score = np.mean(self.scores)
        mean_score = np.round(float(mean_score), decimals=self.decimals)
        return mean_score

    def update(self, output, target):
        score = int(output[0] in target)
        self.scores.append(score)

    def reset(self):
        self.scores = []

    def value(self):
        score = self.compute()
        return {"NN" : score}

    def __str__(self):
        return f'{self.value()}'

    def __len__(self):
        return len(self.scores)