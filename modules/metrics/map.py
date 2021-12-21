import numpy as np
from .base_metric import TemplateMetric

class MAPMetric(TemplateMetric):
    """
    Mean average precision at K
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
        output = output[:self.k]
        score = 0.0
        num_hits = 0.0

        for i, p in enumerate(output):
            if p in target and p not in output[:i]:
                num_hits += 1.0
                score += num_hits / (i+1.0)
        self.scores.append(score)

    def reset(self):
        self.scores = []

    def value(self):
        score = self.compute()
        return {f"MAP@{self.k}" : score}

    def __str__(self):
        return f'{self.value()}'

    def __len__(self):
        return len(self.scores)