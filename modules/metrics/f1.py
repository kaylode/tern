import numpy as np
from .base_metric import TemplateMetric

class F1Metric(TemplateMetric):
    """
    F1 score at K
    Source: https://www.kaggle.com/cdeotte/part-2-rapids-tfidfvectorizer-cv-0-700?scriptVersionId=0

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
        n = len(np.intersect1d(target,output)) # Number of corrects
        score = 2*n / (len(target)+len(output))
        self.scores.append(score)

    def reset(self):
        self.scores = []

    def value(self):
        score = self.compute()
        return {f"F1@{self.k}" : score}

    def __str__(self):
        return f'{self.value()}'

    def __len__(self):
        return len(self.scores)