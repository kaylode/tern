import numpy as np
from .base_metric import TemplateMetric

class FirstTierMetric(TemplateMetric):
    """
    First Tier
    """
    def __init__(self, decimals = 4):
        self.decimals = decimals
        self.scores = []
        
    def compute(self):
        mean_score = np.mean(self.scores)
        mean_score = np.round(float(mean_score), decimals=self.decimals)
        return mean_score

    def update(self, output, target):
        n_relevant_objs = sum([1 if i in target else 0 for i in output])
        retrieved_1st_tier = output[:n_relevant_objs+1]
        score = np.mean([1 if i in target else 0 for i in retrieved_1st_tier])
        self.scores.append(score)

    def reset(self):
        self.scores = []

    def value(self):
        score = self.compute()
        return {"FT" : score}

    def __str__(self):
        return f'{self.value()}'

    def __len__(self):
        return len(self.scores)


class SecondTierMetric(TemplateMetric):
    """
    Second Tier
    """
    def __init__(self, decimals = 4):
        self.decimals = decimals
        self.scores = []
        
    def compute(self):
        mean_score = np.mean(self.scores)
        mean_score = np.round(float(mean_score), decimals=self.decimals)
        return mean_score

    def update(self, output, target):
        n_relevant_objs = sum([1 if i in target else 0 for i in output])
        retrieved_2nd_tier = output[:2*n_relevant_objs+1]
        score = np.mean([1 if i in target else 0 for i in retrieved_2nd_tier])
        self.scores.append(score)

    def reset(self):
        self.scores = []

    def value(self):
        score = self.compute()
        return {"ST" : score}

    def __str__(self):
        return f'{self.value()}'

    def __len__(self):
        return len(self.scores)