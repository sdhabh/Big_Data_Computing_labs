import numpy as np

from .algo_base import AlgoBase


class NormalPredictor(AlgoBase):

    def __init__(self):

        AlgoBase.__init__(self)

    def fit(self, trainset):

        AlgoBase.fit(self, trainset)

        num = sum(
            (r - self.trainset.global_mean) ** 2
            for (_, _, r) in self.trainset.all_ratings()
        )
        denum = self.trainset.n_ratings
        self.sigma = np.sqrt(num / denum)

        return self

    def estimate(self, *_):

        return np.random.normal(self.trainset.global_mean, self.sigma)
