from .algo_base import AlgoBase


class BaselineOnly(AlgoBase):

    def __init__(self, bsl_options={}, verbose=True):

        AlgoBase.__init__(self, bsl_options=bsl_options)
        self.verbose = verbose

    def fit(self, trainset):

        AlgoBase.fit(self, trainset)
        self.bu, self.bi = self.compute_baselines()

        return self

    def estimate(self, u, i):

        est = self.trainset.global_mean
        if self.trainset.knows_user(u):
            est += self.bu[u]
        if self.trainset.knows_item(i):
            est += self.bi[i]

        return est
