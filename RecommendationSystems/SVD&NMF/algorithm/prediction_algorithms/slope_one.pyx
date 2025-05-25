cimport numpy as np
import numpy as np

from .algo_base import AlgoBase
from .predictions import PredictionImpossible


class SlopeOne(AlgoBase):

    def __init__(self):

        AlgoBase.__init__(self)

    def fit(self, trainset):

        cdef int n_items = trainset.n_items

        cdef long [:, ::1] freq = np.zeros((trainset.n_items, trainset.n_items), np.int_)
        cdef double [:, ::1] dev = np.zeros((trainset.n_items, trainset.n_items), np.double)
        cdef int u, i, j, r_ui, r_uj

        AlgoBase.fit(self, trainset)

        for u, u_ratings in trainset.ur.items():
            for i, r_ui in u_ratings:
                for j, r_uj in u_ratings:
                    freq[i, j] += 1
                    dev[i, j] += r_ui - r_uj

        for i in range(n_items):
            dev[i, i] = 0
            for j in range(i + 1, n_items):
                dev[i, j] /= freq[i, j]
                dev[j, i] = -dev[i, j]

        self.freq = np.asarray(freq)
        self.dev = np.asarray(dev)

        self.user_mean = [np.mean([r for (_, r) in trainset.ur[u]])
                          for u in trainset.all_users()]

        return self

    def estimate(self, u, i):

        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            raise PredictionImpossible('User and/or item is unknown.')

        Ri = [j for (j, _) in self.trainset.ur[u] if self.freq[i, j] > 0]
        est = self.user_mean[u]
        if Ri:
            est += sum(self.dev[i, j] for j in Ri) / len(Ri)

        return est
