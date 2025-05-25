cimport numpy as np  # noqa
import numpy as np
from libc.math cimport sqrt

from .algo_base import AlgoBase
from .predictions import PredictionImpossible
from ..utils import get_rng

import cython
from libc.stdlib cimport malloc, free


class SVD(AlgoBase):

    def __init__(self, n_factors=100, n_epochs=20, biased=True, init_mean=0,
                 init_std_dev=.1, lr_all=.005,
                 reg_all=.02, lr_bu=None, lr_bi=None, lr_pu=None, lr_qi=None,
                 reg_bu=None, reg_bi=None, reg_pu=None, reg_qi=None,
                 random_state=None, verbose=False):

        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.biased = biased
        self.init_mean = init_mean
        self.init_std_dev = init_std_dev
        self.lr_bu = lr_bu if lr_bu is not None else lr_all
        self.lr_bi = lr_bi if lr_bi is not None else lr_all
        self.lr_pu = lr_pu if lr_pu is not None else lr_all
        self.lr_qi = lr_qi if lr_qi is not None else lr_all
        self.reg_bu = reg_bu if reg_bu is not None else reg_all
        self.reg_bi = reg_bi if reg_bi is not None else reg_all
        self.reg_pu = reg_pu if reg_pu is not None else reg_all
        self.reg_qi = reg_qi if reg_qi is not None else reg_all
        self.random_state = random_state
        self.verbose = verbose

        AlgoBase.__init__(self)

    def fit(self, trainset):

        AlgoBase.fit(self, trainset)
        self.sgd(trainset)

        return self

    def sgd(self, trainset):

        rng = get_rng(self.random_state)

        # user biases
        cdef double [::1] bu = np.zeros(trainset.n_users, dtype=np.double)
        # item biases
        cdef double [::1] bi = np.zeros(trainset.n_items, dtype=np.double)
        # user factors
        cdef double [:, ::1] pu = rng.normal(self.init_mean, self.init_std_dev, size=(trainset.n_users, self.n_factors))
        # item factors
        cdef double [:, ::1] qi = rng.normal(self.init_mean, self.init_std_dev, size=(trainset.n_items, self.n_factors))

        cdef int u, i, f
        cdef int n_factors = self.n_factors
        cdef bint biased = self.biased

        cdef double r, err, dot, puf, qif
        cdef double global_mean = self.trainset.global_mean

        cdef double lr_bu = self.lr_bu
        cdef double lr_bi = self.lr_bi
        cdef double lr_pu = self.lr_pu
        cdef double lr_qi = self.lr_qi

        cdef double reg_bu = self.reg_bu
        cdef double reg_bi = self.reg_bi
        cdef double reg_pu = self.reg_pu
        cdef double reg_qi = self.reg_qi

        if not biased:
            global_mean = 0

        for current_epoch in range(self.n_epochs):
            if self.verbose:
                print("Processing epoch {}".format(current_epoch))

            for u, i, r in trainset.all_ratings():
                # compute current error
                dot = 0  # <q_i, p_u>
                for f in range(n_factors):
                    dot += qi[i, f] * pu[u, f]
                err = r - (global_mean + bu[u] + bi[i] + dot)

                # update biases
                if biased:
                    bu[u] += lr_bu * (err - reg_bu * bu[u])
                    bi[i] += lr_bi * (err - reg_bi * bi[i])

                # update factors
                for f in range(n_factors):
                    puf = pu[u, f]
                    qif = qi[i, f]
                    pu[u, f] += lr_pu * (err * qif - reg_pu * puf)
                    qi[i, f] += lr_qi * (err * puf - reg_qi * qif)

        self.bu = np.asarray(bu)
        self.bi = np.asarray(bi)
        self.pu = np.asarray(pu)
        self.qi = np.asarray(qi)
    

    def estimate(self, u, i):
        # Should we cythonize this as well?

        known_user = self.trainset.knows_user(u)
        known_item = self.trainset.knows_item(i)

        if self.biased:
            est = self.trainset.global_mean

            if known_user:
                est += self.bu[u]

            if known_item:
                est += self.bi[i]

            if known_user and known_item:
                est += np.dot(self.qi[i], self.pu[u])

        else:
            if known_user and known_item:
                est = np.dot(self.qi[i], self.pu[u])
            else:
                raise PredictionImpossible('User and item are unknown.')

        return est


@cython.cdivision(True)
class SVDpp(AlgoBase):

    def __init__(self, n_factors=20, n_epochs=20, init_mean=0, init_std_dev=.1,
                 lr_all=.007, reg_all=.02, lr_bu=None, lr_bi=None, lr_pu=None,
                 lr_qi=None, lr_yj=None, reg_bu=None, reg_bi=None, reg_pu=None,
                 reg_qi=None, reg_yj=None, random_state=None, verbose=False,
                 cache_ratings=False):

        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.init_mean = init_mean
        self.init_std_dev = init_std_dev
        self.lr_bu = lr_bu if lr_bu is not None else lr_all
        self.lr_bi = lr_bi if lr_bi is not None else lr_all
        self.lr_pu = lr_pu if lr_pu is not None else lr_all
        self.lr_qi = lr_qi if lr_qi is not None else lr_all
        self.lr_yj = lr_yj if lr_yj is not None else lr_all
        self.reg_bu = reg_bu if reg_bu is not None else reg_all
        self.reg_bi = reg_bi if reg_bi is not None else reg_all
        self.reg_pu = reg_pu if reg_pu is not None else reg_all
        self.reg_qi = reg_qi if reg_qi is not None else reg_all
        self.reg_yj = reg_yj if reg_yj is not None else reg_all
        self.random_state = random_state
        self.verbose = verbose
        self.cache_ratings = cache_ratings

        AlgoBase.__init__(self)

    def fit(self, trainset):

        AlgoBase.fit(self, trainset)
        self.sgd(trainset)

        return self

    def sgd(self, trainset):

        rng = get_rng(self.random_state)

        # user biases
        cdef double [::1] bu = np.zeros(trainset.n_users, dtype=np.double)
        # item biases
        cdef double [::1] bi = np.zeros(trainset.n_items, dtype=np.double)
        # user factors
        cdef double [:, ::1] pu = rng.normal(self.init_mean, self.init_std_dev, size=(trainset.n_users, self.n_factors))
        # item factors
        cdef double [:, ::1] qi = rng.normal(self.init_mean, self.init_std_dev, size=(trainset.n_items, self.n_factors))
        # item implicit factors
        cdef double [:, ::1] yj = rng.normal(self.init_mean, self.init_std_dev, size=(trainset.n_items, self.n_factors))

        cdef double [::1] u_impl_fdb = np.zeros(self.n_factors, dtype=np.double)

        cdef int u, i, j, f, k, Iu_length
        cdef int max_Iu_length = 0
        cdef int n_factors = self.n_factors
        cdef bint cache_ratings = self.cache_ratings
        cdef double r, err, dot, puf, qif, sqrt_Iu, _
        cdef double global_mean = self.trainset.global_mean

        cdef double lr_bu = self.lr_bu
        cdef double lr_bi = self.lr_bi
        cdef double lr_pu = self.lr_pu
        cdef double lr_qi = self.lr_qi
        cdef double lr_yj = self.lr_yj

        cdef double reg_bu = self.reg_bu
        cdef double reg_bi = self.reg_bi
        cdef double reg_pu = self.reg_pu
        cdef double reg_qi = self.reg_qi
        cdef double reg_yj = self.reg_yj
        cdef double err_qif_sqrt = 0.0

        cdef int ** Iu_cached = NULL
        cdef int * Iu = NULL
        cdef int * Iu_lengths = NULL

        if cache_ratings:
            Iu_cached = <int **>malloc(trainset.n_users * sizeof(int *))
            Iu_lengths = <int *>malloc(trainset.n_users * sizeof(int))
            for u in range(trainset.n_users):
                Iu_lengths[u] = len(trainset.ur[u])
                Iu_cached[u] = <int *>malloc(Iu_lengths[u] * sizeof(int))
                for k, (j, _) in enumerate(trainset.ur[u]):
                    Iu_cached[u][k] = j
        else:
            for u in range(trainset.n_users):
                # Might as well allocate the max size once and for all
                # instead of allocating the exact size each time
                max_Iu_length = max(max_Iu_length, len(trainset.ur[u]))
                Iu = <int *>malloc(max_Iu_length * sizeof(int))

        for current_epoch in range(self.n_epochs):
            if self.verbose:
                print(" processing epoch {}".format(current_epoch))

            for u, i, r in trainset.all_ratings():

                # items rated by u.
                if cache_ratings:
                    Iu = Iu_cached[u]
                    Iu_length = Iu_lengths[u]
                else:
                    for k, (j, _) in enumerate(trainset.ur[u]):
                        Iu[k] = j
                    Iu_length = k + 1

                sqrt_Iu = sqrt(Iu_length)

                # compute user implicit feedback
                # for f in range(n_factors):
                u_impl_fdb[:] = 0

                for k in range(Iu_length):
                    j = Iu[k]
                    for f in range(n_factors):
                        u_impl_fdb[f] += yj[j, f] / sqrt_Iu
                
                # compute current error
                dot = 0 
                for f in range(n_factors):
                    dot += qi[i, f] * (pu[u, f] + u_impl_fdb[f])

                err = r - (global_mean + bu[u] + bi[i] + dot)

                # update biases
                bu[u] += lr_bu * (err - reg_bu * bu[u])
                bi[i] += lr_bi * (err - reg_bi * bi[i])

                # update factors
                for f in range(n_factors):
                    puf = pu[u, f]
                    qif = qi[i, f]
                    pu[u, f] += lr_pu * (err * qif - reg_pu * puf)
                    qi[i, f] += lr_qi * (err * (puf + u_impl_fdb[f]) - reg_qi * qif)
                    err_qif_sqrt = err * qif / sqrt_Iu
                    for k in range(Iu_length):
                        j = Iu[k]
                        yj[j, f] += lr_yj * (err_qif_sqrt - reg_yj * yj[j, f])

        if cache_ratings:
            for u in range(trainset.n_users):
                free(Iu_cached[u])
            free(Iu_cached)
            free(Iu_lengths)
        else:
            free(Iu)

        self.bu = np.asarray(bu)
        self.bi = np.asarray(bi)
        self.pu = np.asarray(pu)
        self.qi = np.asarray(qi)
        self.yj = np.asarray(yj)

    def estimate(self, u, i):

        est = self.trainset.global_mean

        if self.trainset.knows_user(u):
            est += self.bu[u]

        if self.trainset.knows_item(i):
            est += self.bi[i]

        if self.trainset.knows_user(u) and self.trainset.knows_item(i):
            Iu = len(self.trainset.ur[u])  # nb of items rated by u
            u_impl_feedback = (sum(self.yj[j] for (j, _)
                               in self.trainset.ur[u]) / np.sqrt(Iu))
            est += np.dot(self.qi[i], self.pu[u] + u_impl_feedback)

        return est


class NMF(AlgoBase):

    def __init__(self, n_factors=15, n_epochs=50, biased=False, reg_pu=.06,
                 reg_qi=.06, reg_bu=.02, reg_bi=.02, lr_bu=.005, lr_bi=.005,
                 init_low=0, init_high=1, random_state=None, verbose=False):

        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.biased = biased
        self.reg_pu = reg_pu
        self.reg_qi = reg_qi
        self.lr_bu = lr_bu
        self.lr_bi = lr_bi
        self.reg_bu = reg_bu
        self.reg_bi = reg_bi
        self.init_low = init_low
        self.init_high = init_high
        self.random_state = random_state
        self.verbose = verbose

        if self.init_low < 0:
            raise ValueError('init_low should be greater than zero')

        AlgoBase.__init__(self)

    def fit(self, trainset):

        AlgoBase.fit(self, trainset)
        self.sgd(trainset)

        return self

    def sgd(self, trainset):

        rng = get_rng(self.random_state)

        # user and item factors
        cdef double [:, ::1] pu = rng.uniform(self.init_low, self.init_high, size=(trainset.n_users, self.n_factors))
        cdef double [:, ::1] qi = rng.uniform(self.init_low, self.init_high, size=(trainset.n_items, self.n_factors))

        # user and item biases
        cdef double [::1] bu = np.zeros(trainset.n_users, dtype=np.double)
        cdef double [::1] bi = np.zeros(trainset.n_items, dtype=np.double)

        cdef int u, i, f
        cdef int n_factors = self.n_factors
        cdef double r, est, l, dot, err
        cdef double reg_pu = self.reg_pu
        cdef double reg_qi = self.reg_qi
        cdef double reg_bu = self.reg_bu
        cdef double reg_bi = self.reg_bi
        cdef double lr_bu = self.lr_bu
        cdef double lr_bi = self.lr_bi
        cdef double global_mean = self.trainset.global_mean

        # auxiliary matrices used in optimization process
        cdef double [:, ::1] user_num = np.zeros((trainset.n_users, n_factors))
        cdef double [:, ::1] user_denom = np.zeros((trainset.n_users, n_factors))
        cdef double [:, ::1] item_num = np.zeros((trainset.n_items, n_factors))
        cdef double [:, ::1] item_denom = np.zeros((trainset.n_items, n_factors))


        if not self.biased:
            global_mean = 0

        for current_epoch in range(self.n_epochs):

            if self.verbose:
                print("Processing epoch {}".format(current_epoch))

            # (re)initialize nums and denoms to zero
            # TODO: Use fill or memset??
            user_num[:, :] = 0
            user_denom[:, :] = 0
            item_num[:, :] = 0
            item_denom[:, :] = 0

            # Compute numerators and denominators for users and items factors
            for u, i, r in trainset.all_ratings():

                # compute current estimation and error
                dot = 0  # <q_i, p_u>
                for f in range(n_factors):
                    dot += qi[i, f] * pu[u, f]
                est = global_mean + bu[u] + bi[i] + dot
                err = r - est

                # update biases
                if self.biased:
                    bu[u] += lr_bu * (err - reg_bu * bu[u])
                    bi[i] += lr_bi * (err - reg_bi * bi[i])

                # compute numerators and denominators
                for f in range(n_factors):
                    user_num[u, f] += qi[i, f] * r
                    user_denom[u, f] += qi[i, f] * est
                    item_num[i, f] += pu[u, f] * r
                    item_denom[i, f] += pu[u, f] * est

            # Update user factors
            for u in trainset.all_users():
                n_ratings = len(trainset.ur[u])
                for f in range(n_factors):
                    if pu[u, f] != 0:  # Can happen if user only has 0 ratings
                        user_denom[u, f] += n_ratings * reg_pu * pu[u, f]
                        pu[u, f] *= user_num[u, f] / user_denom[u, f]

            # Update item factors
            for i in trainset.all_items():
                n_ratings = len(trainset.ir[i])
                for f in range(n_factors):
                    if qi[i, f] != 0:
                        item_denom[i, f] += n_ratings * reg_qi * qi[i, f]
                        qi[i, f] *= item_num[i, f] / item_denom[i, f]

        self.bu = np.asarray(bu)
        self.bi = np.asarray(bi)
        self.pu = np.asarray(pu)
        self.qi = np.asarray(qi)

    def estimate(self, u, i):
        # Should we cythonize this as well?

        known_user = self.trainset.knows_user(u)
        known_item = self.trainset.knows_item(i)

        if self.biased:
            est = self.trainset.global_mean

            if known_user:
                est += self.bu[u]

            if known_item:
                est += self.bi[i]

            if known_user and known_item:
                est += np.dot(self.qi[i], self.pu[u])

        else:
            if known_user and known_item:
                est = np.dot(self.qi[i], self.pu[u])
            else:
                raise PredictionImpossible('User and item are unknown.')

        return est
