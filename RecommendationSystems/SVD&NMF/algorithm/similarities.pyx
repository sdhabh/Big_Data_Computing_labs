cimport numpy as np  # noqa
import numpy as np
from libc.math cimport sqrt


def cosine(int n_x, yr, int min_support):

    # sum (r_xy * r_x'y) for common ys
    cdef double [:, ::1] prods = np.zeros((n_x, n_x), np.double)
    # number of common ys
    cdef long [:, ::1] freq = np.zeros((n_x, n_x), np.int_)
    # sum (r_xy ^ 2) for common ys
    cdef double [:, ::1] sqi = np.zeros((n_x, n_x), np.double)
    # sum (r_x'y ^ 2) for common ys
    cdef double [:, ::1] sqj = np.zeros((n_x, n_x), np.double)
    # the similarity matrix
    cdef double [:, ::1] sim = np.zeros((n_x, n_x), np.double)

    cdef int xi, xj, y
    cdef double ri, rj
    cdef int min_sprt = min_support

    for y, y_ratings in yr.items():
        for xi, ri in y_ratings:
            for xj, rj in y_ratings:
                freq[xi, xj] += 1
                prods[xi, xj] += ri * rj
                sqi[xi, xj] += ri**2
                sqj[xi, xj] += rj**2

    for xi in range(n_x):
        sim[xi, xi] = 1
        for xj in range(xi + 1, n_x):
            if freq[xi, xj] < min_sprt:
                sim[xi, xj] = 0
            else:
                denum = sqrt(sqi[xi, xj] * sqj[xi, xj])
                sim[xi, xj] = prods[xi, xj] / denum

            sim[xj, xi] = sim[xi, xj]

    return np.asarray(sim)


def msd(int n_x, yr, int min_support):

    # sum (r_xy - r_x'y)**2 for common ys
    cdef double [:, ::1] sq_diff = np.zeros((n_x, n_x), np.double)
    # number of common ys
    cdef long [:, ::1] freq = np.zeros((n_x, n_x), np.int_)
    # the similarity matrix
    cdef double [:, ::1] sim = np.zeros((n_x, n_x), np.double)

    cdef int xi, xj
    cdef double ri, rj
    cdef int min_sprt = min_support

    for y, y_ratings in yr.items():
        for xi, ri in y_ratings:
            for xj, rj in y_ratings:
                sq_diff[xi, xj] += (ri - rj)**2
                freq[xi, xj] += 1

    for xi in range(n_x):
        sim[xi, xi] = 1  # completely arbitrary and useless anyway
        for xj in range(xi + 1, n_x):
            if freq[xi, xj] < min_sprt:
                sim[xi, xj] = 0
            else:
                # return inverse of (msd + 1) (+ 1 to avoid dividing by zero)
                sim[xi, xj] = 1 / (sq_diff[xi, xj] / freq[xi, xj] + 1)

            sim[xj, xi] = sim[xi, xj]

    return np.asarray(sim)


def pearson(int n_x, yr, int min_support):
    # number of common ys
    cdef long [:, ::1] freq = np.zeros((n_x, n_x), np.int_)
    # sum (r_xy * r_x'y) for common ys
    cdef double [:, ::1] prods = np.zeros((n_x, n_x), np.double)
    # sum (rxy ^ 2) for common ys
    cdef double [:, ::1] sqi = np.zeros((n_x, n_x), np.double)
    # sum (rx'y ^ 2) for common ys
    cdef double [:, ::1] sqj = np.zeros((n_x, n_x), np.double)
    # sum (rxy) for common ys
    cdef double [:, ::1] si = np.zeros((n_x, n_x), np.double)
    # sum (rx'y) for common ys
    cdef double [:, ::1] sj = np.zeros((n_x, n_x), np.double)
    # the similarity matrix
    cdef double [:, ::1] sim = np.zeros((n_x, n_x), np.double)

    cdef int xi, xj, y, n
    cdef double ri, rj, num, denum
    cdef int min_sprt = min_support

    for y, y_ratings in yr.items():
        for xi, ri in y_ratings:
            for xj, rj in y_ratings:
                prods[xi, xj] += ri * rj
                freq[xi, xj] += 1
                sqi[xi, xj] += ri**2
                sqj[xi, xj] += rj**2
                si[xi, xj] += ri
                sj[xi, xj] += rj

    for xi in range(n_x):
        sim[xi, xi] = 1
        for xj in range(xi + 1, n_x):

            if freq[xi, xj] < min_sprt:
                sim[xi, xj] = 0
            else:
                n = freq[xi, xj]
                num = n * prods[xi, xj] - si[xi, xj] * sj[xi, xj]
                denum = sqrt((n * sqi[xi, xj] - si[xi, xj]**2) *
                             (n * sqj[xi, xj] - sj[xi, xj]**2))
                if denum == 0:
                    sim[xi, xj] = 0
                else:
                    sim[xi, xj] = num / denum

            sim[xj, xi] = sim[xi, xj]

    return np.asarray(sim)


def pearson_baseline(
    int n_x,
    yr,
    int min_support,
    double global_mean,
    double [::1] x_biases,
    double [::1] y_biases,
    double shrinkage=100,
):

    # number of common ys
    cdef long [:, ::1] freq = np.zeros((n_x, n_x), np.int_)
    # sum (r_xy - b_xy) * (r_x'y - b_x'y) for common ys
    cdef double [:, ::1] prods = np.zeros((n_x, n_x), np.double)
    # sum (r_xy - b_xy)**2 for common ys
    cdef double [:, ::1] sq_diff_i = np.zeros((n_x, n_x), np.double)
    # sum (r_x'y - b_x'y)**2 for common ys
    cdef double [:, ::1] sq_diff_j = np.zeros((n_x, n_x), np.double)
    # the similarity matrix
    cdef double [:, ::1] sim = np.zeros((n_x, n_x), np.double)

    cdef int y, xi, xj
    cdef double ri, rj, diff_i, diff_j, partial_bias
    cdef int min_sprt = min_support
    cdef double global_mean_ = global_mean

    # Need this because of shrinkage. When pearson coeff is zero when support
    # is 1, so that's OK.
    min_sprt = max(2, min_sprt)

    for y, y_ratings in yr.items():
        partial_bias = global_mean_ + y_biases[y]
        for xi, ri in y_ratings:
            for xj, rj in y_ratings:
                freq[xi, xj] += 1
                diff_i = (ri - (partial_bias + x_biases[xi]))
                diff_j = (rj - (partial_bias + x_biases[xj]))
                prods[xi, xj] += diff_i * diff_j
                sq_diff_i[xi, xj] += diff_i**2
                sq_diff_j[xi, xj] += diff_j**2

    for xi in range(n_x):
        sim[xi, xi] = 1
        for xj in range(xi + 1, n_x):
            if freq[xi, xj] < min_sprt:
                sim[xi, xj] = 0
            else:
                sim[xi, xj] = prods[xi, xj] / (sqrt(sq_diff_i[xi, xj] * sq_diff_j[xi, xj]))
                # the shrinkage part
                sim[xi, xj] *= (freq[xi, xj] - 1) / (freq[xi, xj] - 1 + shrinkage)

            sim[xj, xi] = sim[xi, xj]

    return np.asarray(sim)
