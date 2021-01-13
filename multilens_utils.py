import math
import sys

import graph_tool.all as gt
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class FeatDefUtil():
    def __init__(self):
        None

    @classmethod
    def to_base_feat(cls, feat_def):
        return feat_def.split('-')[-1]


class NeighborOp():
    def __init__(self):
        None

    @classmethod
    def out_nbr(cls, g, v):
        return g.get_out_neighbors(v)

    @classmethod
    def in_nbr(cls, g, v):
        return g.get_in_neighbors(v)

    @classmethod
    def all_nbr(cls, g, v):
        return np.unique(
            np.concatenate((g.get_out_neighbors(v), g.get_in_neighbors(v)),
                           axis=None))


class RelFeatOp():
    def __init__(self):
        None

    @classmethod
    def mean(cls, S, x, na_fill=0.0):
        result = 0.0
        for v in S:
            result += x[v]

        if len(S) == 0:
            result = na_fill
        else:
            result /= len(S)

        return result

    @classmethod
    def sum(cls, S, x):
        result = 0.0
        for v in S:
            result += x[v]

        return result

    @classmethod
    def maximum(cls, S, x, init=-sys.float_info.max):
        result = init
        for v in S:
            result = max(result, x[v])

        return result

    @classmethod
    def minimum(cls, S, x, init=sys.float_info.max):
        result = init
        for v in S:
            result = min(result, x[v])

        return result

    @classmethod
    def hadamard(cls, S, x, init=1.0):
        result = init
        for v in S:
            result *= x[v]

        return result

    @classmethod
    def variance(cls, S, x, init=0.0, na_fill=0.0):
        result = init

        mean = 0.0
        sq_mean = 0.0
        for v in S:
            sq = x[v] * x[v]
            result += sq
            mean += x[v]
            sq_mean += sq
        if len(S) == 0:
            result = na_fill
        else:
            mean /= len(S)
            sq_mean /= len(S)
            result = sq_mean - mean**2

        return result

    @classmethod
    def lp_norm(cls, S, x, p=1, init=0.0):
        if p == 0:
            print("p must not be = 0")

        result = init
        for v in S:
            result += x[v]**p

        return result**(1 / p)

    @classmethod
    def l1_norm(cls, S, x, init=0.0):
        result = init
        for v in S:
            result += x[v]

        return result

    @classmethod
    def l2_norm(cls, S, x, init=0.0):
        result = init
        for v in S:
            result += x[v]**2

        return result**(1 / 2)

    @classmethod
    def rbf(cls, S, x, init=0.0, na_fill=0.0):
        result = init

        mean = 0.0
        sq_mean = 0.0
        for v in S:
            sq = x[v] * x[v]
            result += sq
            mean += x[v]
            sq_mean += sq
        if len(S) == 0:
            result = na_fill
        else:
            mean /= len(S)
            sq_mean /= len(S)
            var = sq_mean - mean**2
            if var == 0:
                result = na_fill
            else:
                try:
                    result = math.exp(-1 * result / var)
                except OverflowError:
                    result = 0.0

        return result
