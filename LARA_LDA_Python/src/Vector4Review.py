import numpy as np

from LRR import *
from SpaVector import *


class Vector4Review:
    def __init__(self, review, ratings, isTrain):
        self.m_ID = review.ReviewID
        self.m_4train = isTrain
        self.m_ratings = ratings    # 0: input total rating, 1: aspect0 rating, 2: aspect1 rating, ...
        self.m_k = len(ratings)-1
        self.m_aspectV = []

        for k in range(self.m_k):
            spa_index = []  # index must start from 1
            spa_value = []
            for w in range(len(review.UniWord)):
                sum_row = sum(review.num_stn_aspect_word[k])
                if sum_row > 0:
                    spa_index.append(1 + len(spa_index))
                    spa_value.append(review.num_stn_aspect_word[k, w] / sum_row)
            spa_vector = SpaVector(spa_index, spa_value)
            self.m_aspectV.append(spa_vector)

        self.m_aspect_rating = np.zeros(self.m_k, dtype=np.float64)
        self.m_pred_cache = np.zeros(self.m_k, dtype=np.float64)
        self.m_alpha = np.zeros(self.m_k, dtype=np.float64)
        self.m_alpha_hat = np.zeros(self.m_k, dtype=np.float64)

    def get_aspect_size(self):
        if len(self.m_aspectV) == 1:
            return LRR.K
        else:
            return len(self.m_aspectV)
    #
    # def set_aspect(self, i, features):
    #     self.m_aspectV[i] = SpaVector(features)