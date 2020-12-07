import numpy as np

from LARA_LDA_Python.src import Utilities


class LRR_Model:
    def __init__(self, k, v):
        self.m_k = k            # num of aspects
        self.m_v = v            # num of words
        self.m_mu = None        # prior for \alpha in each review
        self.m_sigma_inv = None # precision matrix (NOT covariance!)
        self.m_sigma = None     # only used for calculating inverse(\Sigma)
        self.m_beta = None      # word sentiment polarity matrix should have one bias term!
        self.m_delta = 1        # variance of overall rating prediction (\sigma in the manual)

        self.m_mu = np.zeros(self.m_k, dtype=np.float64)
        self.m_sigma = np.zeros((self.m_k, self.m_k), dtype=np.float64)
        self.m_sigma_inv = np.zeros((self.m_k, self.m_k), dtype=np.float64)
        self.m_beta = np.zeros((self.m_k, self.m_v + 1), dtype=np.float64)

    def init(self):
        for i in range(self.m_k):
            self.m_mu[i] = (2.0 * np.random.rand() - 1.0)
            self.m_sigma_inv[i][i] = 1.0
            self.m_sigma[i, i] = 1.0
            Utilities.randomize(self.m_beta[i])

    def calc_covariance(self, vct):
        _sum = 0
        for i in range(self.m_k):
            s = 0
            for j in range(self.m_k):
                s += vct[j] * self.m_sigma_inv[j][i]
            _sum += s * vct[i]
        return _sum

    def calc_det(self):
        return np.linalg.det(self.m_sigma)

    def calc_sigma_inv(self, scale):
        inv = np.linalg.inv(self.m_sigma)
        for i in range(self.m_k):
            for j in range(self.m_k):
                self.m_sigma_inv[i][j] = inv[i][j] * scale

