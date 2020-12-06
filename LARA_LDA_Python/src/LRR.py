import numpy as np

from Vector4Review import *


class LRR:
    def __init__(self, aspect_model, corpus, alpha_step, alpha_tol, beta_step, beta_tol, lamb):
        self.aspect_model = aspect_model
        self.corpus = corpus

        self.PI = 0.5
        self.K = 7

        self.m_model = None
        self.m_old_alpha = None   # in case optimization for alpha failed

        self.m_diag_beta = None
        self.m_g_beta = None
        self.m_beta = None

        self.m_diag_alpha = None
        self.m_g_alpha = None
        self.m_alpha = None
        self.m_alpha_cache = None

        self.m_alpha_step = alpha_step
        self.m_alpha_tol = alpha_tol
        self.m_beta_step = beta_step
        self.m_beta_tol = beta_tol
        self.m_lambda = lamb
        self.m_v = corpus.V
        self.m_k = len(aspect_model.Aspect_Terms)
        self.m_train_size = 0
        self.m_test_size = 0
        self.m_collection = []

        self.init_collection_from_corpus()

    def init_collection_from_corpus(self):
        for rest in self.corpus.Restaurants:
            for review in rest.Reviews:
                is_train = np.random.rand() < 0.75
                if is_train:
                    self.m_train_size += 1
                else:
                    self.m_test_size += 1
                ratings = np.random.rand(self.m_k)*5
                np.insert(ratings, 0, review.Overall)
                v4review = Vector4Review(review, ratings, is_train)
                self.m_collection.append(v4review)

    def init(self, voca_size):
        if len(self.m_collection) == 0:
            print("[Error]Load training data first!")
            return -1

    def em_estimation(self, max_iter, converge):
        iter = 0
        alpha_exp = 0
        alpha_cov = 0
        tag = ''
        diff = 10
        likelihood = 0
