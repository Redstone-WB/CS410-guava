import numpy as np

from LRR import *
from SpaVector import *
from LARA_LDA_Python.src import Utilities

class Vector4Review:
    def __init__(self, review, ratings, isTrain, aspect_model):
        self.m_ID = review.ReviewID
        self.m_4train = isTrain
        self.m_ratings = ratings    # 0: input total rating, 1: aspect0 rating, 2: aspect1 rating, ...
        self.m_k = len(ratings)-1
        self.m_aspectV = None

        review.num_stn_aspect_word = np.zeros((self.m_k, review.NumOfUniWord))
        review.num_stn_aspect = np.zeros(self.m_k)
        review.num_stn_word = np.zeros(review.NumOfUniWord)
        review.num_stn = len(review.Stns)

        for stn in review.Stns:
            aspect_score = np.zeros(self.m_k)
            for i in range(self.m_k):
                for w in stn.stn.keys():
                    topic_word_score = aspect_model.topic_word[i][w]
                    aspect_score[i] += topic_word_score
            s_label = np.where(aspect_score == np.max(aspect_score))[0].tolist()
            stn.label = s_label  # with tie

        for stn in review.Stns:
            if stn.label != -1:  ## remove unlabeled stns
                review.num_stn = review.num_stn + 1
                for l in stn.label:
                    review.num_stn_aspect[l] = review.num_stn_aspect[l] + 1
                    for w in stn.stn.keys():
                        z = np.where(w == review.UniWord)[0]  # index
                        review.num_stn_word[z] = review.num_stn_word[z] + 1
                    for l in stn.label:
                        for w in stn.stn.keys():
                            z = np.where(w == review.UniWord)[0]  # index
                            review.num_stn_aspect_word[l, z] = review.num_stn_aspect_word[l, z] + 1

        self.m_aspectV = np.zeros(self.m_k, dtype=np.float64)
        aspect_word_freq = {}
        for stn in review.Stns:
            aspect_id = stn.label[0]

            word_freq = None
            if aspect_id not in aspect_word_freq:
                word_freq = {}
                aspect_word_freq[aspect_id] = word_freq
            else:
                word_freq = aspect_word_freq[aspect_id]

            for word_id in stn.stn.keys():
                if word_id not in word_freq:
                    word_freq[word_id] = 0
                word_freq[word_id] += 1

        self.m_aspectV = []
        for i in range(self.m_k):
            if i in aspect_word_freq:
                word_freq = aspect_word_freq[i]
                spa_index = []  # index must start from 1
                spa_value = []
                for word_info in sorted(word_freq.items()):
                    word_id = word_info[0]
                    word_cnt = word_info[1]
                    spa_index.append(word_id + 1)
                    spa_value.append(word_cnt)
                spa_vector = SpaVector(spa_index, spa_value)
            else:
                spa_vector = SpaVector([], [])
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

    # apply model onto each aspect
    def get_aspect_rating(self, beta):
        for i in self.m_k:
            self.m_pred_cache[i] = self.m_aspectV[i].dot_product(beta[i])

    def get_doc_length(self):
        _sum = 0
        for vct in self.m_aspectV:
            _sum += vct.L1Norm()
        return _sum

    def normalize(self):
        norm = self.get_doc_length()
        for i in range(self.m_k):
            vct = self.m_aspectV[i]

            aSize = vct.L1Norm()
            vct.normalize(aSize)
            self.m_alpha_hat[i] = np.random.rand() + np.log(aSize / norm)

        norm = Utilities.exp_sum(self.m_alpha_hat)
        for i in range(self.m_k):
            self.m_alpha = np.exp(self.m_alpha_hat[i])/norm

