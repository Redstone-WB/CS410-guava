import numpy as np
from src.Structure import Corpus
import lda


class LDA_utils(Corpus):
    """Sub Class of Corpus"""

    def __init__(self, corpus):
        self.corpus = corpus
        self.vocab_lda = tuple(corpus.Vocab)
        self.num_reviews_tot = self.get_num_reviews_tot()
        self.doc_term_matrix = self.create_doc_term_matrix()

    def get_num_reviews_tot(self):
        num_reviews_tot = 0
        for rest in (self.corpus.Restaurants):
            num_reviews_tot += rest.NumOfReviews
        return num_reviews_tot

    def create_doc_term_matrix(self):
        doc_term_tmp = np.zeros((self.num_reviews_tot, len(self.vocab_lda)))
        i = 0
        for rest in self.corpus.Restaurants:
            for review in rest.Reviews:
                for stn in review.Stns:
                    #             print(stn.stn)
                    for key in stn.stn:
                        value = stn.stn[key]
        #                 print(key)
                        doc_term_tmp[i][key-1] += value
        #                 if i % 100 == 0 :
        #                     print(i)
                i += 1
            print("Restaurant {0} Done!".format(rest.RestaurantID))
        doc_term_tmp = doc_term_tmp.astype(int)
        return doc_term_tmp

    def model_fit(self, n_topics, n_iter, random_state):
        self.model = lda.LDA(
            n_topics=n_topics, n_iter=n_iter, random_state=random_state)
        # model.fit_tranform(X) is also ok.
        self.model.fit(self.doc_term_matrix)
        self.topic_word = self.model.topic_word_

    def describe_topics(self, n_top_words):
        for i, topic_dist in enumerate(self.topic_word):
            self.topic_words = np.array(self.vocab_lda)[np.argsort(
                topic_dist)][:-(n_top_words+1):-1]
            print('Topic {}: {}'.format(i, ' '.join(self.topic_words)))

    # def kl_divergence(p, q):
    #     return np.sum(np.where(p != 0, p * np.log(p / q), 0))

    def describe_single_doc(self, idx):
        print("{} (top topic: {})".format(
            idx, self.model.doc_topic_[idx].argmax()))

    def describe_multiple_doc(self, start, end):
        for i in range(start, end):
            print("{} (top topic: {})".format(i,
                                              self.model.doc_topic_[i].argmax()))
