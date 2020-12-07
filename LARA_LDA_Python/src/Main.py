__author__ = 'zhangyin'

from LARA_LDA_Python.src.LRR import LRR
from CreateVocab import *
from Structure import *
from LARA_LDA_Python.src.lda_preproc import LDA_utils

def run_CreateVocab():
    # step 1: creating a vocab from data
    is_dev_mode = True
    cv_obj = CreateVocab(is_dev_mode)  # create an instance
    cv_obj.create_stopwords()  # create a list of stopwords
    suffix = "json"
    folder = "../data/yelp_sanitation_data/"
    cv_obj.read_data(folder, suffix)
    cv_obj.create_vocab()

    return cv_obj


def run_bootstrap():
    n_topics = 5    # 10
    n_iter = 100    # 1500

    cv_obj = run_CreateVocab()
    corpus = Corpus(cv_obj.corpus, cv_obj.Vocab, cv_obj.Count, cv_obj.VocabDict)

    # aspect_model = Bootstrapping()
    # loadfilepath = "./init_aspect_word.txt"
    # load_Aspect_Terms(aspect_model, loadfilepath, cv_obj.VocabDict)
    # Add_Aspect_Keywords(aspect_model, 5, 5, corpus)

    aspect_model = LDA_utils(corpus)
    aspect_model.model_fit(n_topics=n_topics, n_iter=n_iter, random_state=1)
    aspect_model.describe_topics(n_topics)

    model = LRR(aspect_model, corpus, 500, 1e-2, 5000, 1e-2, 2.0)
    model.em_estimation(10, 1e-4)


if __name__ == "__main__":
    run_bootstrap()
