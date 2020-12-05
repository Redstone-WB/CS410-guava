__author__ = 'zhangyin'

import re
import string
import timeit
from nltk.stem.porter import *
from nltk.corpus import stopwords
from nltk import FreqDist
import json
import os
import nltk
import numpy as np
stemmer = PorterStemmer()

###### data loading and parsing functions#########


def parse_to_sentence(content, stopwords):
    sent_word = []
    sentences = nltk.sent_tokenize(content)  # Sentence tokenization
    for sent in sentences:
        words = nltk.word_tokenize(sent)  # Word tokenization
        temp = [stemmer.stem(w.lower())  # word to lower letters -> stemming (if not punctuation)
                for w in words if w not in string.punctuation]
        temp2 = [v for v in temp if v not in stopwords]  # removing stop words
        if len(temp2) > 0:  # if not [], adding word to sent_word
            sent_word.append(temp2)
    return sent_word


def load_a_json_file(filename):
    with open(filename, encoding="ISO-8859-1") as data_file:
        data = json.load(data_file)
    return data


def load_all_json_files(jsonfolder, suffix):
    data = []

    def load_a_json_folder(folder, suffix):
        if not folder[-1] == "/":
            folder = folder+"/"
        # list all the files and sub folders under the Path
        # Get all the files and folders from the directory
        fs = os.listdir(folder)
        for f in fs:
            if not f.startswith("."):  # ignore files or folders start with period
                fpath = folder+f
                # if this is not a folder, that is, this is a file
                if not os.path.isdir(fpath):
                    # add data loading code
                    if fpath.split(".")[-1] == suffix:
                        with open(fpath, encoding="ISO-8859-1") as data_file:
                            data.append(json.load(data_file))
                else:
                    subfolder = fpath+"/"
                    # else this is a folder
                    load_a_json_folder(subfolder, suffix)
    load_a_json_folder(jsonfolder, suffix)
    return data


class CreateVocab:
    def create_stopwords(self):
        init_stopwords = [stemmer.stem(v) for v in stopwords.words('english')]
        additional_stopwords = ["'s", "...", "'ve",
                                "``", "''", "'m", '--', "'ll", "'d"]
        self.stopwords = additional_stopwords + init_stopwords

    def read_data(self, folder, suffix):
        # suffix="json"
        # folder="/Users/zhangyin/python projects/IR project/data/yelp mp1 data/"
        self.corpus = load_all_json_files(folder, suffix)

    def create_vocab(self):
        All_Contents = []
        i = 0
        for hotel in self.corpus:
            print("loading file No."+str(i+1))
            for review in hotel.get("Reviews"):
                s = []  # s : temporary list for one hotel (one file)
                for v in parse_to_sentence(review.get('Content'), self.stopwords):
                    s = v + s
                All_Contents = All_Contents + s
            i = i+1
        # nltk package provides a function that calculates frequencies of words
        term_freq = FreqDist(All_Contents)
        Vocab = []
        Count = []
        VocabDict = {}

        # term_freq is a dictionary
        for k, v in term_freq.items():
            if v > 5:  # only if term frequency is over 5, added to Vocab
                Vocab.append(k)  # appending key (word)
                Count.append(v)  # appending value (counts)

        # vocab sorting and re-saving
        self.Vocab = np.array(Vocab)[np.argsort(Vocab)].tolist()
        self.Count = np.array(Count)[np.argsort(Vocab)].tolist()
        # making a Vocab list to a dict (Vocab and index)
        self.VocabDict = dict(zip(self.Vocab, range(len(self.Vocab))))

    def save_to_file(self, savefilepath):
        # Saving corpus, Vocab, Count, VocabDict
        np.save(savefilepath, (self.corpus, self.Vocab,
                               self.Count, self.VocabDict))
        print("succeed saving to file "+savefilepath)

    def load_Vocab(self, loadfilepath):
        print("loading data from" + loadfilepath)
        return np.load(loadfilepath)

# savefilepath = "./output/yelp_mp1_corpus"
# loadfilepath = "./output/yelp_mp1_corpus.npy"


def get_top_p_tf(dict, p):
    temp = dict.copy()
    res = []
    for i in range(p):
        key = temp.max()
        v = temp.get(key)
        temp.pop(key)
        # res.append((key,v))
        res.append(key)
    return res
