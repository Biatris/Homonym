import pandas as pd
import nltk
import string
import numpy as np
import pymorphy2
from nltk.tokenize import word_tokenize
from collections.abc import Iterable
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from spacy.lang.ru import Russian
import re

class HomonymFeaturesException(Exception):
    def _init_ (self, *args):
        super()._init_(args)

class HomonymFeatures():
    def __init__(self, sentences, target_pos_labels, target_word_start, target_word_stop, target_word, verbose = False):

        if not isinstance(sentences, Iterable):
            raise HomonymFeaturesException("Sentences list is not iterable")
        if not isinstance(target_pos_labels, Iterable):
            raise HomonymFeaturesException("Pos labels list is not iterable")
        if not isinstance(target_word_start, Iterable):
            raise HomonymFeaturesException("Target word start list is not iterable")
        if not isinstance(target_word_stop, Iterable):
            raise HomonymFeaturesException("Target word stop list is not iterable")
        if not isinstance(target_word, str):
            raise HomonymFeaturesException("Target word is not a string")

        if not ((len(target_pos_labels) == len(sentences) ) and (len(target_word_start) == len(sentences)) and (len(target_word_stop) == len(sentences))):
            raise HomonymFeaturesException("Unmatched input lengths")


        self.fulldata = pd.DataFrame(columns = ["sentence", "pos_label", "target_word_start", "target_word_stop", 'target_pos_label'])

        self.fulldata["sentence"] = sentences
        self.fulldata["target_pos_label"] = target_pos_labels
        self.fulldata["target_word_start"] = target_word_start
        self.fulldata["target_word_stop"] = target_word_stop
        #self.fulldata['word_id'] = word_ids

        self.allpos = ["noun", "adjf", "adjs", "comp", "verb", "infn", "prtf", "prts", "grnd", "numb",
                        "advb", "npro", "pred", "prep", "conj", "prcl", "intj"]
        #self.fulldata['target_pos_label'] = self.fulldata['target_pos_label'].apply(lambda x: self.allpos.index(x.split('_')[1]))

        self.fulldata['target_pos_label'] = self.fulldata['target_pos_label'].apply(lambda x: self.allpos.index(x))

        self.target_word = target_word

    def CreateTokensCorpus(self, tokenizer = "nltk", verbose = False):
        if verbose:
            print("self.fulldata")
            print(self.fulldata)
        fulldata_words = pd.DataFrame(columns = ["sentence_num", "word_num", "token", "target_pos_label", "target_word_num"])
        if tokenizer == "nltk":
            target_pos = []
            for index, row in self.fulldata.iterrows():
                tokenizedsen = nltk.word_tokenize(re.sub(r'[^\s\w_]+', ' ', row["sentence"]) )  ##TODO: add other pre-processing stuff(including stripping all quotes etc prior to tokenizing)
                extra = ['«', '»', '…', '―', '...', '№', '❤', '``', '\'', '..']
                tokenizedsen = [t.casefold() for t in tokenizedsen if not t in string.punctuation and not t in extra and not t.isdigit()] # add user expansion of droplist
                if verbose:
                    print(f"processing sentence {index}")
                    print(tokenizedsen)
                word_num = 0
                ### TODO ### Handle case where target word is in sent multiple times
                print(tokenizedsen)
                target_word_num = tokenizedsen.index(self.target_word) # add an exception if it's not found
                #target_pos.append(self.allpos.index(row['word_id']) )
                for token in tokenizedsen:
                    fulldata_words = fulldata_words.append({"sentence_num": index, "word_num": word_num, "token": token, "target_pos_label": row["target_pos_label"], "target_word_num": target_word_num}, ignore_index = True)
                    word_num += 1
            if verbose:
                print(fulldata_words)
            self.fulldata_words = fulldata_words
            return fulldata_words
        else:
            raise NotImplementedError

    def CreatePosFeature(self, look, test = False, language = "ru", verbose = False):
        if language == "ru":
            if not test:
                vectorizer = CountVectorizer()
                vectorizer.fit([" ".join(self.allpos)])
                self.pos_vectorizer = vectorizer
            def create_pos_features(g):
                context_pos = g[( (g["word_num"] - g["target_word_num"]).abs() <= look ) & ~(g["word_num"] == g["target_word_num"] )]['POS1'].values
                return " ".join( context_pos )
            pos_sentences = self.fulldata_words.groupby("sentence_num").apply(create_pos_features)
            if test:
                return pos_sentences.apply(lambda x: pd.Series(data = self.pos_vectorizer.transform([x]).toarray() [0], index = vectorizer.get_feature_names() ) )
            else:
                return pos_sentences.apply(lambda x: pd.Series(data = vectorizer.transform([x]).toarray() [0], index = vectorizer.get_feature_names() ) )
        elif language == "en":
            raise NotImplementedError
        else:
            NotImplementedError

    def CreateNearestPosFeature(self, look, language = "ru", verbose = False):
        if language == "ru":
            def create_npos_features(g):
                forward_context_pos = g[( (g["word_num"] - g["target_word_num"]).abs() <= look ) & (g["word_num"] > g["target_word_num"] )]['POS1'].values
                backward_context_pos = g[( (g["word_num"] - g["target_word_num"]).abs() <= look ) & (g["word_num"] < g["target_word_num"] )]['POS1'].values
                pos_distances = []
                pos_indices = []
                for pos in self.allpos:
                    pos_forward_in = [(pos in l) for l in forward_context_pos]
                    pos_backward_in = [(pos in l) for l in backward_context_pos]
                    pos_indices.append(f"{pos}_forward")
                    pos_distances.append(pos_forward_in.index(True) if True in pos_forward_in else 0)
                    pos_indices.append(f"{pos}_backward")
                    pos_distances.append(pos_backward_in.index(True) if True in pos_backward_in else 0)
                distance_features = pd.Series(data = pos_distances, index = pos_indices)
                return distance_features
            return self.fulldata_words.groupby("sentence_num").apply(create_npos_features)
        elif language == "en":
            raise NotImplementedError
        else:
            NotImplementedError

    def CreatePosCorpus(self, language = 'ru', verbose = False):
        if language == "ru":
            morph = pymorphy2.MorphAnalyzer()
            tag = morph.TagClass
            def get_pos(q):
                x = morph.tag(q)[0].POS
                if x is not None:
                    return x.casefold()
                else:
                    return 'fail'
            self.fulldata_words['POS1'] = self.fulldata_words['token'].apply(get_pos)
        elif language == "en":
            raise NotImplementedError
        else:
            NotImplementedError
        return self.fulldata_words

    def CreateLemmaCorpus(self, language = "ru", verbose = False):
        if language == "ru":
            morph = pymorphy2.MorphAnalyzer()
            self.fulldata_words['normal_forms'] = self.fulldata_words['token'].apply(lambda q : morph.parse(q)[0].normal_form)

        elif language == "en":
            raise NotImplementedError
        else:
            NotImplementedError

        return self.fulldata_words

    def CreateLemmaFeature(self, look, language = "ru", test = False, verbose = False):
        if language == "ru":
            if not test:
                vectorizer = CountVectorizer()
                vectorizer.fit([" ".join(self.fulldata_words['normal_forms'].values)])
                self.lemma_vectorizer = vectorizer
            def create_lemma_features(g):
                context_lemmas = g[( (g["word_num"] - g["target_word_num"]).abs() <= look ) & ~(g["word_num"] == g["target_word_num"] )]['normal_forms'].values
                return " ".join( context_lemmas )

            lemma_sentences = self.fulldata_words.groupby("sentence_num").apply(create_lemma_features)

            if test:
                return lemma_sentences.apply(lambda x: pd.Series(data = self.lemma_vectorizer.transform([x]).toarray() [0], index = vectorizer.get_feature_names() ) )
            else:
                return lemma_sentences.apply(lambda x: pd.Series(data = vectorizer.transform([x]).toarray() [0], index = vectorizer.get_feature_names() ) )
        elif language == "en":
            raise NotImplementedError
        else:
            NotImplementedError

    def CreateRpeCorpus(self, language = "ru", verbose = False):
        self.fulldata_words['rpe'] = self.fulldata_words['POS1'] + (self.fulldata_words['target_word_num'] - self.fulldata_words['word_num']).astype(str)

        return self.fulldata_words

    def CreateRpeFeature(self, look, test = False, verbose = False):
        if not test:
            vectorizer = HashingVectorizer(n_features=2**8, ngram_range=(1,2))
            vectorizer.fit(self.fulldata_words['rpe'].values)
            self.rpe_vectorizer = vectorizer

        def create_rpe_features(g):
            rpe = g[( (g["word_num"] - g["target_word_num"]).abs() <= look ) & ~(g["word_num"] == g["target_word_num"] )]['rpe'].values
            return " ".join( rpe )
        rpe_sentences = self.fulldata_words.groupby("sentence_num").apply(create_rpe_features)

        if test:
            return rpe_sentences.apply(lambda x: pd.Series(data = self.rpe_vectorizer.transform([x]).toarray() [0], index = [f"rpe_hash_{k}" for k in range(vectorizer.n_features)] ) )
        else:
            return rpe_sentences.apply(lambda x: pd.Series(data = vectorizer.transform([x]).toarray() [0], index = [f"rpe_hash_{k}" for k in range(vectorizer.n_features)] ) )

    def CreateWordNetFeature(self, look, language = "ru", verbose = False):

        import conc_sim.py

        pass
