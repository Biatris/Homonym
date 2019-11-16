import pandas as pd
import nltk
import string
import numpy as np
from nltk.tokenize import word_tokenize
from collections.abc import Iterable
from rnnmorph.predictor import RNNMorphPredictor
from sklearn.feature_extraction.text import CountVectorizer



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

        self.target_word = target_word
        self.fulldata = pd.DataFrame(columns = ["sentence", "pos_label", "target_word_start", "target_word_stop"])

        self.fulldata["sentence"] = sentences
        self.fulldata["target_pos_label"] = target_pos_labels
        self.fulldata["target_word_start"] = target_word_start
        self.fulldata["target_word_stop"] = target_word_stop

        self.allpos = ["NOUN", "ADJF", "ADJS", "COMP", "VERB", "INFN", "PRTF", "PRTS", "GRND", "NUMR",
                  "ADVB", "NPRO", "PRED", "PREP", "CONJ", "PRCL", "INTJ", "ADP"]

    def CreateTokensCorpus(self, tokenizer = "nltk", verbose = False):
        if verbose:
            print("self.fulldata")
            print(self.fulldata)
        fulldata_words = pd.DataFrame(columns = ["sentence_num", "word_num", "token", "target_pos_label", "target_word_num"])
        if tokenizer == "nltk":
            for index, row in self.fulldata.iterrows():
                tokenizedsen = nltk.word_tokenize(row["sentence"]) ##TODO: add other pre-processing stuff
                tokenizedsen = [t.casefold() for t in tokenizedsen if not t in string.punctuation] # add user expansion of droplist
                if verbose:
                    print(f"processing sentence {index}")
                    print(tokenizedsen)
                word_num = 0
                ### TODO ### Handle case where target word is in sent multiple times
                target_word_num = tokenizedsen.index(self.target_word) # add an exception if it's not found
                for token in tokenizedsen:
                    fulldata_words = fulldata_words.append({"sentence_num": index, "word_num": word_num, "token": token, "target_pos_label": row["target_pos_label"], "target_word_num": target_word_num}, ignore_index = True)
                    word_num += 1
            if verbose:
                print(fulldata_words)
            self.fulldata_words = fulldata_words
            return fulldata_words
        else:
            raise NotImplementedError

    def CreatePosCorpus(self, look, language = "ru", verbose = False):
        if language == "ru":
            morph = RNNMorphPredictor(language="ru")
            vectorizer = CountVectorizer()
            vectorizer.fit([" ".join(self.allpos)])

            def create_pos_features(g):
                context_words = g[( (g["word_num"] - g["target_word_num"]).abs() <= look ) & ~(g["word_num"] == g["target_word_num"] )] ["token"].values
                context_pos = [list(p.pos for p in morph.predict(token)) for token in context_words]
                context_pos = [x for l in context_pos for x in l]
                return " ".join( context_pos )
            pos_sentences = self.fulldata_words.groupby("sentence_num").apply(create_pos_features)
            return pos_sentences.apply(lambda x: pd.Series(data = vectorizer.transform([x]).toarray() [0], index = vectorizer.get_feature_names() ) )
            
        elif language == "en":
            raise NotImplementedError
        else:
            NotImplementedError

    def CreateNearestPosFeature(self):
        pass

    def CreateLemmaCorpus(self):
        pass

    def CreateRelativePositionEncoding(self):
        pass

    def CreateWordNetFeatures(self):
        pass

    def CreateFeatures(self):
        pass
