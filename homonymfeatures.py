import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from collections.abc import Iterable





class HomonymFeaturesException(Exception):
    def _init_ (self, *args):
        super()._init_(args)

class HomonymFeatures():
    def __init__(self, sentences, target_pos_labels, target_word_start, target_word_stop, target_word, verbose = False):

        if not isinstance(sentences, Iterable):
            raise HomonymFeaturesException("Sentences list is not iterable")
        if not isinstance(pos_labels, Iterable):
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
        self.fulldata["target_pos_label"] = pos_labels
        self.fulldata["target_word_start"] = target_word_start
        self.fulldata["target_word_stop"] = target_word_stop

        pass

    def CreateTokensCorpus(self, tokenizer = "nltk"):

        fulldata_words = pd.DataFrame(columns = ["sentence_num", "word_num", "target_pos_label", "target_word_num"])
        if tokenizer == "nltk":
            for index, row in self.fulldata.itterrows():
                tokenizedsen = nltk.word_tokenize(row["sentence"])
                word_num = 0
                target_word_num = tokenizedsen.index(self.target_word) # add an exception if it's not found
                for token in tokenizedsen:
                    fulldata_words.append({"sentence_num": index, "word_num": word_num, "target_pos_label": row["target_pos_label"]}, ignore_index = True) 
                    word_num += 1

            else:
                raise NotImplementedError
        pass

    def CreatePosCorpus(self):
        pass

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
