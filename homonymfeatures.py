import pandas as pd
import nltk
import string
import numpy as np
import pymorphy2
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections.abc import Iterable
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import maru
from wiki_ru_wordnet import WikiWordnet
from nltk.corpus import wordnet as wn
from functools import reduce
from operator import add

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
        self.allpos = ['adj', 'adp', 'adv', 'conj', 'det', 'h', 'intj', 'noun', 'part', 'pron', 'punct', 'unkn', 'verb']
        self.fulldata['target_pos_label'] = self.fulldata['target_pos_label'].apply(lambda x: self.allpos.index(x))
        self.target_word = target_word

    def CreateTokenCorpus(self, tokenizer = "russian_tokenizer", verbose = False):
        if verbose:
            print("self.fulldata")
            print(self.fulldata)
        fulldata_words = pd.DataFrame(columns = ["sentence_num", "word_num", "token", "target_pos_label", "target_word_num"])
        if tokenizer == "nltk":
            target_pos = []
            for index, row in self.fulldata.iterrows():
                tokenizedsen = nltk.word_tokenize(re.sub(r'[^\s\w_]+', ' ', row["sentence"]) )
                russian_stopwords = stopwords.words("russian")
                tokenizedsen = [t.casefold() for t in tokenizedsen if not t in string.punctuation and not t in russian_stopwords and not t.isdigit()] # add user expansion of droplist

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
            target_pos = []
            for index, row in self.fulldata.iterrows():
                tokenizedsen = nltk.word_tokenize(re.sub(r'[^\s\w_]+', ' ', row["sentence"]), language = 'russian')
                russian_stopwords = stopwords.words("russian")
                tokenizedsen = [t.casefold() for t in tokenizedsen if not t in string.punctuation and not t in russian_stopwords and not t.isdigit()] # add user expansion of droplist
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
                return pos_sentences.apply(lambda x: pd.Series(data = self.pos_vectorizer.transform([x]).toarray() [0], index = vectorizer.get_feature_names() ))
            else:
                return pos_sentences.apply(lambda x: pd.Series(data = vectorizer.transform([x]).toarray() [0], index = vectorizer.get_feature_names() ) )
        elif language == "en":
            raise NotImplementedError
        else:
            raise NotImplementedError

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
            analyzer = maru.get_analyzer(tagger='linear', lemmatizer='dummy')

            def get_pos(q):
                analyzed = analyzer.analyze([q])
                return list(analyzed)[0].tag.pos.value.casefold() #print pos tag

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

    def FitLemmaFeature(self, train_indices, language = "ru", verbose = False):
        #fulldata_words_train = self.fulldata_words.iloc[train_indices]
        fulldata_words_train = self.fulldata_words[self.fulldata_words['sentence_num'].isin(train_indices)]
        if language == "ru":
            vectorizer = CountVectorizer()
            vectorizer.fit([" ".join(fulldata_words_train['normal_forms'].values)])
            self.lemma_vectorizer = vectorizer
            return vectorizer.get_feature_names()

        elif language == "en":
            raise NotImplementedError
        else:
            NotImplementedError

    def TransformLemmaFeature(self, look, language = "ru", verbose = False):

            def create_lemma_features(g):
                context_lemmas = g[( (g["word_num"] - g["target_word_num"]).abs() <= look ) & ~(g["word_num"] == g["target_word_num"] )]['normal_forms'].values
                return " ".join( context_lemmas )

            lemma_sentences = self.fulldata_words.groupby("sentence_num").apply(create_lemma_features)

            return lemma_sentences.apply(lambda x: pd.Series(data = self.lemma_vectorizer.transform([x]).toarray() [0], index = self.lemma_vectorizer.get_feature_names() ) )

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

    def CreatePrevProbFeature(self, language = "ru", verbose = False):
        if language == 'ru':
            pw1_w0 = pd.read_csv ("/Users/biatris/Desktop/Homonym/data/Pw1_w0.csv", index_col = 0)
            def create_cond_prob(g):

                if not g['target_word_num'].iloc[0] == 0:
                    previous_word = g[ g["word_num"] == (g["target_word_num"] - 1) ]
                    q = pw1_w0[ pw1_w0 ["w0"] == previous_word['POS1'].iloc[0] ]
                    q = q[['w1', 'prob']].set_index('w1')
                    s = q.reindex(index = self.allpos, fill_value = 0)
                    return s['prob']
                else:
                    q = pw1_w0[ pw1_w0 ["w0"] == 'punct']
                    q = q[['w1', 'prob']].set_index('w1')
                    s = q.reindex(index = self.allpos, fill_value = 0)
                    return s ['prob']
            return self.fulldata_words.groupby("sentence_num").apply(create_cond_prob)

        elif language == "en":
            raise NotImplementedError
        else:
            NotImplementedError

            return self.fulldata_words

    def CreateNextProbFeature(self, language = "ru", verbose = False):
        if language == 'ru':
            pw0_w1 = pd.read_csv ("/Users/biatris/Desktop/Homonym/data/Pw0_w1.csv", index_col = 0)
            def create_cond_prob(g):

                if not g['target_word_num'].iloc[0] == g['word_num'].max():
                    next_word = g[ g["word_num"] == (g["target_word_num"] + 1) ]
                    q = pw0_w1[ pw0_w1 ["w1"] == next_word['POS1'].iloc[0] ]
                    q = q[['w0', 'prob']].set_index('w0')
                    s = q.reindex(index = self.allpos, fill_value = 0)
                    return s['prob']
                else:
                    q = pw0_w1[ pw0_w1 ["w1"] == 'punct' ]
                    q = q[['w0', 'prob']].set_index('w0')
                    s = q.reindex(index = self.allpos, fill_value = 0)
                    return s ['prob']
            return self.fulldata_words.groupby("sentence_num").apply(create_cond_prob)

        elif language == "en":
            raise NotImplementedError
        else:
            NotImplementedError

            return self.fulldata_words

    def CreateSynsetCorpus(self, language = "ru", verbose = False):
        if language == "ru":
            wikiwordnet = WikiWordnet()
            self.fulldata_words['synset'] = self.fulldata_words['token'].apply(lambda w: wikiwordnet.get_synsets(w))


        elif language == "en":
            raise NotImplementedError
        else:
            NotImplementedError

        return self.fulldata_words

    def CreateHypernymCorpus(self, language = "ru", verbose = False):
        if language == "ru":
            wikiwordnet = WikiWordnet()
            self.fulldata_words['hypernym'] = self.fulldata_words['synset'][0].apply(lambda w: wikiwordnet.get_hypernyms(w))

        elif language == "en":
            raise NotImplementedError
        else:
            NotImplementedError

        return self.fulldata_words

    def FitWordNetFeature(self, train_indices, num_clues, language = "ru", verbose = False):
        morph = pymorphy2.MorphAnalyzer()
        all_clue_words = {}
        pos_classes = self.fulldata_words['target_pos_label'].unique()
        fulldata_words_train = self.fulldata_words[self.fulldata_words['sentence_num'].isin(train_indices)]
        analyzer = maru.get_analyzer(tagger='linear', lemmatizer='dummy')

        def get_pos(q):
            analyzed = analyzer.analyze([q])
            return list(analyzed)[0].tag.pos.value.casefold()
        for pos_class in pos_classes:
            all_syn = []
            class_words = fulldata_words_train[fulldata_words_train['target_pos_label'] == pos_class]
            class_words = class_words[class_words['token'] != self.target_word]
            class_words = class_words[class_words['POS1'].isin(['noun', 'verb', 'adj', 'adv'])]

            for s in class_words['synset'].values:
                for w in s:
                    for r in w.get_words():
                        word = r.definition().split('~')[0]
                        pos = get_pos(word)
                        if pos in ['noun', 'verb', 'adj']:
                            all_syn.append(morph.parse(word)[0].normal_form)
                            print(word, pos, morph.parse(word)[0].normal_form )
            clue_words = pd.Series(all_syn).value_counts().index.values
            all_clue_words[pos_class] = clue_words

        for w, v in all_clue_words.items():
            all_clue_words[w] = [y for y in all_clue_words[w] if y not in [x for k in all_clue_words.keys() for x in all_clue_words[k] if k != w]]
        all_clue_words = {k: v[:30] for k, v in all_clue_words.items()}
        self.all_clue_words = all_clue_words
        print(all_clue_words)
        return all_clue_words

    def TransformWordNetFeature(self, pos_class, language = "ru", verbose = False):
        pos_class = self.allpos.index(pos_class)
        c = CountVectorizer()
        c.fit(self.all_clue_words[pos_class])
        def create_clue_counts(g):
            return pd.Series(data = c.transform([' '.join (g['normal_forms'].values)]).toarray()[0], index = c.get_feature_names())

        return self.fulldata_words.groupby("sentence_num").apply(create_clue_counts)
