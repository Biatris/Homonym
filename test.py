from homonymfeatures import HomonymFeatures
import pandas as pd

q = pd.read_csv("/Users/biatris/Desktop/Homonym/data/test_data.tsv", sep = "\t")


my_hom = HomonymFeatures(q["sent"].values, q["word_id"].values, q["start"].values, q["stop"].values, target_word = "дуло")
#def __init__(self, sentences, target_pos_labels, target_word_start, target_word_stop, target_word, verbose = False):

my_hom.CreateTokensCorpus(verbose = True)
my_hom.CreatePosCorpus(verbose=True)
my_hom.CreateLemmaCorpus(verbose = True)
my_hom.CreateRpeCorpus(verbose = True)

my_hom.CreatePosFeature(look = 5, verbose = True)
r = my_hom.CreateLemmaFeature(look = 5, verbose = True)
print(r)
q = my_hom.CreateNearestPosFeature(look = 5, verbose = True)
q.to_csv('POS_Test.csv')
z = my_hom.CreateRpeFeature(look = 3, verbose = True)
print(z)
print(my_hom.fulldata_words.to_string())
