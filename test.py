from homonymfeatures import HomonymFeatures
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix
import fileinput
from sklearn.model_selection import ShuffleSplit

q = pd.read_csv("/Users/biatris/Desktop/Homonym/data/Pila.tsv", sep = "\t")

q['word_id'] = q['word_id'].apply(lambda x: x.split('_')[-1])
print(q['word_id'].unique())
my_hom = HomonymFeatures(q["sent"].values, q["word_id"].values, q["start"].values, q["stop"].values, target_word = "пила")

my_hom.CreateTokenCorpus(verbose = False)
my_hom.CreatePosCorpus(verbose=False)
my_hom.CreateLemmaCorpus(verbose = False)
my_hom.CreateRpeCorpus(verbose = False)
my_hom.CreateSynsetCorpus(verbose = True)

#print(my_hom.fulldata_words.to_string())
my_hom.fulldata_words.to_csv('/Users/biatris/Desktop/Homonym/data/fulldata_words_big_df.csv')
pos_global_features = my_hom.CreatePosFeature(look = 5)
pos_local_features = my_hom.CreatePosFeature(look = 1)
nearest_pos_features = my_hom.CreateNearestPosFeature(look = 3)
rpe_features = my_hom.CreateRpeFeature(look = 5)
prev_markov_prob_features = my_hom.CreatePrevProbFeature(verbose = False)
next_markov_prob_features = my_hom.CreateNextProbFeature(verbose = False)

#Xtot = pd.concat((prev_markov_prob_features, next_markov_prob_features), axis = 1)
#Xtot = pd.concat((pos_global_features, pos_local_features, lemma_features, nearest_pos_features, rpe_features), axis = 1)
#print(Xtot)
Y = my_hom.fulldata['target_pos_label'].values

res = []
rounds = 10

"""for k in range(rounds):
X_train, X_test, y_train, y_test = train_test_split(Xtot, Y, test_size=0.2)
mysgd = SGDClassifier(loss="hinge", penalty="elasticnet", tol=1e-5, n_jobs=-1, max_iter=10000, eta0=0.0000001, alpha=1e-3)
mysgd.fit(X_train, y_train)
res.append(mysgd.score( X_test, y_test ) )"""


#X_train, X_test, y_train, y_test = train_test_split(X_aug, Y, test_size=0.2)
rs = ShuffleSplit(n_splits=6, test_size=.2, random_state=0)
for train_index, test_index in rs.split(pos_global_features.values):
    my_hom.FitLemmaFeature(train_indices = train_index)
    my_hom.FitWordNetFeature(num_clues = 30, train_indices = train_index)
    lemma_features = my_hom.TransformLemmaFeature(look = 5)
    word_net_features_noun = my_hom.TransformWordNetFeature(pos_class = 'noun')
    word_net_features_verb = my_hom.TransformWordNetFeature(pos_class = 'verb')

    #print(pos_global_features.shape, pos_local_features.shape, nearest_pos_features.shape, rpe_features.shape, prev_markov_prob_features.shape, next_markov_prob_features.shape, word_net_features_noun.shape, word_net_features_verb.shape)
    Xtot = pd.concat((pos_global_features, pos_local_features, nearest_pos_features, rpe_features, lemma_features, prev_markov_prob_features, next_markov_prob_features, word_net_features_noun, word_net_features_verb), axis = 1)

    X_aug = np.c_[Xtot, q[["sent"]].values]
    print(q.max())
    X_train, X_test, y_train, y_test = X_aug[train_index,:], X_aug[test_index,:], Y[train_index], Y[test_index]

    mysgd = SGDClassifier(loss="hinge", penalty="elasticnet", tol=1e-5, n_jobs=-1, max_iter=10000, eta0=0.0000001, alpha=1e-3)
    X_test_sent = X_test[:,-1:][:,0]
    X_test = X_test[:,:-1]
    X_train = X_train[:,:-1]
    print(X_train.shape[0] / X_train.shape[1], X_train.shape[0], X_train.shape[1])

    mysgd.fit(X_train, y_train)
    res.append(mysgd.score( X_test, y_test ) )
    preds = mysgd.predict(X_test)
    res_df = pd.DataFrame()
    res_df["sent"] = X_test_sent
    res_df["ground_truth"] = y_test
    res_df["preds"] = preds
    res_df["success"] = (res_df["preds"] == res_df["ground_truth"])
Q = confusion_matrix(y_test, preds)
print(Q)
score, var = np.mean(res), np.var(res)
baseline = my_hom.fulldata['target_pos_label'].value_counts().max() / len(my_hom.fulldata)
score = round(score, 2)
var = round(var,7)
baseline = round(baseline, 2)
ratio = round(score/baseline, 2)
print(f"score: {score}, variance: {var}, baseline: {baseline}, score over baseline: {ratio}")
res_df.style.set_properties(subset=['text'], **{'width': '3000px'})
pd.set_option('display.max_colwidth', -1)
res_df.to_html("myresult.html")
target_word = "пила"
with fileinput.FileInput("/Users/biatris/Desktop/Homonym/results/myresult.html", inplace=True) as f:
    for line in f:
        print(line.replace(target_word, f" <span style='color: red'> {target_word} </span>"), end='')
