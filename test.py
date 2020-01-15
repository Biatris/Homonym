from homonymfeatures import HomonymFeatures
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix
import fileinput

q = pd.read_csv("/Users/biatris/Desktop/Homonym/data/dulo_small.tsv", sep = "\t")

q['word_id'] = q['word_id'].apply(lambda x: x.split('_')[-1])
print(q['word_id'].unique())
my_hom = HomonymFeatures(q["sent"].values, q["word_id"].values, q["start"].values, q["stop"].values, target_word = "дуло")
#def __init__(self, sentences, target_pos_labels, target_word_start, target_word_stop, target_word, verbose = False):

my_hom.CreateTokensCorpus(verbose = False)
my_hom.CreatePosCorpus(verbose=False)
my_hom.CreateLemmaCorpus(verbose = False)
my_hom.CreateRpeCorpus(verbose = False)


#q.to_csv('POS_Test.csv')


#print(my_hom.fulldata_words.to_string())
pos_global_features = my_hom.CreatePosFeature(look = 5)
pos_local_features = my_hom.CreatePosFeature(look = 1)
lemma_features = my_hom.CreateLemmaFeature(look = 3)
nearest_pos_features = my_hom.CreateNearestPosFeature(look = 5)
rpe_features = my_hom.CreateRpeFeature(look = 5)

Xtot = pd.concat((pos_global_features, pos_local_features, lemma_features, nearest_pos_features, rpe_features), axis = 1)

#print(Xtot)
Y = my_hom.fulldata['target_pos_label'].values

res = []
rounds = 3

"""for k in range(rounds):
    X_train, X_test, y_train, y_test = train_test_split(Xtot, Y, test_size=0.2)
    mysgd = SGDClassifier(loss="hinge", penalty="elasticnet", tol=1e-5, n_jobs=-1, max_iter=10000, eta0=0.0000001, alpha=1e-3)
    mysgd.fit(X_train, y_train)
    res.append(mysgd.score( X_test, y_test ) )"""
X_aug = np.c_[Xtot, q[["sent"]].values]
X_train, X_test, y_train, y_test = train_test_split(X_aug, Y, test_size=0.2)
mysgd = SGDClassifier(loss="hinge", penalty="elasticnet", tol=1e-5, n_jobs=-1, max_iter=10000, eta0=0.0000001, alpha=1e-3)
X_test_sent = X_test[:,-1:][:,0]
X_test = X_test[:,:-1]
X_train = X_train[:,:-1]
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
score, var = round(np.mean(res),5), round(np.var(res), 5 )
baseline = my_hom.fulldata['target_pos_label'].value_counts().max() / len(my_hom.fulldata)
print(score, var, baseline, score/baseline)
res_df.style.set_properties(subset=['text'], **{'width': '3000px'})
pd.set_option('display.max_colwidth', -1)
res_df.to_html("myresult.html")
target_word = "дуло"
with fileinput.FileInput("/Users/biatris/Desktop/Homonym/myresult.html", inplace=True) as f:
    for line in f:
        print(line.replace(target_word, f" <span style='color: red'> {target_word} </span>"), end='')
