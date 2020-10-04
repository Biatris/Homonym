from homonymfeatures import HomonymFeatures
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix
import fileinput
from sklearn.model_selection import ShuffleSplit
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier
from sys import exit

q = pd.read_csv("/Users/nataliatyulina/Desktop/Homonym/data/dulo_med.tsv", sep = "\t")
target_word = 'дуло' #'пила'#'печь'"дуло"
q = q[q["word"] == target_word]
q['word_id'] = q['word_id'].apply(lambda x: x.split('_')[-1])
print(q['word_id'].unique())
my_hom = HomonymFeatures(q["sent"].values, q["word_id"].values, q["start"].values, q["stop"].values, target_word = target_word)

my_hom.CreateTokenCorpus(verbose = True)
#my_hom.CreatePosCorpus(verbose=False)
#print(my_hom.fulldata_words)
alltrees = my_hom.CreateFunctionRPECorpus(verbose = True, language='ru')
print(alltrees)
my_hom.CreateLemmaCorpus(verbose = False)
my_hom.CreateRpeCorpus(verbose = False)
my_hom.CreateSynsetCorpus(verbose = True)
#my_hom.CreateDepFeature()
#print(my_hom.fulldata_words.to_string())
#my_hom.fulldata_words.to_csv('/Users/nataliatyulina/Desktop/Homonym/data/fulldata_words_big_df.csv')
pos_global_features = my_hom.CreatePosFeature(look = 30)
pos_local_features = my_hom.CreatePosFeature(look = 3)
function_rpe_features = my_hom.CreateFunctionRpeFeature(look = 10)
nearest_pos_features = my_hom.CreateNearestPosFeature(look = 5)
rpe_features = my_hom.CreateRpeFeature(look = 10)
ud_features, ud_dep_features = my_hom.CreateDepFeature(verbose = True)
sent2vec_features = my_hom.CreateSent2VecFeature(verbose = False)
print(sent2vec_features)

#prev_markov_prob_features = my_hom.CreatePrevProbFeature(verbose = False)
#next_markov_prob_features = my_hom.CreateNextProbFeature(verbose = False)

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
feat_importance = pd.DataFrame()
#perm_feat_importance = pd.DataFrame()

pos_global_features.columns = [("9_pos_global_features_" + str(q)) for q in pos_global_features.columns ]
pos_local_features.columns = [("10_pos_local_features_" + str(q)) for q in pos_local_features.columns ]
rpe_features.columns = [("11_rpe_features_" + str(q)) for q in rpe_features.columns ]
nearest_pos_features.columns = [("12_nearest_pos_features_" + str(q)) for q in nearest_pos_features.columns ]
function_rpe_features.columns = [("13_function_rpe_features_" + str(q)) for q in function_rpe_features.columns ]
#sent2vec_features.columns = [("14_function_rpe_features_" + str(q)) for q in sent2vec_features.columns ]
rs = ShuffleSplit(n_splits=6, test_size=.2, random_state=0)
split_n = 0
for train_index, test_index in rs.split(pos_global_features.values):
    my_hom.FitLemmaFeature(train_indices = train_index)
    my_hom.FitWordNetFeature(num_clues = 10, train_indices = train_index)
    lemma_features = my_hom.TransformLemmaFeature(look = 20)
    word_net_features_noun = my_hom.TransformWordNetFeature(pos_class = 'noun')
    word_net_features_verb = my_hom.TransformWordNetFeature(pos_class = 'verb')

    #ud_features.columns = [("1_ud_features_" + str(q)) for q in ud_features.columns ]

    word_net_features_noun.columns = [("6_word_net_features_noun_" + str(q)) for q in word_net_features_noun.columns ]
    word_net_features_verb.columns = [("7_word_net_features_verb_" + str(q) )for q in word_net_features_verb.columns ]
    lemma_features.columns = [("8_lemma_features_" + str(q)) for q in lemma_features.columns ]

    print("ud_features")
    print(ud_features)
    print("ud_dep_features")
    print(ud_dep_features)
    print('word_net_features_noun')
    print(word_net_features_noun)
    print('word_net_features_verb')
    print(word_net_features_verb)
    print('lemma_features')
    print(lemma_features)
    print('rpe_features')
    print(rpe_features)
    print('pos_global_features')
    print(pos_global_features)
    print('pos_local_features')
    print(pos_local_features)
    print('nearest_pos_features')
    print(nearest_pos_features)
    print('function_rpe_features')
    print(function_rpe_features)
    print('sent2vec_features')
    print(sent2vec_features)
    #pos_local_features.columns = ["pos_local_features_" + str(q) for q in pos_local_features.columns ]
    #print(pos_global_features.shape, pos_local_features.shape, nearest_pos_features.shape, rpe_features.shape, prev_markov_prob_features.shape, next_markov_prob_features.shape, word_net_features_noun.shape, word_net_features_verb.shape)
    Xtot = pd.concat((pos_global_features, pos_local_features, nearest_pos_features, rpe_features, lemma_features, word_net_features_noun, word_net_features_verb, ud_features, ud_dep_features, function_rpe_features, sent2vec_features), axis = 1)
    ###Xtot = sent2vec_features
    #Xtot = pd.concat((pos_global_features, lemma_features, word_net_features_noun, word_net_features_verb, ud_features, ud_dep_features, function_rpe_features)), axis = 1)
    #Xtot = ud_features
    #prev_markov_prob_features, next_markov_prob_features
    Xtot = (Xtot - Xtot.mean())
    Xtot = (Xtot/(Xtot.max()-Xtot.min())).fillna(0)
    corr = Xtot.corr()
    sns_heat = sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns)
    fig = sns_heat.get_figure()
    corr.to_csv(f"{split_n}_corr.csv")
    plt.show(block = False)
    fig.savefig(f"{split_n}_corr.png")
    plt.clf()
    #plt.savefig(f"{split_n}_corr.png")
    split_n += 1
    X_aug = np.c_[Xtot, q[["sent"]].values]
    print(q.max())
    X_train, X_test, y_train, y_test = X_aug[train_index,:], X_aug[test_index,:], Y[train_index], Y[test_index]

    #mysgd = SGDClassifier(loss="hinge", penalty="elasticnet", tol=1e-5, n_jobs=-1, max_iter=10000, eta0=0.0000001, alpha=1e-3)
    clf = RandomForestClassifier(random_state=0)
    X_test_sent = X_test[:,-1:][:,0]
    X_test = X_test[:,:-1]
    X_train = X_train[:,:-1]
    print(X_train.shape[0] / X_train.shape[1], X_train.shape[0], X_train.shape[1])


    #mysgd.fit(X_train, y_train)
    clf.fit(X_train, y_train)
    #print(mysgd.feature_importances_)
    feat_importance = feat_importance.append(dict(zip(Xtot.columns, clf.feature_importances_)), ignore_index = True)
    #feat_importance = feat_importance.append(dict(zip(Xtot.columns, mysgd.coef_[0])), ignore_index = True)

    #r = permutation_importance(mysgd, X_train, y_train, n_repeats=30, random_state=1)
    #print(r)
    #print(type(r))
    #perm_feat_importance = perm_feat_importance.append(pd.Series(data = r.importances_mean, index = Xtot.columns), ignore_index = True)
    #res.append(mysgd.score( X_test, y_test ) )
    res.append(clf.score( X_test, y_test ) )
    #preds = mysgd.predict(X_test)
    preds = clf.predict(X_test)
    res_df = pd.DataFrame()
    res_df["sent"] = X_test_sent
    res_df["ground_truth"] = y_test
    res_df["preds"] = preds
    res_df["success"] = (res_df["preds"] == res_df["ground_truth"])


#pl = sns.heatmap(feat_importance, annot=True) #X_train, y_train, n_repeats=30
feat_importance = feat_importance.fillna(0).abs()
#perm_feat_importance = perm_feat_importance.fillna(0).abs()
Xtot['y'] = Y
Xtot.to_csv('all_features.csv')
print('KIWIKIWIKIWIKIWIKWIKWIKWIWKIWKWI')
#sns.heatmap(feat_importance, xticklabels=True, yticklabels=True, center = 0)
#plt.show()
feat_importance.to_csv('feat_importance.csv')
#perm_feat_importance.to_csv('perm_feat_importance.csv')

#feat_importance = feat_importance.groupby(feat_importance.columns.str[0],axis=1).sum().sum()

#feat_importance = feat_importance.groupby(feat_importance.columns.str.split('_').str.get(0),axis=1).sum().sum()[['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13']]
feat_importance = feat_importance.groupby(feat_importance.columns.str.split('_').str.get(0),axis=1).max().max()[['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14']]

#perm_feat_importance = perm_feat_importance.groupby(perm_feat_importance.columns.str.split('_').str.get(0),axis=1).sum().sum()[['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']]

#perm_feat_importance = perm_feat_importance.groupby(perm_feat_importance.columns.str[0],axis=1).sum().sum()
print('NEWESTSUPERKIWINEWESTSUPERKIWINEWESTSUPERKIWINEWESTSUPERKIWINEWESTSUPERKIWINEWESTSUPERKIWI')
print(feat_importance.index)
#feat_importance.index = ['pos_global_features', 'pos_local_features', 'nearest_pos_features', 'rpe_features', 'lemma_features', 'word_net_features_noun', 'word_net_features_verb', 'ud_features_0', 'ud_features_1', 'ud_features_2', 'ud_dep_features_3', 'ud_dep_features_4', 'ud_dep_features_5']
feat_importance.index = ['ud_features_1', 'ud_features_2', 'ud_dep_features_3', 'ud_dep_features_4', 'ud_dep_features_5', 'word_net_features_noun', 'word_net_features_verb', 'lemma_features', 'pos_global_features', 'pos_local_features', 'rpe_features', 'nearest_pos_features', 'function_rpe_features', 'sent2vec_features' ]
#feat_importance.index = ['ud_features_1', 'ud_features_2', 'ud_dep_features_3', 'ud_dep_features_4', 'ud_dep_features_5', 'word_net_features_noun', 'word_net_features_verb', 'lemma_features', 'pos_global_features', 'function_rpe_features' ]

#perm_feat_importance.index = ['ud_features_0', 'ud_features_1', 'ud_features_2', 'ud_dep_features_3', 'ud_dep_features_4', 'ud_dep_features_5', 'word_net_features_noun', 'word_net_features_verb', 'lemma_features', 'pos_global_features', 'pos_local_features', 'rpe_features', 'nearest_pos_features' ]
feat_importance.plot.bar()
plt.show()
#perm_feat_importance.plot.bar()
#plt.show()
#plt.savefig('output.png')
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
res_df.to_html(f"/Users/nataliatyulina/Desktop/Homonym/data/{target_word}_result.html")
target_word = target_word
with fileinput.FileInput(f"/Users/nataliatyulina/Desktop/Homonym/data/{target_word}_result.html", inplace=True) as f:
    for line in f:
        print(line.replace(target_word, f" <span style='color: red'> {target_word} </span>"), end='')
