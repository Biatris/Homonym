from conll_df import conll_df
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import itertools
import logging
import transformers
import torch

def dep(r, df):
    try:
        dep_idx = df[df["w"]==r["w"]].iloc[0]["g"]
        return df[df["i"]==dep_idx].iloc[0]#["w"]
    except:
        return pd.Series()

def i_dep(r, df):
    res = [r]
    q = r
    #breaking cond
    while q.size != 0:
        q = dep(q, df)
        res.append(q)
    return res[:-1]

def create_row(r, target_word, allpos):
    file = open('temp/temp.win', 'w')
    file.write(r)
    file.close()
    df = conll_df("temp/temp.win", file_index=False)
    df = df.reset_index()
    #print(df)
    tree_walk = i_dep(df[df["w"] == target_word].iloc[0], df)
    vectorizer = CountVectorizer()
    vectorizer.fit([" ".join(allpos)])
    #return df[df["w"] == target_word].iloc[0]
    z = vectorizer.transform([" ".join([w['p'] for w in tree_walk])]).todense().tolist()

    return pd.Series(z[0])

def create_row_ranked(r, target_word, allpos, alldeps, max_depth = 3):
    file = open('temp/temp.win', 'w')
    file.write(r)
    file.close()
    df = conll_df("temp/temp.win", file_index=False)
    df = df.reset_index()
    print(df)
    tree_walk = i_dep(df[df["w"] == target_word].iloc[0], df)
    vectorizer = CountVectorizer(token_pattern = r"(?u)\b\w+\b")
    vectorizer_dep = CountVectorizer(token_pattern = r"(?u)\b\w+\b")
    vectorizer.fit([" ".join(allpos)])
    vectorizer_dep.fit([" ".join(alldeps)])

    #print('KIWIKIWIKIWIKIWIKIWIKIWIKWIKWIKWIWKWIKWI')

    q = list(itertools.chain.from_iterable(vectorizer.transform([w['p'] for w in tree_walk[:max_depth]]).todense().tolist()))
    q += [0] * (max_depth * len(allpos) - len(q))
    w = list(itertools.chain.from_iterable(vectorizer.transform([w['f'] for w in tree_walk[:max_depth]]).todense().tolist()))
    w += [0] * (max_depth * len(alldeps) - len(w))
    #print(len(q))
    return q, w

def token_deps(r, df, max_depth = 3):
    q = df.reset_index()
    token_deps = []
    s = r
    for k in range (max_depth):
        s = dep(s, q)
        print("WHERE IS THE W ERROR")
        print('s')
        print(s)
        print("WHERE IS THE W ERROR")
        if not s.empty:
            token_deps.append(s['w'])
    return token_deps

def find_index(target_word, wordpieces):
    #target_word = 'дуло'
    #[target_word.startswith(t) for t in wordpieces]
    for i, t in enumerate(wordpieces):
        if target_word.startswith(t):
            l = [t]
            k = 1
            while target_word.startswith(''.join(l)):
                l.append(wordpieces[i + k].replace('#', ''))
                k += 1
                print(l)
                if target_word == ''.join(l):
                    return slice(i, i + k)
    return False

def target_word_embedding(sentence, target_word, tokenizer, model):

    input_ids = tokenizer(sentence)["input_ids"]
    wordpieces = [tokenizer.decode([input_id]) for input_id in input_ids]

    target_indices = find_index(target_word, wordpieces)
    print(target_indices)
    input_ids_tensor = torch.tensor([input_ids])
    (token_encodings, final_encodings) = model(input_ids_tensor)
    targeted_encodings = token_encodings[:, target_indices, :].squeeze_()
    targeted_encodings_averaged = targeted_encodings.mean(0)

    return targeted_encodings_averaged
