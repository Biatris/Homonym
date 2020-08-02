from conll_df import conll_df
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

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
    tree_walk = i_dep(df[df["w"] == target_word].iloc[0], df)
    vectorizer = CountVectorizer()
    vectorizer.fit([" ".join(allpos)])
    #return df[df["w"] == target_word].iloc[0]
    z = vectorizer.transform([" ".join([w['p'] for w in tree_walk])]).todense().tolist()
    
    return pd.Series(z[0])
