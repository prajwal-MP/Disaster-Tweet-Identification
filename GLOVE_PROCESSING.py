import numpy as np
import pandas as pd
from tqdm import tqdm
import time
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
embeddings_index={}
with open('glove.6B.200d.txt','r',encoding='utf-8') as f:
    for line in f:
        values=line.split()
        word=values[0]
        vectors=np.asarray(values[1:],'float32')
        embeddings_index[word]=vectors
f.close()

print('Found %s word vectors.' % len(embeddings_index))

def sent2vec(s):
    words = str(s).lower()
    words = word_tokenize(words)
    words = [w for w in words if not w in stopwords.words('english')]
    words = [w for w in words if w.isalpha()]
    M = []
    for w in words:
        try:
            M.append(embeddings_index[w])
        except:
            continue
    M = np.array(M)
    v = M.sum(axis=0)
    if type(v) != np.ndarray:
        return np.zeros(200)
    return v / np.sqrt((v ** 2).sum())
xtrain = pd.read_pickle('xtrain.pkl')
xvalid = pd.read_pickle('xvalid.pkl')
xtrain_glove = np.array([sent2vec(x) for x in tqdm(xtrain)])
xvalid_glove = np.array([sent2vec(x) for x in tqdm(xvalid)])


pd.to_pickle(xtrain_glove,'xtrain_glove.pkl')
pd.to_pickle(xvalid_glove,'xvalid_glove.pkl')