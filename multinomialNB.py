import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings("ignore")
xtrain_vectors = pd.read_pickle('xtrain_vectors.pkl')
xvalid_vectors = pd.read_pickle('xvalid_vectors.pkl')
xtrain_tfidf = pd.read_pickle('xtrain_tfidf.pkl')
xvalid_tfidf = pd.read_pickle('xvalid_tfidfs.pkl')
ytrain = pd.read_pickle('ytrain.pkl')
yvalid = pd.read_pickle('yvalid.pkl')

# Fitting a MultinomialNB on TFIDF
clf = MultinomialNB()
clf.fit(xtrain_tfidf, ytrain)

predictions = clf.predict(xvalid_tfidf)
print('MultinomialNB on TFIDF')
print ("f1_score :", np.round(f1_score(yvalid, predictions),5))

# Fitting a MultinomialNB on CountVec
clf = MultinomialNB()
clf.fit(xtrain_vectors, ytrain)

predictions = clf.predict(xvalid_vectors)
print('MultinomialNB on CountVectorizer')
print ("f1_score :", np.round(f1_score(yvalid, predictions),5))