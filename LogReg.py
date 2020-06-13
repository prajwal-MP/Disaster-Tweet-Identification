import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn import model_selection
import warnings
warnings.filterwarnings("ignore")
xtrain_vectors = pd.read_pickle('xtrain_vectors.pkl')
xvalid_vectors = pd.read_pickle('xvalid_vectors.pkl')
xtrain_tfidf = pd.read_pickle('xtrain_tfidf.pkl')
xvalid_tfidf = pd.read_pickle('xvalid_tfidfs.pkl')
ytrain = pd.read_pickle('ytrain.pkl')
yvalid = pd.read_pickle('yvalid.pkl')

clf = LogisticRegression(C=1.0)
clf.fit(xtrain_tfidf, ytrain)
scores = model_selection.cross_val_score(clf, xtrain_tfidf, ytrain, cv=5, scoring="f1")

predictions = clf.predict(xvalid_tfidf)
print('scores TFIDF',scores)
print ("f1_score :", np.round(f1_score(yvalid, predictions),5))

# Fitting a simple Logistic Regression on CountVec
clf = LogisticRegression(C=1.0)
clf.fit(xtrain_vectors, ytrain)
scores = model_selection.cross_val_score(clf, xtrain_vectors, ytrain, cv=5, scoring="f1")

predictions = clf.predict(xvalid_vectors)
print('scores CountVec',scores)
print ("f1_score :", np.round(f1_score(yvalid, predictions),5))