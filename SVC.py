import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings("ignore")
xtrain_vectors = pd.read_pickle('xtrain_vectors.pkl')
xvalid_vectors = pd.read_pickle('xvalid_vectors.pkl')
xtrain_tfidf = pd.read_pickle('xtrain_tfidf.pkl')
xvalid_tfidf = pd.read_pickle('xvalid_tfidfs.pkl')
ytrain = pd.read_pickle('ytrain.pkl')
yvalid = pd.read_pickle('yvalid.pkl')
# Fitting a LinearSVC on TFIDF
clf = make_pipeline(StandardScaler(with_mean=False),SVC(gamma='auto'))
clf.fit(xtrain_tfidf, ytrain)

predictions = clf.predict(xvalid_tfidf)
print('SVC on TFIDF')
print ("f1_score :", np.round(f1_score(yvalid, predictions),5))

# Fitting a LinearSVC on CountVec
clf = make_pipeline(StandardScaler(with_mean=False),SVC(gamma='auto'))
clf.fit(xtrain_vectors, ytrain)

predictions = clf.predict(xvalid_vectors)
print('SVC on CountVectorizer')
print ("f1_score :", np.round(f1_score(yvalid, predictions),5))