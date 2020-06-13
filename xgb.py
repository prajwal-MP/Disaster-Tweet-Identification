import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings("ignore")
xtrain_vectors = pd.read_pickle('xtrain_vectors.pkl')
xvalid_vectors = pd.read_pickle('xvalid_vectors.pkl')
xtrain_tfidf = pd.read_pickle('xtrain_tfidf.pkl')
xvalid_tfidf = pd.read_pickle('xvalid_tfidfs.pkl')
ytrain = pd.read_pickle('ytrain.pkl')
yvalid = pd.read_pickle('yvalid.pkl')

# Fitting a simple xgboost on TFIDF
clf = xgb.XGBClassifier(max_depth=5, n_estimators=300, colsample_bytree=0.8, 
                        subsample=0.5, nthread=10, learning_rate=0.1)
clf.fit(xtrain_tfidf.tocsc(), ytrain)
predictions = clf.predict(xvalid_tfidf.tocsc())

print('XGBClassifier on TFIDF')
print ("f1_score :", np.round(f1_score(yvalid, predictions),5))

# Fitting a simple xgboost on CountVec
clf = xgb.XGBClassifier(max_depth=5, n_estimators=300, colsample_bytree=0.8, 
                        subsample=0.5, nthread=10, learning_rate=0.1)
clf.fit(xtrain_vectors, ytrain)

predictions = clf.predict(xvalid_vectors)
print('XGBClassifier on CountVectorizer')
print ("f1_score :", np.round(f1_score(yvalid, predictions),5))