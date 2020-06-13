import numpy as np
import pandas as pd
from tqdm import tqdm
import time
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

print("starting......")
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
sub = pd.read_csv('sample_submission.csv')

print(train.head())
print(test.head())

xtrain, xvalid, ytrain, yvalid = train_test_split(train.text, 
                                                  train.target,
                                                  random_state=42, 
                                                  test_size=0.1, shuffle=True)

print(xtrain.shape)
print(xvalid.shape)

def clean_text(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text


xtrain = xtrain.apply(lambda x: clean_text(x))
xvalid = xvalid.apply(lambda x: clean_text(x))
print(xtrain.head(3))

tokenizer1 = nltk.tokenize.WhitespaceTokenizer()
tokenizer2 = nltk.tokenize.TreebankWordTokenizer()
tokenizer3 = nltk.tokenize.WordPunctTokenizer()
tokenizer4 = nltk.tokenize.RegexpTokenizer(r'\w+')
tokenizer5 = nltk.tokenize.TweetTokenizer()

# appling tokenizer5
xtrain = xtrain.apply(lambda x: tokenizer5.tokenize(x))
xvalid = xvalid.apply(lambda x: tokenizer5.tokenize(x))
print(xtrain.head(3))

def remove_stopwords(text):
    words = [w for w in text if w not in stopwords.words('english')]
    return words


xtrain = xtrain.apply(lambda x : remove_stopwords(x))
xvalid = xvalid.apply(lambda x : remove_stopwords(x))

def combine_text(list_of_text):
    combined_text = ' '.join(list_of_text)
    return combined_text

xtrain = xtrain.apply(lambda x : combine_text(x))
xvalid = xvalid.apply(lambda x : combine_text(x))

# Stemmer
stemmer = nltk.stem.PorterStemmer()

# Lemmatizer
lemmatizer=nltk.stem.WordNetLemmatizer()

# Appling Lemmatizer
xtrain = xtrain.apply(lambda x: lemmatizer.lemmatize(x))
xvalid = xvalid.apply(lambda x: lemmatizer.lemmatize(x))

# Appling CountVectorizer()
count_vectorizer = CountVectorizer()
xtrain_vectors = count_vectorizer.fit_transform(xtrain)
xvalid_vectors = count_vectorizer.transform(xvalid)

# Appling TFIDF
tfidf = TfidfVectorizer(min_df=2, max_df=0.5, ngram_range=(1, 2), norm='l2')
xtrain_tfidf = tfidf.fit_transform(xtrain)
xvalid_tfidf = tfidf.transform(xvalid)

pd.to_pickle(xtrain,'xtrain.pkl')
pd.to_pickle(xvalid,'xvalid.pkl')
pd.to_pickle(xtrain_vectors,'xtrain_vectors.pkl')
pd.to_pickle(xvalid_vectors,'xvalid_vectors.pkl')
pd.to_pickle(xtrain_tfidf,'xtrain_tfidf.pkl')
pd.to_pickle(xvalid_tfidf,'xvalid_tfidfs.pkl')
pd.to_pickle(ytrain,'ytrain.pkl')
pd.to_pickle(yvalid,'yvalid.pkl')