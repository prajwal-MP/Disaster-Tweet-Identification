import numpy as np
import pandas as pd
from tqdm import tqdm
import time
import re
import string
import nltk
from sklearn.metrics import f1_score

from keras.models import Sequential
from keras.layers.recurrent import LSTM, GRU
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.layers import GlobalMaxPooling1D, Conv1D, MaxPooling1D, Flatten, Bidirectional, SpatialDropout1D
from keras.preprocessing import sequence, text
from keras.callbacks import EarlyStopping
xtrain = pd.read_pickle('xtrain.pkl')
xvalid = pd.read_pickle('xvalid.pkl')
ytrain = pd.read_pickle('ytrain.pkl')
yvalid = pd.read_pickle('yvalid.pkl')
embeddings_index={}
with open('glove.6B.200d.txt','r',encoding='utf-8') as f:
    for line in f:
        values=line.split()
        word=values[0]
        vectors=np.asarray(values[1:],'float32')
        embeddings_index[word]=vectors
f.close()

print('Found %s word vectors.' % len(embeddings_index))
token = text.Tokenizer(num_words=None)
max_len = 80

token.fit_on_texts(list(xtrain) + list(xvalid))
xtrain_seq = token.texts_to_sequences(xtrain)
xvalid_seq = token.texts_to_sequences(xvalid)

# zero pad the sequences
xtrain_pad = sequence.pad_sequences(xtrain_seq, maxlen=max_len)
xvalid_pad = sequence.pad_sequences(xvalid_seq, maxlen=max_len)

word_index = token.word_index

embedding_matrix = np.zeros((len(word_index) + 1, 200))
for word, i in tqdm(word_index.items()):
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

model = Sequential()
model.add(Embedding(len(word_index) + 1,
                     200,
                     weights=[embedding_matrix],
                     input_length=max_len,
                     trainable=False))
model.add(SpatialDropout1D(0.3))
model.add(LSTM(100, dropout=0.3, recurrent_dropout=0.3))

model.add(Dense(1000, activation='relu'))
model.add(Dropout(0.8))

model.add(Dense(1000, activation='relu'))
model.add(Dropout(0.8))

model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto')

model.fit(xtrain_pad, y=ytrain, batch_size=512, epochs=100, verbose=1, validation_data=(xvalid_pad, yvalid), callbacks=[earlystop])

predictions = model.predict(xvalid_pad)
predictions = np.round(predictions).astype(int)

print('LSTM')
print ("f1_score :", np.round(f1_score(yvalid, predictions),5))