import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from sklearn.utils import resample
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix,classification_report
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers import Flatten, Dropout

from hazm import Normalizer, word_tokenize
import re

def clean_persianText(txt):
    normalizer = Normalizer()
    txt = normalizer.character_refinement(txt)
    txt = normalizer.affix_spacing(txt)
    txt = normalizer.punctuation_spacing(txt)
    txt = txt.replace('.', '')
    txt = normalizer.normalize(txt)
    return txt

data = pd.read_csv('data.csv')
data = data[['text','sentiment']]

data['text'] = data['text'].apply(lambda x: x.lower())
data['text'] = data['text'].apply(lambda x: clean_persianText(x))

print(data[ data['sentiment'] == 'positive'].size)
print(data[ data['sentiment'] == 'negative'].size)

for idx,row in data.iterrows():
    row[0] = row[0].replace('rt','')
data.head()

max_fatures = 2000
tokenizer = Tokenizer(num_words=max_fatures, split=' ')
tokenizer.fit_on_texts(data['text'].values)
X = tokenizer.texts_to_sequences(data['text'].values)
X = pad_sequences(X)
X[:2]

embed_dim = 300

model = Sequential()
model.add(Embedding(max_fatures, embed_dim,input_length = X.shape[1]))
model.add(Conv1D(128, 7, activation='relu',padding='same'))
model.add(MaxPooling1D())
model.add(Conv1D(256, 5, activation='relu',padding='same'))
model.add(MaxPooling1D())
model.add(Conv1D(512, 3, activation='relu',padding='same'))
model.add(MaxPooling1D())
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2,activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
print(model.summary())

Y = pd.get_dummies(data['sentiment']).values
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.20, random_state = 42)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)

batch_size = 128
model.fit(X_train, Y_train, epochs = 30, batch_size=batch_size, verbose = 1)