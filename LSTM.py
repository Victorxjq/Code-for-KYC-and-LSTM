'''Trains a LSTM on the IMDB sentiment classification task.
The dataset is actually too small for LSTM to be of any advantage
compared to simpler, much faster methods such as TF-IDF + LogReg.
Notes:
- RNNs are tricky. Choice of batch size is important,
choice of loss and optimizer is critical, etc.
Some configurations won't converge.
- LSTM loss decrease patterns during training can be quite different
from what you see with CNNs/MLPs/etc.
'''
from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

import pandas as pd
import keras
import os
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from sklearn.feature_extraction.text import CountVectorizer
from Word2VecUtility import Word2VecUtility
from keras.layers import Dense, Dropout, Activation, Embedding
from keras.layers import LSTM, SimpleRNN, GRU
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import base_filter
# from keras.datasets import imdb
from sklearn.cross_validation import train_test_split

max_features = 20000
maxlen = 80  # cut texts after this number of words (among top max_features most common words)
batch_size = 32

print('Loading data...')
if __name__ == '__main__':
    train = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'labeledTrainData.tsv'), header=0, \
                    delimiter="\t", quoting=3)
    test = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'LSTMlabeledTestData.tsv'), header=0, delimiter="\t", \
                   quoting=3 )
    # train=rawdata[0:19999]
    print('The first review is:')
    print(train["review"][0])
# Initialize an empty list to hold the clean reviews
clean_train_reviews = []

# Loop over each review; create an index i that goes from 0 to the length
# of the movie review list

print ("Cleaning and parsing the training set movie reviews...\n")
for i in range( 0, len(train["review"])):
    clean_train_reviews.append(" ".join(Word2VecUtility.review_to_wordlist(train["review"][i], True)))


# ****** Create a bag of words from the training set
#
# print ("Creating the bag of words...\n")


# Initialize the "CountVectorizer" object, which is scikit-learn's
# bag of words tool.
vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 5000)

# fit_transform() does two functions: First, it fits the model
# and learns the vocabulary; second, it transforms our training data
# into feature vectors. The input to fit_transform should be a list of
# strings.
# train = train.iloc[:].values
# test = test.iloc[:].values

# ###################################
tk = Tokenizer(nb_words=2000, filters=base_filter(), lower=False)
tk.fit_on_texts(clean_train_reviews)
# print(clean_train_reviews[0])
#
train_data_features = tk.texts_to_sequences(clean_train_reviews)
# print("keras",train_data_features[0])

###################################
# ttrain_data_features = vectorizer.fit_transform(clean_train_reviews)
# print("sklearn",ttrain_data_features[0])
# Numpy arrays are easy to work with, so convert the result to an
# array
# train_data_features = vectorizer.fit_transform(clean_train_reviews)
# train_data_features = train_data_features.toarray()
# Create an empty list and append the clean reviews one by one
clean_test_reviews = []
#
print ("Cleaning and parsing the test set movie reviews...\n")
for i in range(0,len(test["review"])):
    clean_test_reviews.append(" ".join(Word2VecUtility.review_to_wordlist(test["review"][i], True)))

tk = Tokenizer(nb_words=2000, filters=base_filter(), lower=False, split=" ")
tk.fit_on_texts(clean_train_reviews)
#
test_data_features = tk.texts_to_sequences(clean_test_reviews)

# Get a bag of words for the test set, and convert to a numpy array
# test_data_features = vectorizer.fit_transform(clean_test_reviews)
# test_data_features = test_data_features.toarray()

X_train=train_data_features
y_train=train['sentiment']
X_test=test_data_features
y_test=test['sentiment']
print(y_test.value_counts())
# (X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=max_features)
print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')

print('Pad sequences (samples x time)')
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

print('Build model...')
model = Sequential()
model.add(Embedding(max_features, 128, dropout=0.2))
model.add(LSTM(128, dropout_W=0.2, dropout_U=0.2))  # try using a GRU instead, for fun
model.add(Dense(1))
model.add(Activation('sigmoid'))

# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Train...')
model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=15,
          validation_data=(X_test, y_test))
score, acc = model.evaluate(X_test, y_test,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)