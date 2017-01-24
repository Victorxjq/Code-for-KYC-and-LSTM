#!/usr/bin/env python
# *************************************** #

import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.ensemble import RandomForestClassifier
from Word2VecUtility import Word2VecUtility
import pandas as pd
import numpy
from sklearn.ensemble import AdaBoostClassifier
from sklearn import svm
from sklearn.cross_validation import cross_val_score
import numpy as np

if __name__ == '__main__':
    # train = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'labeledTrainData.tsv'), header=0, \
    #                 delimiter="\t", quoting=3)
    # test = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'testData.tsv'), header=0, delimiter="\t", \
    #                quoting=3 )
    # Data = pd.read_csv('C:\waitingforprocess\InputDataclean.csv',names=['ID','TEXT','CUSTOMER','ORIGINAL','LABEL'], encoding='latin1')
    # print('The first review is:')
    train = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'labeledTrainData.tsv'), header=0,
                        delimiter="\t", quoting=3)
    test = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'LSTMlabeledTestData.tsv'), header=0,
                        delimiter="\t", quoting=3)
    # print(train["review"][0])

    #Split Train and test set
    # train=Data.sample(frac=0.8,random_state=1)
    # train=Data.copy()
    # test=Data.drop(train.index)
    # print ('Download text data sets. If you already have NLTK datasets downloaded, just close the Python download window...')
    #nltk.download()  # Download text data sets, including stop words

    # Initialize an empty list to hold the clean reviews
    clean_train_text = []

    # Loop over each review; create an index i that goes from 0 to the length
    # of the movie review list

    print ("Cleaning and parsing the training set raw text...\n")
    for i in range( 0, len(train["review"])):
        clean_train_text.append(" ".join(Word2VecUtility.text_to_wordlist(train["review"][i], True)))

    # Initialize an empty list to hold the clean reviews
    clean_test_text = []

    # Loop over each review; create an index i that goes from 0 to the length
    # of the movie review list

    print ("Cleaning and parsing the training set raw text...\n")
    for i in range( 0, len(test["review"])):
        clean_test_text.append(" ".join(Word2VecUtility.text_to_wordlist(test["review"][i], True)))

    # numpy.save('Model/rawtext.npy',clean_train_text)

    # ****** Create a bag of words from the training set
    #
    print ("Creating the bag of words...\n")


    # Initialize the "CountVectorizer" object, which is scikit-learn's
    # bag of words tool.
    vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 5000)
    #tf-idf vector
    # vectorizer = TfidfVectorizer(analyzer="word", \
    #                              tokenizer=None, \
    #                              preprocessor=None, \
    #                              stop_words=None, \
    #                              max_features=5000)

    #hash vector
    # vectorizer = HashingVectorizer()
    # fit_transform() does two functions: First, it fits the model
    # and learns the vocabulary; second, it transforms our training data
    # into feature vectors. The input to fit_transform should be a list of
    # strings.
    train_data_features = vectorizer.fit_transform(clean_train_text)
    test_data_features = vectorizer.fit_transform(clean_test_text)
    # train_data_features = vectorizer.transform(clean_train_text)
    # Numpy arrays are easy to work with, so convert the result to an
    # array
    train_data_features = train_data_features.toarray()
    test_data_features = test_data_features.toarray()


    # ******* Train a random forest using the bag of words
    #
    # print ("Training the random forest (this may take a while)...")


    # Initialize a Random Forest classifier with 100 trees
    # forest = RandomForestClassifier(n_estimators = 300)
    # forest = svm.SVC(gamma=0.00001,C=150)
    # forest = AdaBoostClassifier(n_estimators=300)
    # Fit the forest to the training set, using the bag of words as
    # features and the sentiment labels as the response variable
    #
    # This may take a few minutes to run
    #Evaluate via accuracy
    # final=pd.DataFrame( train_data_features, train["LABEL"])
    # final.to_csv('C:\waitingforprocess\output\modelbagofwords.csv')
    # print(len(train_data_features))
    # print(len(train["LABEL"]))
    # result=np.load('Model/rawtext.npy')
    # print(result[0])
    # print(clean_train_text[0])
    # numpy.save('c:\label',train["LABEL"])
    numpy.save('train/bagofwords.npy',train_data_features)
    numpy.save('test/bagofwords.npy', test_data_features)

    # forest = forest.fit( train_data_features, train["LABEL"] )
    # scores_accuracy=cross_val_score(forest,train_data_features,train["LABEL"],cv=5)
    # scores_precision=cross_val_score(forest,train_data_features,train["LABEL"],cv=5,scoring='precision')
    # scores_recall=cross_val_score(forest,train_data_features,train["LABEL"],cv=5,scoring='recall')
    # print('accuracy:', scores_accuracy.mean())
    # print('precison:', scores_precision.mean())
    # print('recall:', scores_recall.mean())

