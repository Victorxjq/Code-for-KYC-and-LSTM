#!/usr/bin/env python


# ****** Read the two training sets and the test set
#
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import os
from sklearn.cross_validation import cross_val_score
from nltk.corpus import stopwords
import nltk.data
import logging
import numpy as np  # Make sure that numpy is imported
from gensim.models import Word2Vec
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

from Word2VecUtility import Word2VecUtility


# ****** Define functions to create average word vectors
#

def makeFeatureVec(words, model, num_features):
    # Function to average all of the word vectors in a given
    # paragraph
    #
    # Pre-initialize an empty numpy array (for speed)
    featureVec = np.zeros((num_features,),dtype="float32")
    #
    nwords = 0.
    #
    # Index2word is a list that contains the names of the words in
    # the model's vocabulary. Convert it to a set, for speed
    index2word_set = set(model.wv.index2word)
    #
    # Loop over each word in the review and, if it is in the model's
    # vocaublary, add its feature vector to the total
    for word in words:
        if word in index2word_set:
            nwords = nwords + 1.
            featureVec = np.add(featureVec,model[word])
    #
    # Divide the result by the number of words to get the average
    featureVec = np.divide(featureVec,nwords)
    return featureVec


def getAvgFeatureVecs(texts, model, num_features):
    # Given a set of reviews (each one a list of words), calculate
    # the average feature vector for each one and return a 2D numpy array
    #
    # Initialize a counter
    counter = 0.
    #
    # Preallocate a 2D numpy array, for speed
    FeatureVecs = np.zeros((len(texts), num_features), dtype="float32")
    #
    # Loop through the texts
    for text in texts:
       # #
       # # Print a status message every 1000th review
       # if counter%1000. == 0.:
       #     print ("Review %d of %d" % (counter, len(texts)))
       # #
       # Call the function (defined above) that makes average feature vectors
       FeatureVecs[counter] = makeFeatureVec(text, model,
           num_features)
       #
       # Increment the counter
       counter = counter + 1.
    return FeatureVecs


def getCleanText(texts):
    clean_texts = []
    for text in texts["review"]:
        clean_texts.append(Word2VecUtility.text_to_wordlist(text, remove_stopwords=True))
    return clean_texts



if __name__ == '__main__':

    # Read data from files
    train = pd.read_csv( os.path.join(os.path.dirname(__file__), 'data', 'labeledTrainData.tsv'), header=0, delimiter="\t", quoting=3 )
    test = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'LSTMlabeledTestData.tsv'), header=0, delimiter="\t", quoting=3 )
    # unlabeled_train = pd.read_csv( os.path.join(os.path.dirname(__file__), 'data', "unlabeledTrainData.tsv"), header=0,  delimiter="\t", quoting=3 )
    # Data = pd.read_csv('C:\waitingforprocess\InputDataclean.csv', names=['ID', 'TEXT', 'CUSTOMER', 'ORIGINAL','LABEL'],
    #                    encoding='latin1')
    # # Verify the number of reviews that were read (100,000 in total)
    # print ("Read %d labeled train reviews, %d labeled test reviews, " \
    #  "and %d unlabeled reviews\n" % (train["review"].size,
    #  test["review"].size, unlabeled_train["review"].size ))
    # train=Data.copy()


    # Load the punkt tokenizer
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')



    # ****** Split the labeled and unlabeled training sets into clean sentences
    #
    sentences = []  # Initialize an empty list of sentences

    print ("Parsing sentences from training set")
    for review in train["review"]:
        sentences += Word2VecUtility.text_to_sentences(review, tokenizer)

    print ("Parsing sentences from test set")
    for review in test["review"]:
        sentences += Word2VecUtility.text_to_sentences(review, tokenizer)
    # ****** Set parameters and train the word2vec model
    #
    # Import the built-in logging module and configure it so that Word2Vec
    # creates nice output messages
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
        level=logging.INFO)

    # Set values for various parameters
    num_features = 300    # Word vector dimensionality
    min_word_count = 40   # Minimum word count
    num_workers = 4       # Number of threads to run in parallel
    context = 30          # Context window size
    downsampling = 1e-5   # Downsample setting for frequent words

    # Initialize and train the model (this will take some time)
    print ("Loading Word2Vec model...")
    model = Word2Vec(sentences,sg=0, workers=num_workers,
                size=num_features, min_count = min_word_count,
                window = context, sample = downsampling, seed=1)

    # If you don't plan to train the model any further, calling
    # init_sims will make the model much more memory-efficient.
    model.init_sims(replace=True)

    # It can be helpful to create a meaningful model name and
    # save the model for later use. You can load it later using Word2Vec.load()
    model_name = "300features_40minwords_10context_CBOW"
    # model_name = "300features_40minwords_10context_skipgram"
    model.save(model_name)
    #
    # model.doesnt_match("man woman child kitchen".split())
    # model.doesnt_match("france england germany berlin".split())
    # model.doesnt_match("paris berlin london austria".split())
    # model.most_similar("man")
    # model.most_similar("queen")
    # model.most_similar("awful")



    # ****** Create average vectors for the training and test sets
    #
    print ("Creating average feature vecs for training text")


    trainDataVecs = getAvgFeatureVecs(getCleanText(train), model, num_features)
    testDataVecs = getAvgFeatureVecs(getCleanText(test), model, num_features)
    # np.save('train/CBOW.npy', trainDataVecs)
    # np.save('test/CBOW.npy', testDataVecs)
    np.save('train/skip.npy', trainDataVecs)
    np.save('test/skip.npy', testDataVecs)

    # print ("Creating average feature vecs for test reviews")
    #
    # testDataVecs = getAvgFeatureVecs( getCleanReviews(test), model, num_features )


    # ****** Fit a random forest to the training set, then make predictions
    #
    # Fit a random forest to the training data, using 100 trees
    # forest = RandomForestClassifier( n_estimators = 300 )
    # # forest = svm.SVC(gamma=0.00001, C=150)
    # # forest = AdaBoostClassifier(n_estimators=300)
    #
    # print ("Fitting a random forest to labeled training data...")
    # #Evaluage via accuracy
    # # final=pd.DataFrame(trainDataVecs, train["LABEL"])
    # # final.to_csv('C:\waitingforprocess\output\modelaveragevectorskipgram.csv')
    # forest = forest.fit( trainDataVecs, train["sentiment"] )
    # # scores=cross_val_score(forest,trainDataVecs,train["LABEL"],cv=5,scoring='precision')
    # scores_accuracy=cross_val_score(forest,trainDataVecs,train["sentiment"],cv=5)
    # scores_precision=cross_val_score(forest,trainDataVecs,train["sentiment"],cv=5,scoring='precision')
    # scores_recall=cross_val_score(forest,trainDataVecs,train["sentiment"],cv=5,scoring='recall')
    # print('accuracy:', scores_accuracy.mean())
    # print('precison:', scores_precision.mean())
    # print('recall:', scores_recall.mean())




    # # Test & extract results
    # result = forest.predict( testDataVecs )
    #
    # # Write the test results
    # output = pd.DataFrame( data={"id":test["id"], "sentiment":result} )
    # output.to_csv( "Word2Vec_AverageVectors.csv", index=False, quoting=3 )
    # print ("Wrote Word2Vec_AverageVectors.csv")
