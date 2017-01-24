import pandas as pd
import gensim
from gensim.models import doc2vec
from collections import namedtuple
from Word2VecUtility import Word2VecUtility
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from gensim.models.doc2vec import TaggedDocument
import pickle
from sklearn import linear_model
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
import nltk
from sklearn import svm
import os

if __name__ == '__main__':

    # Data = pd.read_csv('C:\waitingforprocess\InputDataclean.csv', names=['ID', 'TEXT', 'CUSTOMER', 'ORIGINAL','LABEL'],
    #                    encoding='latin1')
    # train=Data.copy()
    train = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'labeledTrainData.tsv'), header=0,
                        delimiter="\t", quoting=3)
    test = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'LSTMlabeledTestData.tsv'), header=0,
                       delimiter="\t", quoting=3)
# Initialize an empty list to hold the clean reviews
    clean_train_text = []

    # Loop over each review; create an index i that goes from 0 to the length
    # of the movie review list

    print ("Cleaning and parsing the training set raw text...\n")
    for i in range( 0, len(train["review"])):
        clean_train_text.append(" ".join(Word2VecUtility.text_to_wordlist(train["review"][i], True)))

    clean_test_text = []

    # Loop over each review; create an index i that goes from 0 to the length
    # of the movie review list

    print("Cleaning and parsing the training set raw text...\n")
    for i in range(0, len(test["review"])):
        clean_test_text.append(" ".join(Word2VecUtility.text_to_wordlist(test["review"][i], True)))

#Transform data into doc format
traindocuments=[]
trainindex=[]
testdocuments=[]
testindex=[]
# Load the punkt tokenizer
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
for i,text in enumerate(clean_train_text):
    tags='train_'+str(i)
    # doc=tokenizer.tokenize(text)
    trainindex.append(tags)
    traindocuments.append(TaggedDocument(text, [tags]))

for i,text in enumerate(clean_test_text):
    tags='test_'+str(i)
    # doc=tokenizer.tokenize(text)
    testindex.append(tags)
    testdocuments.append(TaggedDocument(text, [tags]))
documents=traindocuments+testdocuments
# print(documents[0])
#Train models
# model=doc2vec.Doc2Vec(documents,window=30,min_count=1,pretrained_emb='300features_40minwords_10context_CBOW_vector')
model=doc2vec.Doc2Vec(min_count=1, window=10, size=100, sample=1e-4, negative=5, workers=7,dm=0)
model.build_vocab(documents)
model.train(documents)

train_data_features=[]
test_data_features=[]
print(model.docvecs)
for item in trainindex:
    train_data_features.append(model.docvecs[item])

# testmodel=doc2vec.Doc2Vec(window=30,min_count=2,dm=1,size=300)
# testmodel.build_vocab(testdocuments)
# testmodel.train(testdocuments)
for item in testindex:
    test_data_features.append(model.docvecs[item])
# print(train_data_features[0])
# print(len(train_data_features))
# np.save('train/DMPVparagraphvectorfeature',train_data_features)
# np.save('test/DMPVparagraphvectorfeature',train_data_features)
np.save('train/DBOWparagraphvectorfeature',train_data_features)
np.save('test/DBOWparagraphvectorfeature',test_data_features)

# Initialize a Random Forest classifier with 100 trees
# forest = RandomForestClassifier(n_estimators=300)
# forest = AdaBoostClassifier(n_estimators=300)
# forest = GradientBoostingRegressor(n_estimators=100)

# Fit the forest to the training set, using the bag of words as
# features and the sentiment labels as the response variable
#
# This may take a few minutes to run
# Evaluate via accuracy
# final = pd.DataFrame(train_data_features, train["LABEL"])
# final.to_csv('C:\waitingforprocess\output\modelbagofwords.csv')
# print(len(train_data_features))
# print(len(train["LABEL"]))
# forest= linear_model.LinearRegression()
# classmodel = forest.fit(train_data_features, train["LABEL"])
# forest = svm.SVC(gamma=0.00001,C=150)
# print(len(train_data_features))
# print(train_data_features[0])
# print(len(train["LABEL"]))
# print(train["LABEL"][0])
# scores = cross_val_score(classmodel,train_data_features, train["LABEL"], cv=5,scoring='precision')
# print(classmodel.score(train_data_features, train["LABEL"]))
# forest = forest.fit( train_data_features, train["sentiment"] )
# scores_accuracy = cross_val_score(forest, train_data_features, train["sentiment"], cv=5)
# scores_precision = cross_val_score(forest, train_data_features, train["sentiment"], cv=5, scoring='precision')
# scores_recall = cross_val_score(forest, train_data_features, train["sentiment"], cv=5, scoring='recall')
# print('accuracy:',scores_accuracy.mean())
# print('precison:' ,scores_precision.mean())
# print('recall:' ,scores_recall.mean())