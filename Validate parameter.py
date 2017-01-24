from sklearn.ensemble import GradientBoostingClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
import numpy as np
from scipy.stats import randint as sp_randint
from sklearn.grid_search import GridSearchCV
from sklearn.grid_search import RandomizedSearchCV
import pandas as pd
from sklearn.linear_model import LogisticRegression
import os
from time import time
# Data = pd.read_csv('C:\waitingforprocess\InputDataclean.csv', names=['ID', 'TEXT', 'CUSTOMER', 'ORIGINAL', 'LABEL'],
#                    encoding='latin1')
train = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'labeledTrainData.tsv'), header=0,
                    delimiter="\t", quoting=3)
test = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'LSTMlabeledTestData.tsv'), header=0,
                   delimiter="\t", quoting=3)
# train=Data.copy()
#train models
# model_name = '300features_40minwords_10context_CBOW'
# trainmodel_name = 'train/CBOW.npy'
trainmodel_name = 'train/DBOWparagraphvectorfeature.npy'


#test models
# testmodel_name = 'test/CBOW.npy'
testmodel_name = 'test/DBOWparagraphvectorfeature.npy'
trainDataVecs=np.load(trainmodel_name)
testDataVecs=np.load(testmodel_name)
# msk=np.random.rand(len(trainDataVecs))<0.8
# msk=np.load('index.npy')
trainset=trainDataVecs
testset=testDataVecs
# print(len(trainDataVecs))
traintarget=train["sentiment"]
# traintarget=traintarget.to_arrary()
testtarget=test["sentiment"]
# testtarget=testtarget.to_arrary()
# print('trainbaseline:',traintarget.value_counts())
# print('testbaseline:',testtarget.value_counts())
# print(len(trainset))
# np.save('index.npy',msk)

# finalmodel=RandomForestClassifier(n_estimators=100,min_samples_leaf=5,criterion='gini',max_features=10,min_samples_split=10,max_depth=15,bootstrap=False)
# finalmodel=svm.SVC(C=100,gamma=0.0001)
# finalmodel=AdaBoostClassifier(n_estimators=100,learning_rate=1)
finalmodel=LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, penalty='l2', random_state=None, tol=0.0001)
finalmodel = finalmodel.fit(trainset, traintarget)
predict=finalmodel.predict(testset)
# print(predict)
# print(testtarget)
from sklearn.metrics import classification_report
# print(finalmodel.score(testset,testtarget))
from sklearn import metrics
print('accuracy:',metrics.accuracy_score(testtarget,predict))
print('precision:',metrics.precision_score(testtarget,predict))
print('recall:',metrics.recall_score(testtarget,predict))
print('AUC:',metrics.roc_auc_score(testtarget,predict))






# finalmodel=RandomForestClassifier(n_estimators=100,min_samples_leaf=2,criterion='gini',max_features=1,min_samples_split=10,max_depth=8,bootstrap=True)
# # finalmodel=svm.SVC(C=100,gamma=0.0001)
# # finalmodel=AdaBoostClassifier(n_estimators=100,learning_rate=1)
# finalmodel = finalmodel.fit(trainDataVecs, train["LABEL"])
# # scores=cross_val_score(forest,trainDataVecs,train["LABEL"],cv=5,scoring='precision')
# scores_accuracy = cross_val_score(finalmodel, trainDataVecs, train["LABEL"], cv=5)
# scores_precision = cross_val_score(finalmodel, trainDataVecs, train["LABEL"], cv=5, scoring='precision')
# scores_recall = cross_val_score(finalmodel, trainDataVecs, train["LABEL"], cv=5, scoring='recall')
# print('accuracy:', scores_accuracy.mean())
# print('precison:', scores_precision.mean())
# print('recall:', scores_recall.mean())