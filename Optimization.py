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
from time import time
import os

# Data = pd.read_csv('C:\waitingforprocess\InputDataclean.csv', names=['ID', 'TEXT', 'CUSTOMER', 'ORIGINAL', 'LABEL'],
#                    encoding='latin1')
train = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'labeledTrainData.tsv'), header=0,
                    delimiter="\t", quoting=3)
# # Verify the number of reviews that were read (100,000 in total)
# print ("Read %d labeled train reviews, %d labeled test reviews, " \
#  "and %d unlabeled reviews\n" % (train["review"].size,
#  test["review"].size, unlabeled_train["review"].size ))
def report(results,n_top=3):
    for i in range(1,n_top+1):
        candidates=np.flatnonzero(results['rank_test_score']==i)
        for candidate in candidates:
            print("Model with Rank:{0}".format(i))
            print("Mean validation score:{0:.3f}(std:{1:.3f}".format(results['mean_test_score'][candidate],results['std_test_score'][candidate]))
            print("Parameter:{0}".format(results['params'][candidate]))
            print("")
msk=np.load('index.npy')
# train = Data[msk]
# model_name = 'train/bagofwords.npy'
model_name = 'train/CBOW.npy'
# model_name = 'train/CBOWtfidf.npy'
# model_name = 'train/DMPVparagraphvectorfeature.npy'
# model_name = 'train/paragraphvectorfeature.npy'
# model_name = 'train/skip.npy'
# model_name = 'train/skiptfidf.npy'
trainDataVecs=np.load(model_name)
# print(len(trainDataVecs))

# Fit a random forest to the training data, using 100 trees
forest = RandomForestClassifier(n_estimators=100)

#Fit a SVM
# forest=svm.SVC()
#Fit a adaboost
# forest=AdaBoostClassifier(n_estimators=100)
#Specify parameters:
# param_dist={"max_depth":sp_randint(3,11),
#             "max_features":sp_randint(2,11),
#             "min_samples_split":sp_randint(2,11),
#             "min_samples_leaf":sp_randint(2,11),
#             "bootstrap":[True,False],
#             "criterion":["gini","entropy"]
#             }
#
# #run randomized search
#
# n_iter_search=50
# random_search=RandomizedSearchCV(forest,param_distributions=param_dist,n_iter=n_iter_search)
#
# start=time()
# random_search.fit(trainDataVecs,train["LABEL"])
#
# print("RandomizedSearchCV took %.2f seconds for %d candidates""parameter settings."%((time()-start),n_iter_search))
# report(random_search.cv_results_)




#Specify parameters for Randomforest:
param_grid={"max_depth":[3,8,15],
            "max_features":[1,3,10],
            "min_samples_split":[2,5,10],
            "min_samples_leaf":[2,5,10],
            "bootstrap":[True,False],
            "criterion":["gini","entropy"]
            }

#parameters for SVM
# param_grid={"C":[1,10,100,1000],
#             "gamma":[0.0001,0.001,0.01]
#             }

#parameters for Adaboost
# param_grid={"learning_rate":[0.1,0.5,1]
#             }
#run Grid search

n_iter_search=20
grid_search=GridSearchCV(forest,param_grid=param_grid)

start=time()
grid_search.fit(trainDataVecs,train["sentiment"])

print("GridSearchCV took %.2f seconds for %d candidates""parameter settings."%((time()-start),len(grid_search.cv_results_['params'])))
report(grid_search.cv_results_)



# # forest = svm.SVC(gamma=0.00001, C=150)
# # forest = AdaBoostClassifier(n_estimators=300)
#
# print("Fitting a random forest to labeled training data...")
# # Evaluage via accuracy
# # final=pd.DataFrame(trainDataVecs, train["LABEL"])
# # final.to_csv('C:\waitingforprocess\output\modelaveragevectorskipgram.csv')
# finalmodel=RandomForestClassifier(n_estimators=100)
# finalmodel = finalmodel.fit(trainDataVecs, train["LABEL"])
# # scores=cross_val_score(forest,trainDataVecs,train["LABEL"],cv=5,scoring='precision')
# scores_accuracy = cross_val_score(finalmodel, trainDataVecs, train["LABEL"], cv=5)
# scores_precision = cross_val_score(finalmodel, trainDataVecs, train["LABEL"], cv=5, scoring='precision')
# scores_recall = cross_val_score(finalmodel, trainDataVecs, train["LABEL"], cv=5, scoring='recall')
# print('accuracy:', scores_accuracy.mean())
# print('precison:', scores_precision.mean())
# print('recall:', scores_recall.mean())