import numpy as np
import pandas as pd
from sklearn import decomposition
import matplotlib.pyplot as plt


Data = pd.read_csv('C:\waitingforprocess\InputDataclean.csv', names=['ID', 'TEXT', 'CUSTOMER', 'ORIGINAL', 'LABEL'],
                   encoding='latin1')
train = Data.copy()
# model_name = 'Model/tfidffeature.npy'
model_name = 'Model/DMPVparagraphvectorfeature.npy'
# model_name = 'Model/CBOWtfidf.npy'
# model_name = 'Model/DMPVparagraphvectorfeature.npy'
# model_name = 'Model/skip.npy'
# model_name = 'Model/skiptfidf.npy'
trainDataVecs=np.load(model_name)
# print(trainDataVecs)
# print(train['TEXT'])
#
pca=decomposition.PCA(n_components=2)
pca.fit(trainDataVecs)
reduced=pca.transform(trainDataVecs)
#
# for index,vec in enumerate(reduced):
#      x,y=vec[0],vec[1]
#      if train['ORIGINAL'][index]=='A':
#         plt.scatter(x,y,marker='o',c="r")
#         plt.annotate(train['ORIGINAL'][index],xy=(x,y))
#      if train['ORIGINAL'][index]=='B':
#          plt.scatter(x, y, marker='s', c="b")
#          plt.annotate(train['ORIGINAL'][index], xy=(x, y))

# Plot for reduced vectors
for index,vec in enumerate(reduced):
    x, y = vec[0], vec[1]
    if train['LABEL'][index]==1:
        plt.scatter(x,y,marker='o',c='r')
    else:
        plt.scatter(x,y, marker='s', c='b')
plt.show()

#Distribution analysis
# temp1=Data['ORIGINAL'].value_counts(ascending=True)
# print(temp1)
# fig=plt.figure(figsize=(8,4))
# ax1=fig.add_subplot(121)
# ax1.set_xlabel('Customer category')
# ax1.set_ylabel('count')
# ax1.set_title('Customer distribution analysis')
# temp1.plot(kind='bar')
# temp1.hist()
# plt.show()


