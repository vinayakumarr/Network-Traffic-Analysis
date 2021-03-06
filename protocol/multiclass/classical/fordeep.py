import numpy as np
import pandas as pd
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier
from sklearn.cross_validation import train_test_split
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (precision_score, recall_score,f1_score, accuracy_score,mean_squared_error,mean_absolute_error)
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Normalizer


traindata = pd.read_csv('kdd/multiclass/test.csv', header=None)
testdata = pd.read_csv('kdd/multiclass/test.csv', header=None)

X = traindata.iloc[:,0:22]
Y = traindata.iloc[:,22]
C = testdata.iloc[:,22]
T = testdata.iloc[:,0:22]

scaler = Normalizer().fit(X)
trainX = scaler.transform(X)

scaler = Normalizer().fit(T)
testT = scaler.transform(T)


traindata = np.array(trainX)
trainlabel = np.array(Y)

testdata = np.array(testT)
testlabel = np.array(C)


model = LogisticRegression()
model.fit(traindata, trainlabel)


# make predictions
expected = testlabel
predicted = model.predict(testdata)


print("***************************************************************")

model = LogisticRegression()
model.fit(traindata, trainlabel)


# make predictions
expected = testlabel
predicted = model.predict(testdata)

np.savetxt('deep/LRpredicted.txt',predicted , fmt='%01d')

# fit a Naive Bayes model to the data
model = GaussianNB()
model.fit(traindata, trainlabel)
print(model)
# make predictions
expected = testlabel
predicted = model.predict(testdata)

np.savetxt('deep/expected.txt',expected, fmt='%01d')
np.savetxt('deep/NBpredicted.txt',predicted , fmt='%01d')


# fit a k-nearest neighbor model to the data
model = KNeighborsClassifier()
model.fit(traindata, trainlabel)
print(model)
# make predictions
expected = testlabel
predicted = model.predict(testdata)
# summarize the fit of the model

np.savetxt('deep/KNNpredicted.txt',predicted , fmt='%01d')


model = DecisionTreeClassifier()
model.fit(traindata, trainlabel)
print(model)
# make predictions
expected = testlabel
predicted = model.predict(testdata)
# summarize the fit of the model
np.savetxt('deep/DTpredicted.txt',predicted , fmt='%01d')

model = AdaBoostClassifier(n_estimators=100)
model.fit(traindata, trainlabel)

# make predictions
expected = testlabel
predicted = model.predict(testdata)
# summarize the fit of the model
np.savetxt('deep/ABpredicted.txt',predicted , fmt='%01d')

model = RandomForestClassifier(n_estimators=100)
model = model.fit(traindata, trainlabel)

# make predictions
expected = testlabel
predicted = model.predict(testdata)
# summarize the fit of the model
np.savetxt('deep/RFpredicted.txt',predicted , fmt='%01d')



