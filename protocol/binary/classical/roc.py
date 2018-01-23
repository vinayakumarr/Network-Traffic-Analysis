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
from sklearn.cross_validation import train_test_split
import pandas as pd
import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding
from keras.layers import LSTM, SimpleRNN, GRU
from keras.datasets import imdb
from keras.utils.np_utils import to_categorical
from sklearn.metrics import (precision_score, recall_score,
                             f1_score, accuracy_score,mean_squared_error,mean_absolute_error)
from sklearn import metrics
from sklearn.preprocessing import Normalizer
import h5py
from keras import callbacks
from keras import callbacks
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from sklearn.metrics import roc_curve, auc
import itertools
from scipy import interp
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
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import (precision_score, recall_score,f1_score, accuracy_score,mean_squared_error,mean_absolute_error, roc_curve, classification_report,auc)


def calc_macro_roc(fpr, tpr):
    """Calcs macro ROC on log scale"""
    # Create log scale domain
    all_fpr = sorted(itertools.chain(*fpr))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(len(tpr)):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    return all_fpr, mean_tpr / len(tpr), auc(all_fpr, mean_tpr) / len(tpr)

traindata = pd.read_csv('kdd/binary/train/train.csv', header=None)
testdata = pd.read_csv('kdd/binary/test/NIMS.csv', header=None)

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


model = LogisticRegression()
model.fit(traindata, trainlabel)


t_probs = model.predict(testdata)
print(t_probs)
t_fpr, t_tpr, _ = roc_curve(testlabel, t_probs)
fpr = []
tpr = []

fpr.append(t_fpr)
tpr.append(t_tpr)

lr_binary_fpr, lr_binary_tpr, lr_binary_auc = calc_macro_roc(fpr, tpr)


# fit a Naive Bayes model to the data
model = GaussianNB()
model.fit(traindata, trainlabel)
print(model)

t_probs = model.predict(testdata)
print(t_probs)
t_fpr, t_tpr, _ = roc_curve(testlabel, t_probs)
fpr1 = []
tpr1 = []

fpr1.append(t_fpr)
tpr1.append(t_tpr)

nb_binary_fpr, nb_binary_tpr, nb_binary_auc = calc_macro_roc(fpr1, tpr1)


# fit a k-nearest neighbor model to the data
model = KNeighborsClassifier()
model.fit(traindata, trainlabel)
print(model)

t_probs = model.predict(testdata)
print(t_probs)
t_fpr, t_tpr, _ = roc_curve(testlabel, t_probs)
fpr2 = []
tpr2 = []

fpr2.append(t_fpr)
tpr2.append(t_tpr)

knn_binary_fpr, knn_binary_tpr, knn_binary_auc = calc_macro_roc(fpr2, tpr2)

model = DecisionTreeClassifier()
model.fit(traindata, trainlabel)
print(model)

t_probs = model.predict(testdata)
print(t_probs)
t_fpr, t_tpr, _ = roc_curve(testlabel, t_probs)
fpr5 = []
tpr5 = []

fpr5.append(t_fpr)
tpr5.append(t_tpr)

dt_binary_fpr, dt_binary_tpr, dt_binary_auc = calc_macro_roc(fpr5, tpr5)

model = AdaBoostClassifier(n_estimators=100)
model.fit(traindata, trainlabel)

t_probs = model.predict(testdata)
print(t_probs)
t_fpr, t_tpr, _ = roc_curve(testlabel, t_probs)
fpr6 = []
tpr6 = []

fpr6.append(t_fpr)
tpr6.append(t_tpr)

ada_binary_fpr, ada_binary_tpr, ada_binary_auc = calc_macro_roc(fpr6, tpr6)




model = RandomForestClassifier(n_estimators=100)
model = model.fit(traindata, trainlabel)

t_probs = model.predict(testdata)
print(t_probs)
t_fpr, t_tpr, _ = roc_curve(testlabel, t_probs)
fpr7 = []
tpr7 = []

fpr7.append(t_fpr)
tpr7.append(t_tpr)

rf_binary_fpr, rf_binary_tpr, rf_binary_auc = calc_macro_roc(fpr7, tpr7)



AL = pd.read_csv('trl.csv', header=None)
AR = pd.read_csv('trr.csv', header=None)
ALR = pd.read_csv('te.csv', header=None)

l = AL.iloc[:,0]
r = AR.iloc[:,0]
t = ALR.iloc[:,0]

l = np.array(l)
r = np.array(r)
t = np.array(t)

t_fpr, t_tpr, _ = roc_curve(t, l)
fpr8 = []
tpr8 = []

fpr8.append(t_fpr)
tpr8.append(t_tpr)

lstm_binary_fpr, lstm_binary_tpr, lstm_binary_auc = calc_macro_roc(fpr8, tpr8)


t_fpr, t_tpr, _ = roc_curve(t, r)
fpr9 = []
tpr9 = []

fpr9.append(t_fpr)
tpr9.append(t_tpr)

rnn_binary_fpr, rnn_binary_tpr, rnn_binary_auc = calc_macro_roc(fpr9, tpr9)



import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
with plt.style.context('bmh'):
  plt.plot(lr_binary_fpr, lr_binary_tpr, label='LSTM (AUC = %.4f)' % (lr_binary_auc, ), rasterized=True,linewidth=0.3)
  plt.plot(nb_binary_fpr, nb_binary_tpr,label='RNN (AUC = %.4f)' % (nb_binary_auc, ), rasterized=True,linewidth=0.3)
  plt.plot(dt_binary_fpr, dt_binary_tpr,label='GRU (AUC = %.4f)' % (dt_binary_auc, ), rasterized=True,linewidth=0.3)
  plt.plot(knn_binary_fpr, knn_binary_tpr,label='KNN (AUC = %.4f)' % (knn_binary_auc, ), rasterized=True,linewidth=0.3)
  plt.plot(ada_binary_fpr, ada_binary_tpr,label='SVM (AUC = %.4f)' % (ada_binary_auc, ), rasterized=True,linewidth=0.3)
  plt.plot(rf_binary_fpr, rf_binary_tpr,label='RF (AUC = %.4f)' % (rf_binary_auc, ), rasterized=True,linewidth=0.3)
  plt.plot(lstm_binary_fpr, lstm_binary_tpr,label='SVM (AUC = %.4f)' % (lstm_binary_auc, ), rasterized=True,linewidth=1.0)
  plt.plot(rnn_binary_fpr, rnn_binary_tpr,label='RF (AUC = %.4f)' % (rnn_binary_auc, ), rasterized=True,linewidth=0.7)
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel('False Positive Rate', fontsize=10)
  plt.ylabel('True Positive Rate', fontsize=10)
  plt.title('ROC - Binary Classification', fontsize=10)
  plt.legend(loc="lower right", fontsize=10)
  plt.tick_params(axis='both', labelsize=10)
  plt.savefig('ALL.png')

