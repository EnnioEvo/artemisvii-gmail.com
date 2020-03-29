import pandas as pd
import scipy as sp
from sklearn import svm
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# Data import from folder
X_train_raw = sp.array(pd.read_csv("../data/train_features.csv"), dtype=sp.float64)
Y_train = sp.array(pd.read_csv("../data/train_labels.csv"), dtype=sp.float64)
test_features = sp.array(pd.read_csv("../data/test_features.csv"), dtype=sp.float64)

clf = svm.SVC(kernel='linear', verbose=True, tol=1e-1)
X_train = X_train_raw
X_train[np.isnan(X_train_raw)]= 0
X_train = X_train[0:1000,:]
Y_train = X_train
clf.fit(X_train, Y_train)
