import numpy as np
import pandas as pd

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from keras import metrics

import sklearn.metrics as skmetrics
from sklearn.model_selection import train_test_split

labels_tests = ['LABEL_BaseExcess', 'LABEL_Fibrinogen', 'LABEL_AST',
                'LABEL_Alkalinephos', 'LABEL_Bilirubin_total', 'LABEL_Lactate',
                'LABEL_TroponinI', 'LABEL_SaO2', 'LABEL_Bilirubin_direct',
                'LABEL_EtCO2']
labels_sepsis = ['LABEL_Sepsis']
labels_VS_mean = ['LABEL_RRate', 'LABEL_ABPm', 'LABEL_SpO2', 'LABEL_Heartrate']

raw_train_path = "../data/train_features.csv"
raw_test_path = "../data/test_features.csv"
clean_columned_train_path = "../data/train_features_clean_columned.csv"
clean_columned_test_path = "../data/test_features_clean_columned.csv"

# Data import from folder
X = pd.read_csv(clean_columned_train_path, index_col=0)
Y_df = pd.read_csv("../data/train_labels.csv", index_col=0)

X_test = pd.read_csv(clean_columned_test_path, index_col=0)


# Definition of test and val data size:
train_size = 15000

X_train = X.iloc[0:train_size, :]
X_val = X.iloc[train_size:, :]
#X_train, X_val = train_test_split(X, test_size=0.2)

nanx = np.isnan(X_train)
nanxval = np.isnan(X_val)

#labels_target = [labels_tests]
labels_target = ['LABEL_BaseExcess']

# MODEL FOR SUBTASK 1
Y1_train = np.array(Y_df[labels_target])[0:train_size]
Y1_val = np.array(Y_df[labels_target])[train_size:]
#Y1_train_df, Y1_val_df = train_test_split(Y_df, test_size=0.2)

# MODEL TASK 1
print("######### TASK 1 #########")
epochs1 = 5

input_dim = X.shape[1]  # 35 uncolumned, #409 columned
output_dim = Y1_train.shape[1]  # 10

weights = (Y1_train.shape[0] + Y1_val.shape[0])/(sum(Y1_train) + sum(Y1_val))
weights = dict(zip([i+2 for i in range(Y1_train.shape[1])], weights))

model1 = Sequential()
model1.add(Dense(units=input_dim, input_dim=input_dim))
model1.add(Dense(units=100, activation = "relu"))
model1.add(Dense(units=200, activation = "relu"))
model1.add(Dense(units=200, activation = "relu"))
model1.add(Dense(units=100, activation = "relu"))
model1.add(Dense(units=output_dim, activation='sigmoid'))
sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model1.compile(loss=keras.losses.binary_crossentropy,
               optimizer=optimizers.Adadelta(),
               #optimizer=sgd,
               metrics= [metrics.categorical_accuracy])

model1.fit(X_train, Y1_train, batch_size=32, epochs=epochs1, shuffle=True)
test_loss, test_acc = model1.evaluate(X_val, Y1_val)
print("TASK 1 --> test_acc: ", test_acc)
Y1_pred = model1.predict(X_val)
task1 = 0
for i in range(len(labels_target)):
    task1 = task1 + 1 / len(labels_target) * (skmetrics.roc_auc_score(Y1_val[:, i], Y1_pred[:, i]))
print("AOC score -- task 1: ", task1, "\n\n")
