import pandas as pd
import numpy as np
import keras
import sys
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Conv1D, MaxPooling1D, Activation, Dropout, Flatten
from keras import backend as K
import sklearn.metrics as skmetrics
from math import sqrt
from math import floor

np.random.seed(seed=123)
# from sklearn.metrics import classification_report, confusion_matrix

# DATA CLEANING
# import cleaning_script

# cleaned data import:
train_features = pd.read_csv("../data/train_features_clean_all.csv")
test_features_pre = pd.read_csv("../data/test_features_clean_all.csv")
train_labels = pd.read_csv("../data/train_labels.csv")
sample = pd.read_csv("../sample.csv")

# Informatons on the headers -- Extracting information:
patient_characteristics = ["Age"]  # TIME VARIABLE IS EXCLUDED
vital_signs = ["Heartrate", "SpO2", "ABPs", "ABPm", "ABPd", "RRate"]
tests = ['EtCO2', 'PTT', 'BUN', 'Lactate', 'Temp', 'Hgb', 'HCO3', 'BaseExcess',
         'Fibrinogen', 'Phosphate', 'WBC', 'Creatinine', 'PaCO2', 'AST', 'FiO2',
         'Platelets', 'SaO2', 'Glucose', 'Magnesium', 'Potassium', 'Calcium',
         'Alkalinephos', 'Bilirubin_direct', 'Chloride', 'Hct',
         'Bilirubin_total', 'TroponinI', 'pH']
labels_tests = ['LABEL_BaseExcess', 'LABEL_Fibrinogen', 'LABEL_AST',
                'LABEL_Alkalinephos', 'LABEL_Bilirubin_total', 'LABEL_Lactate',
                'LABEL_TroponinI', 'LABEL_SaO2', 'LABEL_Bilirubin_direct',
                'LABEL_EtCO2']
labels_sepsis = ['LABEL_Sepsis']
labels_VS_mean = ['LABEL_RRate', 'LABEL_ABPm', 'LABEL_SpO2', 'LABEL_Heartrate']
all_labels = labels_tests + labels_sepsis + labels_VS_mean

headers_train = train_features.columns
index_train = train_features.index
headers_test = test_features_pre.columns
index_test = test_features_pre.index

N_patients_train = np.array(train_features.shape[0] / 12).astype(int)
N_patients_test = np.array(test_features_pre.shape[0] / 12).astype(int)

# Drop pid feature:
train_features = train_features.drop(labels="pid", axis=1)
test_features = test_features_pre.drop(labels="pid", axis=1)

# Definition of test and val data size:
train_size = 15000
X_old = np.array(train_features)[0:train_size * 12, :]
X_val = np.array(train_features)[train_size * 12:, :]
X_test = np.array(test_features)
# Standardize the data
X_old = (X_old - np.mean(X_old,0))/np.std(X_old,0)
X_val = (X_val - np.mean(X_val,0))/np.std(X_val,0)
X_test = (X_test - np.mean(X_test,0))/np.std(X_test,0)

#####################################################################################################
# MODEL FOR SUBTASK 1
print("train set labels: ", X_old.shape)
print("train set features: ", train_features.shape)

print("######### TASK 1 #########")

# NEED TO FEED THE NETWORK WITH SIZE (N_PATIENTS, N_HOURS, N_TESTS, 1) IMPUT (N_HOURS, N_TESTS, 1)
N_patients_train = np.array(X_old.shape[0]/12).astype(int) ## ONLY IN VALIDATION PHASE
X = X_old.reshape((N_patients_train, 12, 35))


print("X shape: ", X.shape)
input_shape = (12, 35)
# num_classes = len(labels_target)

# -----------  MODEL FOR TASK 1 ---------------
model_t1 = Sequential()
model_t1.add(Conv1D(100, kernel_size=5,
                    activation='relu',
                    input_shape=input_shape))
model_t1.add(Conv1D(100, 3, activation='relu'))
model_t1.add(MaxPooling1D(pool_size=2))
model_t1.add(Dropout(0.25))
model_t1.add(Flatten())
model_t1.add(Dense(100, activation='relu'))
model_t1.add(Dropout(0.5))
model_t1.add(Dense(2, activation='softmax'))

model_t1.compile(optimizer=keras.optimizers.Adam(),
                 loss=keras.losses.sparse_categorical_crossentropy,
                 metrics=[keras.metrics.categorical_accuracy])

# -----------  MODEL FOR TASK 2 ---------------
model_t2 = Sequential()
model_t2.add(Conv1D(50, kernel_size=5,
                    activation='relu',
                    input_shape=input_shape))
model_t2.add(Conv1D(50, 3, activation='relu'))
model_t2.add(MaxPooling1D(pool_size=2))
model_t2.add(Dropout(0.25))
model_t2.add(Flatten())
model_t2.add(Dense(50, activation='relu'))
model_t2.add(Dropout(0.25))
model_t2.add(Dense(50, activation='relu'))
model_t2.add(Dropout(0.25))
model_t2.add(Dense(1, activation='relu'))

model_t2.compile(optimizer=keras.optimizers.Adam(),
                 loss=keras.losses.mean_squared_error,
                 metrics=[keras.metrics.categorical_accuracy])

#---------------------------------------------------------
# ------------------- TRAINING TASK 2 --------------------
#---------------------------------------------------------

N_patients_val = np.array(X_val.shape[0] / 12).astype(int)  ## ONLY IN VALIDATION PHASE
X_val = X_val.reshape((N_patients_val, 12, 35))
X_test = X_test.reshape((N_patients_test, 12, 35))

print(model_t2.summary())
epochs_t2 = 1
#labels_target = ['LABEL_Sepsis']
labels_target = labels_VS_mean

Y_test_tot = pd.DataFrame(np.zeros([N_patients_test, len(all_labels)]), columns=all_labels)
for i in range(0, len(labels_target)):

    label_target = labels_target[i]
    Y_t2 = train_labels[label_target].iloc[0:train_size]
    Y_val_t2 = train_labels[label_target].iloc[train_size:]


    model_t2.fit(X, Y_t2, epochs=epochs_t2, verbose=0)

    Y_val_pred = model_t2.predict(X_val)
    Y_val_pred = Y_val_pred.flatten()
    task2 = skmetrics.r2_score(Y_val_t2, Y_val_pred, sample_weight=None, multioutput='uniform_average')
    print("R2 score ", i, " ", label_target, " :", task2)

    Y_test_pred = model_t2.predict(X_test)
    Y_test_tot.loc[:,label_target] = Y_test_pred

# ------------------- TRAINING TASK 1 --------------------
#   from array to list
sample_weights = np.array(pd.read_csv("../data/sample_weights.csv"))[0:train_size].flatten()
sample_weights = sample_weights/1000
sample_weights = sample_weights/sqrt(np.dot(sample_weights, sample_weights))

print(model_t1.summary())
epochs_t1 = 1
labels_target = labels_tests + ['LABEL_Sepsis']
#labels_target = ['LABEL_BaseExcess','LABEL_EtCO2','LABEL_SaO2']
#labels_target = ['LABEL_BaseExcess']
#labels_target = ['LABEL_SaO2']
#labels_target = ['LABEL_Sepsis']
#labels_target = labels_VS_mean

Y_val_tot = np.zeros([X_val.shape[0], len(labels_target)])
for i in range(0, len(labels_target)):

    label_target = labels_target[i]
    Y_t1 = train_labels[label_target].iloc[0:train_size]
    Y_val_t1 = train_labels[label_target].iloc[train_size:]
    # Y = keras.utils.to_categorical(Y_t1, 2)

    weight0 = (Y_t1.shape[0] + Y_val_t1.shape[0]) / (sum(Y_t1 != 0) + sum(Y_val_t1 != 0) + 1)
    weight1 = (Y_t1.shape[0] + Y_val_t1.shape[0]) / (sum(Y_t1 == 0) + sum(Y_val_t1 == 0) + 1)

    class_weights = {0: weight0, 1: weight1}

    model_t1.fit(X, Y_t1, epochs=epochs_t1, verbose=0, class_weight=class_weights)

    Y_val_pred = model_t1.predict(X_val)
    Y_val_tot[:,i] = Y_val_pred[:, -1]
    task1 = np.mean([skmetrics.roc_auc_score(Y_val_t1, Y_val_pred[:, -1])])
    print("ROC AUC -- score ", i, " ", label_target, " :", task1)

    Y_test_pred = model_t1.predict(X_test)
    Y_test_tot.loc[:,label_target] = Y_test_pred[:, -1]

submSet = pd.DataFrame(Y_val_tot, index=None, columns=labels_target)
submSet.to_csv('../data/predictions.csv', header=labels_target, index=False)
print()

Y_test_tot.insert(0, 'pid', sample['pid'])
Y_test_tot.to_csv('../data/submission.csv', header=True, index=False, float_format='%.7f')