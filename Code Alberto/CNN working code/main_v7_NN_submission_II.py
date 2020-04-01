import pandas as pd
import numpy as np
import keras
import sys
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Conv1D, MaxPooling1D, Activation, Dropout, Flatten
from keras import backend as K
import sklearn.metrics as skmetrics
np.random.seed(seed=123)
#from sklearn.metrics import classification_report, confusion_matrix

# DATA CLEANING
#import cleaning_script

#cleaned data import:
train_features = pd.read_csv("../../data/train_features_clean_all.csv")
test_features_pre = pd.read_csv("../../data/test_features_clean_all.csv")
train_labels = pd.read_csv("../../data/train_labels.csv")

# Informatons on the headers -- Extracting information:
patient_characteristics = ["Age"] # TIME VARIABLE IS EXCLUDED
vital_signs = ["Heartrate", "SpO2", "ABPs", "ABPm", "ABPd", "RRate"]
tests = ['EtCO2', 'PTT', 'BUN', 'Lactate', 'Temp', 'Hgb', 'HCO3', 'BaseExcess',
       'Fibrinogen', 'Phosphate', 'WBC', 'Creatinine', 'PaCO2', 'AST', 'FiO2',
       'Platelets', 'SaO2', 'Glucose', 'Magnesium', 'Potassium', 'Calcium',
       'Alkalinephos', 'Bilirubin_direct', 'Chloride', 'Hct',
       'Bilirubin_total', 'TroponinI', 'pH']
labels_tests = ['LABEL_BaseExcess', 'LABEL_Fibrinogen', 'LABEL_AST',
       'LABEL_Alkalinephos', 'LABEL_Bilirubin_total', 'LABEL_Lactate',
       'LABEL_TroponinI', 'LABEL_SaO2', 'LABEL_Bilirubin_direct',
       'LABEL_EtCO2' ]
labels_sepsis = ['LABEL_Sepsis']
labels_VS_mean = ['LABEL_RRate', 'LABEL_ABPm', 'LABEL_SpO2', 'LABEL_Heartrate']
headers_train = train_features.columns
headers_test = test_features_pre.columns
N_patients_train = np.array(train_features.shape[0]/12).astype(int)
N_patients_test = np.array(test_features_pre.shape[0]/12).astype(int)

# Drop pid feature:
train_features = train_features.drop(labels = "pid", axis = 1)
test_features = test_features_pre.drop(labels = "pid", axis = 1)

# Definition of test and val data size:
train_size = 15000
X_val = train_features.iloc[train_size*12:,:]
X_old = train_features.iloc[0:train_size*12,:]

#####################################################################################################
# MODEL FOR SUBTASK 1
print("train set labels: ", X_old.shape)
print("train set features: ", train_features.shape)
Y_t1 = train_labels['LABEL_BaseExcess'].iloc[0:train_size]
Y_val_t1 = train_labels['LABEL_BaseExcess'].iloc[train_size:]
print(Y_t1.shape)
print("######### TASK 1 #########")
epochs_t1 = 5
# NEED TO FEED THE NETWORK WITH SIZE (N_PATIENTS, N_HOURS, N_TESTS, 1) IMPUT (N_HOURS, N_TESTS, 1)
N_patients_train = np.array(X_old.shape[0]/12).astype(int) ## ONLY IN VALIDATION PHASE
X = np.array(X_old).reshape((N_patients_train, 12, 35))
print(X.shape)
input_shape =(12, 35)
num_classes = 1

model_t1 =Sequential()

model_t1 = Sequential()
model_t1.add(Conv1D(51, kernel_size=5,
                 activation='relu',
                 input_shape=input_shape))
model_t1.add(Conv1D(420, 3, activation='relu'))
model_t1.add(MaxPooling1D(pool_size=2))
model_t1.add(Dropout(0.25))
model_t1.add(Flatten())
model_t1.add(Dense(500, activation='relu'))
model_t1.add(Dropout(0.5))
model_t1.add(Dense(2, activation='softmax'))

model_t1.compile(optimizer=keras.optimizers.Adadelta(),
                loss=keras.losses.sparse_categorical_crossentropy,
                metrics=[keras.metrics.categorical_accuracy])

Y = keras.utils.to_categorical(Y_t1, 2)

model_t1.fit(X, Y_t1, epochs=epochs_t1)

N_patients_val = np.array(X_val.shape[0]/12).astype(int) ## ONLY IN VALIDATION PHASE
X_val = np.array(X_val).reshape((N_patients_val, 12, 35))
Y_val_pred = model_t1.predict(X_val) 
print(Y_val_pred)

task1 = np.mean([skmetrics.roc_auc_score(Y_val_t1, Y_val_pred[:,1])])
print("ROC AUC -- score: ", task1)

print()
