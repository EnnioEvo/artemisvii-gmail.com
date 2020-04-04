import pandas as pd
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras import backend as K
import sklearn.metrics as skmetrics
#from sklearn.metrics import classification_report, confusion_matrix

# DATA CLEANING
#import cleaning_script

#cleaned data import:
train_features = pd.read_csv("data/train_features_clean.csv")
test_features = pd.read_csv("data/test_features_clean.csv")
train_labels = pd.read_csv("data/train_labels.csv")

# Informatons on the headers -- Extracting information:
#patient_characteristics = ["Age"] # TIME VARIABLE IS EXCLUDED
#vital_signs = ["Heartrate", "SpO2", "ABPs", "ABPm", "ABPd", "RRate"]
# tests = ['EtCO2', 'PTT', 'BUN', 'Lactate', 'Temp', 'Hgb', 'HCO3', 'BaseExcess',
#        'Fibrinogen', 'Phosphate', 'WBC', 'Creatinine', 'PaCO2', 'AST', 'FiO2',
#        'Platelets', 'SaO2', 'Glucose', 'Magnesium', 'Potassium', 'Calcium',
#        'Alkalinephos', 'Bilirubin_direct', 'Chloride', 'Hct',
#        'Bilirubin_total', 'TroponinI', 'pH']
labels_tests = ['LABEL_BaseExcess', 'LABEL_Fibrinogen', 'LABEL_AST',
       'LABEL_Alkalinephos', 'LABEL_Bilirubin_total', 'LABEL_Lactate',
       'LABEL_TroponinI', 'LABEL_SaO2', 'LABEL_Bilirubin_direct',
       'LABEL_EtCO2' ]
labels_sepsis = ['LABEL_Sepsis']
labels_VS_mean = ['LABEL_RRate', 'LABEL_ABPm', 'LABEL_SpO2', 'LABEL_Heartrate']
#headers_train = test_features.columns
#headers_test = test_features.columns
#N_patients_train = np.array(train_features.shape[0]/12).astype(int)
#N_patients_test = np.array(test_features.shape[0]/12).astype(int)

# Drop pid feature:
train_features = train_features.drop(labels = "pid", axis = 1)
test_features = test_features.drop(labels = "pid", axis = 1)

# Definition of test and val data size:
train_size = 15000
X_val = train_features.iloc[train_size:,:]
X = train_features.iloc[0:train_size,:]

# MODEL FOR SUBTASK 1
Y_t1 = train_labels[labels_tests].iloc[0:train_size]
Y_val_t1 = train_labels[labels_tests].iloc[train_size:]


# MODEL TASK 1
print("######### TASK 1 #########")
epochs_t1 = 10

model_t1 =Sequential()
model_t1.add(Dense(units = 35, input_dim= 35))
model_t1.add(Dense(1500, activation = "relu"))
model_t1.add(Dense(740, activation = "relu"))
model_t1.add(Dense(200, activation = "relu"))
model_t1.add(Dense(Y_t1.shape[1], activation="sigmoid"))

model_t1.compile(optimizer=keras.optimizers.Adadelta(),
 loss=keras.losses.binary_crossentropy,
  metrics=[keras.metrics.categorical_accuracy])
model_t1.fit(X, Y_t1, epochs=epochs_t1)
test_loss, test_acc = model_t1.evaluate(X_val, Y_val_t1)
print("TASK 1 --> test_acc: ", test_acc)
Y_pred_task1 = model_t1.predict(X_val)

Y_val_t1 = np.array(Y_val_t1)
task1 = 0
for ii in range(len(labels_tests)):
       task1 = task1 + 1/len(labels_tests)*(skmetrics.roc_auc_score(Y_val_t1[:,ii], Y_pred_task1[:,ii]))
print("AOC score -- task 1: ", task1)

print()
print()

# MODEL FOR SUBTASK 2
print("######### TASK 2 #########")
Y_t2 = train_labels[labels_sepsis].iloc[0:15000]
Y_val_t2 = train_labels[labels_sepsis].iloc[15000:]
# MODEL
epochs_t2 = 10

model_t2 =Sequential()
model_t2.add(Dense(units = 35, input_dim= 35))
model_t2.add(Dense(1500, activation = "relu"))
model_t2.add(Dense(400, activation = "relu"))
model_t2.add(Dense(500, activation = "relu"))
model_t2.add(Dense(Y_t2.shape[1], activation="softmax"))

model_t2.compile(optimizer=keras.optimizers.Adadelta(),
 loss=keras.losses.binary_crossentropy,
  metrics=[keras.metrics.categorical_accuracy])

model_t2.fit(X, Y_t2, epochs=epochs_t2)

test_loss, test_acc = model_t2.evaluate(X_val, Y_val_t2)
print("TASK 2 --> test_acc: ", test_acc)
Y_pred_task2 = model_t2.predict(X_val)
Y_val_t2 = np.array(Y_val_t2)

task2 = skmetrics.roc_auc_score(Y_val_t2[:], Y_pred_task2[:])

print("AOC score -- task 2: ", task2)

print()
print()

# MODEL FOR SUBTASK 3
print("######### TASK 3 #########")
Y_t3 = train_labels[labels_VS_mean].iloc[0:15000]
Y_val_t3 = train_labels[labels_VS_mean].iloc[15000:]
# MODEL
epochs_t2 = 1

model_t3 =Sequential()
model_t3.add(Dense(units = 35, input_dim= 35))
model_t3.add(Dense(1500, activation = "relu"))
model_t3.add(Dense(1200, activation = "relu"))
model_t3.add(Dense(500, activation = "relu"))
model_t3.add(Dense(Y_t3.shape[1], activation="relu"))

def r2_keras(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return 0.5 + 0.5*( 1 - SS_res/(SS_tot + K.epsilon()) )

model_t3.compile(optimizer=keras.optimizers.Adadelta(),
 loss=keras.losses.mean_squared_error,
  metrics=[r2_keras])
model_t3.fit(X, Y_t3, epochs=epochs_t2)
test_loss, test_r2 = model_t3.evaluate(X_val, Y_val_t3)
print("TASK 3 --> R2SCORE: ", test_r2)

Y_pred_task3 = model_t3.predict(X_val)
Y_val_t3 = np.array(Y_val_t3)

task3 = np.zeros(len(labels_VS_mean))
for ii in range(len(labels_VS_mean)):
       task3[ii] = 0.5 + 0.5 * np.maximum(0, skmetrics.r2_score(Y_val_t3[:,ii], Y_pred_task3[:,ii]))
task3_mean = np.mean(task3)
print("R2 score -- task 3: ", task3)

print()
print()

print("-------------------------------------")
score = np.mean([task1, task2, task3_mean])
print("TOTAL SCORE: ", score)