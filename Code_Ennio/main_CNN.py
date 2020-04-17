import pandas as pd
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Activation, Dropout, Flatten, LSTM, TimeDistributed
import sklearn.metrics as skmetrics
from sklearn.linear_model import LinearRegression
from sklearn import svm

np.random.seed(seed=123)
# from sklearn.metrics import classification_report, confusion_matrix

# DATA CLEANING
# import cleaning_script

# cleaned data import:
train_features = pd.read_csv("../data/train_features_clean_all.csv")
test_features = pd.read_csv("../data/test_features_clean_all.csv")
train_labels = pd.read_csv("../data/train_labels.csv")
sample = pd.read_csv("../sample.csv")

# Informatons on the headers -- Extracting information:

#features
patient_characteristics = ["Age"]  # TIME VARIABLE IS EXCLUDED
vital_signs = ["Heartrate", "Temp", "SpO2", "ABPs", "ABPm", "ABPd", "RRate"]
tests = ['EtCO2', 'PTT', 'BUN', 'Lactate',  'Hgb', 'HCO3', 'BaseExcess',
         'Fibrinogen', 'Phosphate', 'WBC', 'Creatinine', 'PaCO2', 'AST', 'FiO2',
         'Platelets', 'SaO2', 'Glucose', 'Magnesium', 'Potassium', 'Calcium',
         'Alkalinephos', 'Bilirubin_direct', 'Chloride', 'Hct',
         'Bilirubin_total', 'TroponinI', 'pH']
all_features = patient_characteristics + vital_signs + tests
#labels
labels_tests = ['LABEL_BaseExcess', 'LABEL_Fibrinogen', 'LABEL_AST',
                'LABEL_Alkalinephos', 'LABEL_Bilirubin_total', 'LABEL_Lactate',
                'LABEL_TroponinI', 'LABEL_SaO2', 'LABEL_Bilirubin_direct',
                'LABEL_EtCO2']
labels_sepsis = ['LABEL_Sepsis']
labels_VS_mean = ['LABEL_RRate', 'LABEL_ABPm', 'LABEL_SpO2', 'LABEL_Heartrate']
all_labels = labels_tests + labels_sepsis + labels_VS_mean

# Drop pid feature:
train_features = train_features.drop(labels="pid", axis=1)
test_features = test_features.drop(labels="pid", axis=1)

# #shuflle dataset
# rd_permutation = np.random.permutation(train_features.index)
# train_features = train_features.reindex(rd_permutation).set_index(np.arange(0, train_features.shape[0], 1))
# train_labels = train_labels.reindex(rd_permutation).set_index(np.arange(0, train_labels.shape[0], 1))

# Definition of test and val data size:
train_size = 15000
X_old_t1 = np.array(train_features)[0:train_size * 12, :]
X_val_t1 = np.array(train_features)[train_size * 12:, :]
X_test_t1 = np.array(test_features)

#select_features = vital_signs
#labels_VS_mean = ['LABEL_RRate', 'LABEL_ABPm', 'LABEL_SpO2', 'LABEL_Heartrate']
#select_features = 'Heartrate'
select_features = vital_signs

X_old_t3 = np.array(train_features[select_features])[0:train_size * 12,:]
X_val_t3 = np.array(train_features[select_features])[train_size * 12:,:]
X_test_t3 = np.array(test_features[select_features])

# Standardize the data
X_old_t1 = (X_old_t1 - np.mean(X_old_t1, 0)) / np.std(X_old_t1, 0)
X_val_t1 = (X_val_t1 - np.mean(X_val_t1, 0)) / np.std(X_val_t1, 0)
X_test_t1 = (X_test_t1 - np.mean(X_test_t1, 0)) / np.std(X_test_t1, 0)

# Standardize the data
X_old_t3 = (X_old_t3 - np.mean(X_old_t3, 0)) / np.std(X_old_t3, 0)
X_val_t3 = (X_val_t3 - np.mean(X_val_t3, 0)) / np.std(X_val_t3, 0)
X_test_t3 = (X_test_t3 - np.mean(X_test_t3, 0)) / np.std(X_test_t3, 0)

# ---------------------------------------------
# -----------  MODEL FOR TASK 1 ---------------
# ---------------------------------------------

input_shape = (12, X_old_t1.shape[1])


model_t1 = Sequential()
model_t1.add(Conv1D(40, kernel_size=4,
                   activation='relu',
                   input_shape=input_shape))
#model_t1.add(Conv1D(20, 3, activation='relu'))
#model_t1.add(MaxPooling1D(pool_size=2))
model_t1.add(LSTM(20, return_sequences=False, input_shape=input_shape))
#model_t1.add(LSTM(5))
model_t1.add(Dropout(0.25))
#model_t1.add(Flatten())
#model_t1.add(Dense(20, activation='relu'))
#model_t1.add(Dropout(0.25))
model_t1.add(Dense(2, activation='softmax'))


model_t1.compile(optimizer=keras.optimizers.Adam(),
                 loss=keras.losses.sparse_categorical_crossentropy,
                 metrics=[keras.metrics.categorical_accuracy])

# ---------------------------------------------
# -----------  MODEL FOR TASK 3 ---------------
# ---------------------------------------------

try:
    N_features = X_old_t3.shape[1]
except IndexError:
    N_features = 1
input_shape = (12, N_features)

model_t3 = Sequential()
model_t3.add(Conv1D(1, kernel_size=4,
                     activation='relu',
                     input_shape=input_shape))
# # #model_t3.add(Conv1D(15, 3, activation='relu'))
# # model_t3.add(MaxPooling1D(pool_size=2))
# # #model_t3.add(Dropout(0.25))
# model_t3.add(Flatten())
# # model_t3.add(Dense(15, activation='relu'))
# # model_t3.add(Dropout(0.25))
# model_t3.add(Dense(1, activation='relu'))


model_t3.add(LSTM(5, return_sequences=False, input_shape=input_shape))
model_t3.add(Dropout(0.25))
model_t3.add(Dense(1, activation='relu'))
# #model_t3.add(Dropout(0.25))

model_t3.compile(optimizer=keras.optimizers.Adam(learning_rate=10e-4),
                 loss=keras.losses.mean_squared_error,
                 metrics=[keras.metrics.categorical_accuracy])

#reshape for input in the NN
N_patients_train_t3 = np.array(X_old_t3.shape[0] / 12).astype(int)
N_patients_test_t3 = np.array(X_test_t3.shape[0] / 12).astype(int) ## ONLY IN VALIDATION PHASE
N_patients_val_t3 = np.array(X_val_t3.shape[0] / 12).astype(int)  ## ONLY IN VALIDATION PHASE
#these dataframe will contain every prediction
Y_test_tot = pd.DataFrame(np.zeros([N_patients_test_t3, len(all_labels)]), columns=all_labels) #predictions for test set
Y_val_tot = pd.DataFrame(np.zeros([N_patients_val_t3, len(all_labels)]), columns=all_labels) #predictions for val set


# --------------------------------------------------------
# ------------------- TRAINING TASK 1 --------------------
# ---------------------------------------------------------
N_patients_train_t1 = np.array(X_old_t1.shape[0] / 12).astype(int)
N_patients_val_t1 = np.array(X_val_t1.shape[0] / 12).astype(int)  ## ONLY IN VALIDATION PHASE
N_patients_test_t1 = np.array(X_test_t1.shape[0] / 12).astype(int)

X_t1 = X_old_t1.reshape((N_patients_train_t1, 12, X_old_t1.shape[1]))
X_val_t1 = X_val_t1.reshape((N_patients_val_t1, 12, X_val_t1.shape[1]))
X_test_t1 = X_test_t1.reshape((N_patients_test_t1, 12, X_test_t1.shape[1]))

print(model_t1.summary())
epochs_t1 = 1
labels_target = labels_tests + ['LABEL_Sepsis']
# labels_target = ['LABEL_BaseExcess','LABEL_EtCO2','LABEL_SaO2']
# labels_target = ['LABEL_BaseExcess']
# labels_target = ['LABEL_SaO2']
# labels_target = ['LABEL_Sepsis']
scores = []
for i in range(0, len(labels_target)):
    stay = True
    label_target = labels_target[i]
    Y_t1 = train_labels[label_target].iloc[0:train_size]
    Y_val_t1 = train_labels[label_target].iloc[train_size:]

    #find class_weights
    weight0 = (Y_t1.shape[0] + Y_val_t1.shape[0]) / (sum(Y_t1 != 0) + sum(Y_val_t1 != 0) + 1)
    weight1 = (Y_t1.shape[0] + Y_val_t1.shape[0]) / (sum(Y_t1 == 0) + sum(Y_val_t1 == 0) + 1)
    class_weights = {0: weight0, 1: weight1}

    #fit
    model_t1.fit(X_t1, Y_t1, epochs=epochs_t1, verbose=0, class_weight=class_weights)

    #predict and save into dataframe
    Y_val_pred = model_t1.predict(X_val_t1)
    Y_test_pred = model_t1.predict(X_test_t1)
    Y_val_tot.loc[:, label_target] = Y_val_pred[:, -1]
    Y_test_tot.loc[:, label_target] = Y_test_pred[:, -1]

    #score
    scores = scores + [np.mean([skmetrics.roc_auc_score(Y_val_t1, Y_val_pred[:, -1])])]
    print("ROC AUC -- score ", i, " ", label_target, " :", scores[i])
print("ROC AUC -- score tot: ", sum(scores)/len(scores))


# ---------------------------------------------------------
# ------------------- TRAINING TASK 3 --------------------
# ---------------------------------------------------------


X_val_old_t3 = X_val_t3
X_t3 = X_old_t3.reshape((N_patients_train_t3, 12, N_features))
X_val_t3 = X_val_t3.reshape((N_patients_val_t3, 12, N_features))
X_test_t3 = X_test_t3.reshape((N_patients_test_t3, 12, N_features))

print(model_t3.summary())
epochs_t3 = 1
# labels_target = ['LABEL_Heartrate']
labels_target = labels_VS_mean


for i in range(0, len(labels_target)):
    #get the set corresponding tu the feature
    label_target = labels_target[i]
    Y_t3 = train_labels[label_target].iloc[0:train_size]
    Y_val_t3 = train_labels[label_target].iloc[train_size:]

    #fit
    model_t3.fit(X_t3, Y_t3, epochs=epochs_t3, verbose=0)

    # fit
    reg = LinearRegression()
    reg.fit(X_old_t3[0::12,:], Y_t3)

    # predict and save into dataframe
    print()
    Y_val_pred = reg.predict(X_val_old_t3[0::12,:]).flatten()
    task2 = skmetrics.r2_score(Y_val_t3, Y_val_pred, sample_weight=None, multioutput='uniform_average')
    print("LR R2 score ", i, " ", label_target, " :", task2)

    # predict and save into dataframe
    Y_test_pred = model_t3.predict(X_test_t3)
    Y_val_pred = model_t3.predict(X_val_t3).flatten()
    Y_test_tot.loc[:, label_target] = Y_test_pred
    Y_val_tot.loc[:, label_target] = Y_val_pred

    #score
    task2 = skmetrics.r2_score(Y_val_t3, Y_val_pred, sample_weight=None, multioutput='uniform_average')
    print("R2 score ", i, " ", label_target, " :", task2)

# --------------------------------------------------------
# ------------------- SAVE ---------- --------------------
# --------------------------------------------------------
Y_val_tot.insert(0, 'pid', sample['pid'])
Y_test_tot.insert(0, 'pid', sample['pid'])
Y_val_tot.to_csv('../data/predictions.csv', header=True, index=False, float_format='%.7f')
Y_test_tot.to_csv('../data/submission.csv', header=True, index=False, float_format='%.7f')
