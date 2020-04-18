import pandas as pd
import numpy as np
import sklearn.metrics as skmetrics
from sklearn import svm
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.ensemble import BaggingClassifier

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, BatchNormalization, Dense,Activation
from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras.losses import binary_crossentropy

np.random.seed(seed=397)
# from sklearn.metrics import classification_report, confusion_matrix

# ---------------------------------------------------------
# ------------ DATA IMPORT AND DEFINITIONS ----------------
# ---------------------------------------------------------

# cleaned data import:
url = 'https://github.com/EnnioEvo/task2/blob/master/data/train_features_clean_all.zip?raw=true'
train_features_NN = pd.read_csv(url, compression='zip', header=0, sep=',', quotechar='"')
url_labels = 'https://raw.githubusercontent.com/EnnioEvo/task2/master/Code_Alberto/data/train_labels.csv'
train_labels_NN = pd.read_csv(url_labels)

train_features = pd.read_csv("../data/train_features_clean_wmean_diff.csv")
test_features = pd.read_csv("../data/test_features_clean_wmean_diff.csv")
train_labels = pd.read_csv("../data/train_labels.csv")
sample = pd.read_csv("../sample.csv")
stored_usefulness_matrix_t1 = pd.read_csv("../data/feature_selection/usefulness_matrix_t1_sum.csv", index_col=0)
stored_usefulness_matrix_t3 = pd.read_csv("../data/feature_selection/usefulness_matrix_t3_sum.csv", index_col=0)

# features
patient_characteristics = ["Age"]  # TIME VARIABLE IS EXCLUDED
vital_signs = ["Heartrate", "SpO2", "ABPs", "ABPm", "ABPd", "RRate", 'Temp']
tests = ['EtCO2', 'PTT', 'BUN', 'Lactate', 'Hgb', 'HCO3', 'BaseExcess',
         'Fibrinogen', 'Phosphate', 'WBC', 'Creatinine', 'PaCO2', 'AST', 'FiO2',
         'Platelets', 'SaO2', 'Glucose', 'Magnesium', 'Potassium', 'Calcium',
         'Alkalinephos', 'Bilirubin_direct', 'Chloride', 'Hct',
         'Bilirubin_total', 'TroponinI', 'pH']
dummy_tests = ['dummy_' + test for test in tests]
standard_features = patient_characteristics + vital_signs + tests
diff_features_suffixes = ['_n_extrema', '_diff_mean', '_diff_median', '_diff_max', '_diff_min']
diff_features = sum(
    [[VS + diff_features_suffix for VS in vital_signs] for diff_features_suffix in diff_features_suffixes], [])
all_features = standard_features + dummy_tests + diff_features

# labels
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

# ---------------------------------------------------------
# ----------------- SET PARAMETERS ------------------------
# ---------------------------------------------------------
use_diff = False
features_selection = True
dummy = False
threshold = 4
shuffle = True
np.set_printoptions(precision=4)
submit = False
# ---------------------------------------------------------
# ----------------- DATA SELECTION ------------------------
# ---------------------------------------------------------

if shuffle:
    rd_permutation = np.random.permutation(train_features.index)
    train_features = train_features.reindex(rd_permutation).set_index(np.arange(0, train_features.shape[0], 1))
    train_labels = train_labels.reindex(rd_permutation).set_index(np.arange(0, train_labels.shape[0], 1))


def build_set(selected_features, train_size, submit):
    # Definition of test and val data size:
    # task 1
    if submit:
        X = train_features.loc[:, selected_features]
        X_val = train_features.loc[train_size:, selected_features]
        X_test = test_features[selected_features]
    else:
        X = train_features.loc[0:train_size - 1, selected_features]
        X_val = train_features.loc[train_size:, selected_features]
        X_test = test_features[selected_features]
     
    # add dummy features
    if dummy:
        X = np.column_stack([X, np.array(train_features.loc[0:train_size - 1, dummy_tests])])
        X_val = np.column_stack([X_val, np.array(train_features.loc[train_size:, dummy_tests])])
        X_test = np.column_stack([X_test, np.array(test_features[dummy_tests])])

    # Standardize the data
    X = (X - np.mean(X, 0)) / np.std(X, 0)
    X_val = (X_val - np.mean(X_val, 0)) / np.std(X_val, 0)
    X_test = (X_test - np.mean(X_test, 0)) / np.std(X_test, 0)

    return X, X_val, X_test


# Build sets
# task 1
train_size = int(train_features.shape[0] * 0.8)
selected_features_t1 = standard_features + dummy_tests
if use_diff:
    selected_features_t1 = selected_features_t1 + diff_features
#selected_features_t2 = vital_signs + diff_features
X_t1, X_val_t1, X_test_t1 = build_set(selected_features_t1, train_size, submit)
X_t2, X_val_t2, X_test_t2 = build_set(selected_features_t1, train_size, submit)

# task3
train_size = int(train_features.shape[0] * 0.8)
selected_features_t3 = selected_features_t1
X_t3, X_val_t3, X_test_t3 = build_set(selected_features_t3, train_size, submit)

#Variable for storing prediction
Y_test_tot = pd.DataFrame(np.zeros([X_test_t3.shape[0], len(all_labels)]),
                          columns=all_labels)  # predictions for test set
Y_val_tot = pd.DataFrame(np.zeros([X_val_t3.shape[0], len(all_labels)]), columns=all_labels)  # predictions for val set

# --------------------------------------------------------
# ------------------- TRAINING TASK 1 --------------------
# ---------------------------------------------------------

labels_target = labels_tests
scores_t1 = []
for i in range(0, len(labels_target)):
    label_target = labels_target[i]
    Y_t1 = train_labels[label_target].iloc[0:train_size]
    Y_val_t1 = train_labels[label_target].iloc[train_size:]

    if submit:
        Y_t1 = train_labels[label_target].iloc[:]
        Y_val_t1 = train_labels[label_target].iloc[train_size:]

    # # find class_weights
    # weight0 = (Y_t1.shape[0] + Y_val_t1.shape[0]) / (sum(Y_t1 != 0) + sum(Y_val_t1 != 0) + 1)
    # weight1 = (Y_t1.shape[0] + Y_val_t1.shape[0]) / (sum(Y_t1 == 0) + sum(Y_val_t1 == 0) + 1)
    # class_weights = {0: weight0, 1: weight1}
    # #class_weights = dict(zip())

    if features_selection:
        usefulness_column = stored_usefulness_matrix_t1[label_target].sort_values(ascending=False)
        useful_features_mask = np.array(usefulness_column) >= threshold
        useful_features = [feature for feature, mask in zip(usefulness_column.index, useful_features_mask) if mask]
        useful_features_augmented = sum(
            [[feature, 'dummy_' + feature] for feature in useful_features if feature in tests], []) + \
                                    [feature for feature in useful_features if feature in vital_signs] + \
                                    sum([sum(
                                        [[feature + suffix] for feature in useful_features if feature in vital_signs],
                                        []) for suffix in diff_features_suffixes], [])
        X_t1_useful = X_t1[list(set(useful_features_augmented) & set(X_t1.columns))]
        X_val_t1_useful = X_val_t1[list(set(useful_features_augmented) & set(X_t1.columns))]
        X_test_t1_useful = X_test_t1[list(set(useful_features_augmented) & set(X_t1.columns))]
    else:
        X_t1_useful = X_t1
        X_val_t1_useful = X_val_t1
        X_test_t1_useful = X_test_t1


    # fit
    # clf = svm.LinearSVC(C=10e-4, class_weight='balanced', tol=10e-3, verbose=0)
    clf = svm.SVC(C=0.1, kernel='rbf', tol=0.0001)
    clf.fit(X_t1_useful, Y_t1)

    # predict and save into dataframe
    Y_temp = np.array([clf.decision_function(X_val_t1_useful)])
    Y_val_pred = (1 / (1 + np.exp(-Y_temp))).flatten()
    Y_temp = np.array([clf.decision_function(X_test_t1_useful)])

    Y_test_pred = (1 / (1 + np.exp(-Y_temp))).flatten()
    Y_val_tot.loc[:, label_target] = Y_val_pred
    Y_test_tot.loc[:, label_target] = Y_test_pred

    score = np.mean([skmetrics.roc_auc_score(Y_val_t1, Y_val_pred)])
    scores_t1 = scores_t1 + [score]
    print("ROC AUC -- score ", i, " ", label_target, " :", score)

task1 = sum(scores_t1[:-1]) / len(scores_t1[:-1])
print("ROC AUC task1 score  ", task1)
task2 = scores_t1[-1]
print("ROC AUC task2 score ", task2)

# usefulness_matrix.to_csv('../data/usefulness_matrix.csv', header=True, index=True, float_format='%.7f')

# --------------------------------------------------------
# ------------------- TRAINING TASK 2 --------------------
# ---------------------------------------------------------
scaler = preprocessing.StandardScaler().fit(np.array(train_features_NN))
X = scaler.transform(np.array(train_features_NN))
X_test = scaler.transform(np.array(test_features_NN))

N_patients_train = np.array(X.shape[0]/12).astype(int)
N_patients_test = np.array(X_test.shape[0]/12).astype(int)

train_size = 15000
Y_t2 = train_labels_NN[labels_sepsis].iloc[0:train_size]
Y_val_t2 = train_labels_NN[labels_sepsis].iloc[train_size:]

X = np.array(X).reshape((N_patients_train, 12, X.shape[1], 1))
X_test = np.array(X_test).reshape((N_patients_test, 12, X_test.shape[1], 1))

print("train_set shape: ", X.shape)
print("test_set shape: ", X_test.shape)

# CNet parameters:
epochs_t1 = 50
input_shape =(12, X.shape[2],1)

model_t1 = tf.keras.models.Sequential()

model_t1.add(Conv2D(32, 
                    kernel_size=(2,2),
                    input_shape=input_shape, 
                    ))

model_t1.add(BatchNormalization())
model_t1.add(Activation('relu'))
model_t1.add(MaxPooling2D(pool_size=(2,2)))


model_t1.add(Conv2D(64, 
                    kernel_size=(2,2),
                    input_shape=input_shape 
                    ))

model_t1.add(BatchNormalization())
model_t1.add(Activation('relu'))
model_t1.add(MaxPooling2D(pool_size=(2,2)))

model_t1.add(Conv2D(128, 
                    kernel_size=(2,8),
                    input_shape=input_shape 
                    ))

model_t1.add(Flatten())

model_t1.add(BatchNormalization())
model_t1.add(Dense(500, activation='relu'))
model_t1.add(Dropout(0.5))

model_t1.add(Dense(1, activation='sigmoid')) 

model_t1.compile(optimizer=Adadelta(),
                loss=binary_crossentropy
                )
                
print(model_t1.summary())

print("########## test ", ['LABEL_Sepsis'], " ##########")

# Class_weights:
a = Y_t1.shape[0] / (2 * np.array([Y_t1.shape[0]- np.sum(np.array(Y_t1)), np.sum(np.array(Y_t1))]))
weights = {
    0 : a[0],
    1 : a[1]
    }

#Y = keras.utils.to_categorical(Y_t1)
model_t1.fit(X, Y_t1, epochs=epochs_t1, class_weight=weights)

Y_val_pred_2 = model_t1.predict(X_test) 
print(Y_val_pred)
print(Y_val_t2)
task2 = np.mean([skmetrics.roc_auc_score(Y_val_t2, Y_val_pred_2.flatten())])
print("ROC AUC, test: ", test," -- task 1 score: ", task2)


# ---------------------------------------------------------
# ------------------- TRAINING TASK 3 --------------------
# ---------------------------------------------------------

labels_target = labels_VS_mean
# labels_target = ['LABEL_' + select_feature for select_feature in select_features]
scores_t3 = []
for i in range(0, len(labels_target)):
    # get the set corresponding tu the feature
    label_target = labels_target[i]
    Y_t3 = train_labels[label_target].iloc[0:train_size]
    Y_val_t3 = train_labels[label_target].iloc[train_size:]

    if submit:
        Y_t3 = train_labels[label_target].iloc[:]
        Y_val_t3 = train_labels[label_target].iloc[train_size:]

    if features_selection:
        usefulness_column = stored_usefulness_matrix_t3[label_target].sort_values(ascending=False)
        useful_features_mask = np.array(usefulness_column) >= threshold
        useful_features = [feature for feature, mask in zip(usefulness_column.index, useful_features_mask) if mask]
        useful_features_augmented = sum(
            [[feature, 'dummy_' + feature] for feature in useful_features if feature in tests], []) + \
                                    [feature for feature in useful_features if feature in vital_signs] + \
                                    sum([sum(
                                        [[feature + suffix] for feature in useful_features if feature in vital_signs],
                                        []) for suffix in diff_features_suffixes], [])
        X_t3_useful = X_t3[list(set(useful_features_augmented) & set(X_t3.columns))]
        X_val_t3_useful = X_val_t3[list(set(useful_features_augmented) & set(X_t3.columns))]
        X_test_t3_useful = X_test_t3[list(set(useful_features_augmented) & set(X_t3.columns))]
    else:
        X_t3_useful = X_t3
        X_val_t3_useful = X_val_t3
        X_test_t3_useful = X_test_t3

    # fit
    #reg = LinearRegression()
    reg = Lasso(alpha=0.001,fit_intercept=True)
    reg.fit(X_t3_useful, Y_t3)

    # predict and save into dataframe
    Y_test_pred = reg.predict(X_test_t3_useful).flatten()
    Y_val_pred = reg.predict(X_val_t3_useful).flatten()
    Y_test_tot.loc[:, label_target] = Y_test_pred
    Y_val_tot.loc[:, label_target] = Y_val_pred

    score
    score = 0.5 + 0.5 * skmetrics.r2_score(Y_val_t3, Y_val_pred, sample_weight=None, multioutput='uniform_average')
    scores_t3 = scores_t3 + [score]
    print("R2 score ", i, " ", label_target, " :", score)

task3 = np.mean(scores_t3)
print("Task3 score = ", task3)
print("Total score = ", np.mean([task1, task2, task3]))

# save into file
Y_val_tot.insert(0, 'pid', sample['pid'])
Y_test_tot.insert(0, 'pid', sample['pid'])
Y_val_tot.to_csv('../data/predictions.csv', header=True, index=False, float_format='%.7f')
Y_test_tot.to_csv('submission.csv', header=True, index=False, float_format='%.7f')
Y_test_tot.to_csv('submission.zip', header=True, index=False, float_format='%.7f', compression='zip')
