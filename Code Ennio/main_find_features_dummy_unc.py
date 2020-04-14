import pandas as pd
import numpy as np
import sklearn.metrics as skmetrics
from sklearn import svm
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import Lasso
import copy

np.random.seed(seed=179)
# from sklearn.metrics import classification_report, confusion_matrix

# ---------------------------------------------------------
# ------------ DATA IMPORT AND DEFINITIONS ----------------
# ---------------------------------------------------------

# cleaned data import:
train_features = pd.read_csv("../data/train_features_clean_mean.csv")
test_features = pd.read_csv("../data/test_features_clean_mean.csv")
train_labels = pd.read_csv("../data/train_labels.csv")
sample = pd.read_csv("../sample.csv")
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
all_features = standard_features + dummy_tests

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
# ----------------- DATA SELECTION ------------------------
# ---------------------------------------------------------

def build_set(selected_features, train_size):

    # Definition of test and val data size:
    # task 1
    X = np.array(train_features.loc[0:train_size - 1, selected_features])
    X_val = np.array(train_features.loc[train_size:, selected_features])
    X_test = np.array(test_features[selected_features])

    # add dummy features
    X = np.column_stack([X, np.array(train_features.loc[0:train_size - 1, dummy_tests])])
    X_val = np.column_stack([X_val, np.array(train_features.loc[train_size:, dummy_tests])])
    X_test = np.column_stack([X_test, np.array(test_features[dummy_tests])])

    # Standardize the data
    X = (X - np.mean(X, 0)) / np.std(X, 0)
    X_val = (X_val - np.mean(X_val, 0)) / np.std(X_val, 0)
    X_test = (X_test - np.mean(X_test, 0)) / np.std(X_test, 0)


    return X, X_val, X_test




# ---------------------------------------------------------
# ------------------- TRAINING TASK 3 --------------------
# ---------------------------------------------------------

labels_target = labels_VS_mean
for k in range(10):
    # shuffle
    rd_permutation = np.random.permutation(train_features.index)
    train_features = train_features.reindex(rd_permutation).set_index(np.arange(0, train_features.shape[0], 1))
    train_labels = train_labels.reindex(rd_permutation).set_index(np.arange(0, train_labels.shape[0], 1))

    train_size = 15000
    selected_features_t3 = standard_features
    X_t3, X_val_t3, X_test_t3 = build_set(selected_features_t3, train_size)

    usefulness_matrix_t3 = pd.DataFrame(index=standard_features, columns=labels_target)
    scores_t3 = []
    for i in range(0, len(labels_target)):
        # get the set corresponding tu the feature
        label_target = labels_target[i]
        Y_t3 = train_labels[label_target].iloc[0:train_size]
        Y_val_t3 = train_labels[label_target].iloc[train_size:]

        # fit
        reg = LinearRegression()
        reg.fit(X_t3, Y_t3)
        # reg = Lasso(alpha=0.01)
        # reg.fit(X_train_subtask3, np.ravel(Y_train))

        # score
        Y_val_pred = reg.predict(X_val_t3).flatten()
        score = 0.5 + 0.5 * skmetrics.r2_score(Y_val_t3, Y_val_pred, sample_weight=None, multioutput='uniform_average')
        scores_t3 = scores_t3 + [score]
        print("R2 score initial features ", i, " ", label_target, " :", score)

        useful_features_mask = np.repeat([True], len(standard_features))
        for _ in range(10):
            useful_features_mask_temp = useful_features_mask
            for j in range(0, len(standard_features)):
                # build a new mask
                new_useful_features_mask = copy.deepcopy(useful_features_mask)
                new_useful_features_mask[j] = ~new_useful_features_mask[j]
                long_useful_features_mask = np.concatenate(
                    (new_useful_features_mask[:len(vital_signs) + len(tests) + 1], new_useful_features_mask[len(vital_signs) + 1:])
                )

                # mask the dataset
                X_t3_useful = X_t3[:, long_useful_features_mask]
                X_val_t3_useful = X_val_t3[:, long_useful_features_mask]

                # fit
                reg = LinearRegression()
                reg.fit(X_t3_useful, Y_t3)

                # predict
                Y_val_pred = reg.predict(X_val_t3_useful).flatten()

                # score
                new_score = 0.5 + 0.5 * skmetrics.r2_score(Y_val_t3, Y_val_pred, sample_weight=None, multioutput='uniform_average')
                print("Removed: ", standard_features[j], "\nscore ", label_target, " :", new_score)
                if new_score > score:
                    score = new_score
                    useful_features_mask = copy.deepcopy(new_useful_features_mask)
            if np.all(
                    useful_features_mask_temp == useful_features_mask):  # if the list of useful features has not changed exit
                break

        # print useful and useless features
        useful_features = [d for d, s in zip(standard_features, useful_features_mask) if s]
        print('\nUseful features for label ', label_target, ':\n')
        print(useful_features)
        useless_features = [d for d, s in zip(standard_features, ~useful_features_mask) if s]
        print('\nUseless features for label ', label_target, ':\n')
        print(useless_features)

        # upload usefulness matrix
        usefulness_matrix_t3[label_target] = useful_features_mask

        # print score
        scores_t3 = scores_t3 + [score]
        print("ROC AUC final score ", i, " ", label_target, " :", score, '\n')

        # mask with found useful features
        long_useful_features_mask = np.concatenate(
            (new_useful_features_mask[:len(vital_signs) + len(tests) + 1], new_useful_features_mask[len(vital_signs) + 1:])
        )
        X_test_t3_useful = X_test_t3[:, long_useful_features_mask]
        X_t3_useful = X_t3[:, long_useful_features_mask]

        # last fit for current label
        reg = LinearRegression()
        reg.fit(X_t3_useful, Y_t3)

        # predict
        Y_test_pred = reg.predict(X_test_t3_useful).flatten()

    scores_t3 = [score_t3 for score_t3 in scores_t3]

    print("Task 3 single scores: ")
    for score_t3 in scores_t3:
        print(score_t3, "\n")

    task3 = np.mean(scores_t3)
    print("Task3 total score: ", task3)
    # print("Total score = ", np.mean([task1, task2, task3]))

    usefulness_matrix_t3.to_csv('../data/feature_selection/usefulness_matrix_t3_dummy_'+ k + '.csv', header=True, index=True, float_format='%.7f')

# --------------------------------------------------------
# ------------------- TRAINING TASK 1 --------------------
# ---------------------------------------------------------

labels_target = labels_tests + ['LABEL_Sepsis']
train_size = 15000
selected_features_t1 = standard_features
for k in range(10):
    # shuffle
    rd_permutation = np.random.permutation(train_features.index)
    train_features = train_features.reindex(rd_permutation).set_index(np.arange(0, train_features.shape[0], 1))
    train_labels = train_labels.reindex(rd_permutation).set_index(np.arange(0, train_labels.shape[0], 1))


    X_t1, X_val_t1, X_test_t1 = build_set(selected_features_t1, train_size)
    usefulness_matrix_t1 = pd.DataFrame(index=standard_features, columns=labels_target)
    scores_t1 = []
    for i in range(0, len(labels_target)):
        label_target = labels_target[i]
        Y_t1 = train_labels[label_target].iloc[0:train_size]
        Y_val_t1 = train_labels[label_target].iloc[train_size:]

        # fit
        clf = svm.LinearSVC(C=1e-4, class_weight='balanced', tol=10e-2, verbose=0)
        clf.fit(X_t1, Y_t1)
        # predict
        Y_temp = np.array([clf.decision_function(X_val_t1)])
        Y_val_pred = (1 / (1 + np.exp(-Y_temp))).flatten()
        # score
        score = np.mean([skmetrics.roc_auc_score(Y_val_t1, Y_val_pred)])
        print("Removed: Nothing", "\nscore ", label_target, " :", score)

        useful_features_mask = np.repeat([True], len(standard_features))
        for _ in range(10):
            useful_features_mask_temp = useful_features_mask
            for j in range(0, len(standard_features)):
                # build a new mask
                new_useful_features_mask = copy.deepcopy(useful_features_mask)
                new_useful_features_mask[j] = ~new_useful_features_mask[j]
                long_useful_features_mask = np.concatenate(
                    (new_useful_features_mask[:len(vital_signs) + len(tests) + 1], new_useful_features_mask[len(vital_signs) + 1:])
                )
                # mask the dataset
                X_t1_useful = X_t1[:, long_useful_features_mask]
                X_val_t1_useful = X_val_t1[:, long_useful_features_mask]

                # fit
                clf = svm.LinearSVC(C=1e-4, class_weight='balanced', tol=10e-2, verbose=0)
                clf.fit(X_t1_useful, Y_t1)

                # predict
                Y_temp = np.array([clf.decision_function(X_val_t1_useful)])
                Y_val_pred = (1 / (1 + np.exp(-Y_temp))).flatten()

                # score
                new_score = np.mean([skmetrics.roc_auc_score(Y_val_t1, Y_val_pred)])
                print("Removed: ", standard_features[j], "\nscore ", label_target, " :", new_score)
                print()
                if new_score > score:
                    score = new_score
                    useful_features_mask = copy.deepcopy(new_useful_features_mask)
            if np.all(
                    useful_features_mask_temp == useful_features_mask):  # if the list of useful features has not changed exit
                break

        useful_features = [d for d, s in zip(standard_features, useful_features_mask) if s]
        print('\nUseful features for label ', label_target, ':\n')
        print(useful_features)
        useless_features = [d for d, s in zip(standard_features, ~useful_features_mask) if s]
        print('\nUseless features for label ', label_target, ':\n')
        print(useless_features)

        usefulness_matrix_t1[label_target] = useful_features_mask
        scores_t1 = scores_t1 + [score]
        print("ROC AUC final score ", i, " ", label_target, " :", score, '\n')

        # mask with found useful features
        long_useful_features_mask = np.concatenate(
            (new_useful_features_mask[:len(vital_signs) + len(tests) + 1], new_useful_features_mask[len(vital_signs) + 1:])
        )
        X_test_t1_useful = X_test_t1[:, long_useful_features_mask]
        X_t1_useful = X_t1[:, long_useful_features_mask]

        # last fit for current label
        clf = svm.LinearSVC(C=1e-4, class_weight='balanced', tol=10e-2, verbose=0)
        clf.fit(X_t1_useful, Y_t1)

        # predict and save
        Y_temp = np.array([clf.decision_function(X_test_t1_useful)])
        Y_test_pred = (1 / (1 + np.exp(-Y_temp))).flatten()

    task1 = sum(scores_t1[:-1]) / len(scores_t1[:-1])
    print("ROC AUC task1 score  ", task1)
    task2 = scores_t1[-1]
    print("ROC AUC task2 score ", task2)
    usefulness_matrix_t1.to_csv('../data/feature_selection/usefulness_matrix_t1_dummy_' + k + '.csv', header=True, index=True, float_format='%.7f')

