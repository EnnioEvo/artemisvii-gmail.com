import pandas as pd
import numpy as np
import sklearn.metrics as skmetrics
from sklearn import svm
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import Lasso

np.random.seed(seed=317)
# from sklearn.metrics import classification_report, confusion_matrix

# ---------------------------------------------------------
# ------------ DATA IMPORT AND DEFINITIONS ----------------
# ---------------------------------------------------------

# cleaned data import:
train_features_all = pd.read_csv("../data/train_features_clean_all.csv")
train_features = pd.read_csv("../data/train_features_clean_wmean_diff.csv")
test_features = pd.read_csv("../data/test_features_clean_wmean_diff.csv")
train_labels = pd.read_csv("../data/train_labels.csv")
sample = pd.read_csv("../sample.csv")
stored_usefulness_matrix_t2 = pd.read_csv("../data/feature_selection/usefulness_matrix_t1_sum.csv", index_col=0)

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
shuffle = True
use_diff = True
features_selection = False
threshold = 4
relevance_selection = False
relevance_thr = 30
# ---------------------------------------------------------
# ----------------- DATA SELECTION ------------------------
# ---------------------------------------------------------
# select relevant samples
if relevance_selection:
    usefulness_column = stored_usefulness_matrix_t2[labels_sepsis].sort_values(labels_sepsis, ascending=[0])
    useful_features_mask = np.array(usefulness_column) >= 4
    useful_features = [feature for feature, mask in zip(usefulness_column.index, useful_features_mask) if mask]
    useful_features_augmented = sum([[feature, 'dummy_' + feature] for feature in useful_features if feature in tests],
                                    []) + \
                                [feature for feature in useful_features if feature in vital_signs]
    train_dummy = train_features[[feature for feature in useful_features_augmented if feature in dummy_tests]]
    relevance_train = np.dot(train_dummy, np.array(usefulness_column)[:10])
    relevance_train_frame = pd.DataFrame(index=train_features.index, columns=['Relevance'], data=relevance_train)
    train_features = train_features[np.array(relevance_train_frame) >= relevance_thr]
    train_labels = train_labels[np.array(relevance_train_frame) >= relevance_thr]

if shuffle:
    rd_permutation = np.random.permutation(train_features.index)
    train_features = train_features.reindex(rd_permutation).set_index(np.arange(0, train_features.shape[0], 1))
    train_labels = train_labels.reindex(rd_permutation).set_index(np.arange(0, train_labels.shape[0], 1))


# ---------------------------------------------------------
# ----------------- DATA NORMALIZATION ------------------------
# ---------------------------------------------------------
def build_set(selected_features, train_size):
    # Definition of test and val data size:
    # task 1
    X = train_features.loc[0:train_size - 1, selected_features]
    X_val = train_features.loc[train_size:, selected_features]
    X_test = test_features[selected_features]

    # # add dummy features
    # X = np.column_stack([X, np.array(train_features.loc[0:train_size - 1, dummy_tests])])
    # X_val = np.column_stack([X_val, np.array(train_features.loc[train_size:, dummy_tests])])
    # X_test = np.column_stack([X_test, np.array(test_features[dummy_tests])])

    # Standardize the data
    X = (X - np.mean(X, 0)) / np.std(X, 0)
    X_val = (X_val - np.mean(X_val, 0)) / np.std(X_val, 0)
    X_test = (X_test - np.mean(X_test, 0)) / np.std(X_test, 0)

    return X, X_val, X_test


# Build sets
# task 1
train_size = int(train_features.shape[0] * 0.8)
selected_features_t2 = standard_features + dummy_tests
if use_diff:
    selected_features_t2 = selected_features_t2 + diff_features
#selected_features_t2 = vital_signs + diff_features
X_t2, X_val_t2, X_test_t2 = build_set(selected_features_t2, train_size)

# Variable for storing prediction
Y_test_tot = pd.DataFrame(np.zeros([X_test_t2.shape[0], len(labels_sepsis)]),
                          columns=labels_sepsis)  # predictions for test set
Y_val_tot = pd.DataFrame(np.zeros([X_val_t2.shape[0], len(labels_sepsis)]),
                         columns=labels_sepsis)  # predictions for val set

# --------------------------------------------------------
# ------------------- TRAINING TASK 2 --------------------
# ---------------------------------------------------------

labels_target = ['LABEL_Sepsis']
scores_t2 = []
for i in range(0, len(labels_target)):
    label_target = labels_target[i]
    Y_t2 = train_labels[label_target].iloc[0:train_size]
    Y_val_t2 = train_labels[label_target].iloc[train_size:]

    # # find class_weights
    # weight0 = (Y_t2.shape[0] + Y_val_t2.shape[0]) / (sum(Y_t2 != 0) + sum(Y_val_t2 != 0) + 1)
    # weight1 = (Y_t2.shape[0] + Y_val_t2.shape[0]) / (sum(Y_t2 == 0) + sum(Y_val_t2 == 0) + 1)
    # class_weights = {0: weight0, 1: weight1}
    # #class_weights = dict(zip())

    if features_selection:
        usefulness_column = stored_usefulness_matrix_t2[label_target].sort_values(ascending=False)
        useful_features_mask = np.array(usefulness_column) >= threshold
        useful_features = [feature for feature, mask in zip(usefulness_column.index, useful_features_mask) if mask]
        useful_features_augmented = sum(
            [[feature, 'dummy_' + feature] for feature in useful_features if feature in tests], []) + \
                                    [feature for feature in useful_features if feature in vital_signs] + \
                                    sum([sum(
                                        [[feature + suffix] for feature in useful_features if feature in vital_signs],
                                        []) for suffix in diff_features_suffixes], [])
        X_t2_useful = X_t2[list(set(useful_features_augmented) & set(X_t2.columns))]
        X_val_t2_useful = X_val_t2[list(set(useful_features_augmented) & set(X_t2.columns))]
        X_test_t2_useful = X_test_t2[list(set(useful_features_augmented) & set(X_t2.columns))]
    else:
        X_t2_useful = X_t2
        X_val_t2_useful = X_val_t2
        X_test_t2_useful = X_test_t2

    # fit
    clf = svm.LinearSVC(C=10e-4, class_weight='balanced', tol=10e-3, verbose=0)
    # clf = svm.SVC(C=10e-4, class_weight='balanced', tol=10e-3, verbose=0, kernel='rbf')
    clf.fit(X_t2_useful, Y_t2)

    # predict and save into dataframe
    Y_temp = np.array([clf.decision_function(X_val_t2_useful)])
    Y_val_pred = (1 / (1 + np.exp(-Y_temp))).flatten()
    Y_temp = np.array([clf.decision_function(X_test_t2_useful)])
    Y_test_pred = (1 / (1 + np.exp(-Y_temp))).flatten()
    Y_val_tot.loc[:, label_target] = Y_val_pred
    Y_test_tot.loc[:, label_target] = Y_test_pred

    score = np.mean([skmetrics.roc_auc_score(Y_val_t2, Y_val_pred)])
    scores_t2 = scores_t2 + [score]
    #print("ROC AUC -- score ", i, " ", label_target, " :", score)

task2 = scores_t2[-1]
print("ROC AUC task2 score ", task2)

# save into file
# Y_val_tot.insert(0, 'pid', sample['pid'])
# Y_test_tot.insert(0, 'pid', sample['pid'])
# Y_val_tot.to_csv('../data/predictions.csv', header=True, index=False, float_format='%.7f')
# Y_test_tot.to_csv('../data/submission.csv', header=True, index=False, float_format='%.7f')
# Y_test_tot.to_csv('../data/submission.zip', header=True, index=False, float_format='%.7f', compression='zip')
