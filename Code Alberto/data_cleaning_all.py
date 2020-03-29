import numpy as np
import pandas as pd
import time
import sys
from progress.bar import IncrementalBar


# Data import:
train_features = pd.read_csv("data/train_features.csv")
test_features = pd.read_csv("data/test_features.csv")
train_labels = pd.read_csv("data/train_labels.csv")

# Extracting information:
patient_characteristics = ["pid", "Age"]
vital_signs = ["Heartrate", "SpO2", "ABPs", "ABPm", "ABPd", "RRate"]
tests = ['EtCO2', 'PTT', 'BUN', 'Lactate', 'Temp', 'Hgb', 'HCO3', 'BaseExcess',
       'Fibrinogen', 'Phosphate', 'WBC', 'Creatinine', 'PaCO2', 'AST', 'FiO2',
       'Platelets', 'SaO2', 'Glucose', 'Magnesium', 'Potassium', 'Calcium',
       'Alkalinephos', 'Bilirubin_direct', 'Chloride', 'Hct',
       'Bilirubin_total', 'TroponinI', 'pH']
headers = train_features.columns
N_patients = np.array(train_features.shape[0]/12).astype(int)


# Vital signs data -> mean:
# VS feature dataset:
def VS_imputation(data_set, N_patients, vital_signs, hours_obs):
    print("First imputation loop -- VS features")
    bar = IncrementalBar('### VS features processing ### patient analyzed:', max = N_patients)
    headers = data_set.columns
    h_vect = np.array([ii for ii in range(hours_obs)])
    VS_feature = data_set.loc[:,vital_signs] # .loc create a new dataframe
    VS_mean = np.nanmean(np.array(VS_feature), axis=0)
    nan_matrix = VS_feature.isna()
    for patient in range(N_patients):
        ii = 0
        for VS in vital_signs:
            bool_val = nan_matrix[VS].iloc[patient*hours_obs:patient*hours_obs+hours_obs]
            if np.sum(np.array(bool_val)): # There exista a nan
                if np.prod(np.array(bool_val)): # The entire column have nan
                    VS_feature[VS].iloc[patient*hours_obs:patient*hours_obs+hours_obs] =  VS_mean[ii]*np.ones(h_vect.shape)
                else:
                    X_nan = h_vect[bool_val]
                    Y_nan = VS_feature[VS].iloc[patient*hours_obs+X_nan]
                    X = h_vect[bool_val == False]
                    Y = VS_feature[VS].iloc[patient*hours_obs+X]
                    # MODEL OF IMPUTATION:
                    # regr.fit(X[:, None], Y)
                    # data_set[pat_row+X_nan, A] = regr.predict(X_nan[:, None])
                    # to increase speed: we use the mean
                    not_nan_mean = np.mean(Y)
                    VS_feature[VS].iloc[patient * hours_obs + X_nan] =  not_nan_mean
        ii = ii + 1
        bar.next()
    bar.finish()
    return VS_feature


def tests_cleaning(data_set_tests, tests):
    print("Second imputation loop -- tests features")
    test_min = np.ones(len(tests))
    test_max = np.ones(len(tests))
    col_headers = data_set_tests.columns
    tests_data = data_set_tests.loc[:, tests]
    shape_idx = tests_data.shape[0]
    ii = 0
    bar2 = IncrementalBar('tests features: test analyzed:', max = len(tests))
    for test in tests:
        test_col = np.array(tests_data.loc[:, test])
        test_min[ii] = np.nanmin(test_col)
        test_max[ii] = np.nanmax(test_col)
        tezt = '### Tests features processing ### test_feature in analysis: ' + str(ii+1) + '/' + str(len(tests))+' ...Analizing patients...'
        bar3 = IncrementalBar(tezt, max = shape_idx)
        for values_idx in np.arange(shape_idx):
            if np.isnan(test_col[values_idx]):
                tests_data.iloc[values_idx, ii] = 0
            else:
                tests_data.iloc[values_idx, ii] = 1 + (test_col[values_idx] - test_min[ii]) / (
                            test_max[ii] - test_min[ii])
            bar3.next()
        bar3.finish()
        ii = ii + 1
        print(ii)
        bar2.next()
    bar2.finish()
    return tests_data #, test_min, test_max 


# VS data cleaning:
start = time.time()
data_VS = VS_imputation(train_features, 3000, vital_signs, 12)
data = 1
end = time.time()
print("VS analysis -- Execution time: ", end - start)
print() 
print() 
# tests data cleaning:
start = time.time()
train_features_new_tests= tests_cleaning(train_features, tests)
end = time.time()
print("tests_features analysis -- Execution time: ", end - start)

# Elaborating data for saving:
data_set_clean = np.column_stack((train_features[patient_characteristics], data_VS, train_features_new_tests))
col = patient_characteristics + vital_signs + tests
print("data_vs shape: ", data_VS.shape)
print("train_features_new_tests shape: ", train_features_new_tests.shape)
print("train_features[patient_characteristics] shape: ", train_features[patient_characteristics].shape)
print('Len headers: ', len(col))
print('Data_set_clean shape: ', data_set_clean.shape)
print('## CHECK COMPLETE ##')

submSet = pd.DataFrame(data_set_clean, index=None, columns=col)
print(submSet.head())
print("Saved file shape: ", submSet.shape)
submSet.to_csv('data/data_set_clean_entire_dataset.csv', header=True, index=False)
