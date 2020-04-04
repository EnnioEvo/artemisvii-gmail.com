##################  -- CLEANING DATA ALGORITHM - data ALL -- ##################
# LEGEND --> VS= vital signs, tests: medical tests (from data)
#           train: training data, test: testa data to be submitted
#                   !TESTS -> standard normalization!
import numpy as np
import pandas as pd
import time
import sys
from progress.bar import IncrementalBar
print()
print("#############  -- CLEANING DATA ALGORITHM - data ALL -- #############")
# Imputation and claning:
# Data import:
train_features = pd.read_csv("data/train_features.csv")
test_features = pd.read_csv("data/test_features.csv")
# Informatons on the headers -- Extracting information:
patient_characteristics = ["pid", "Age"] # TIME VARIABLE IS EXCLUDED
vital_signs = ["Heartrate", "SpO2", "ABPs", "ABPm", "ABPd", "RRate", 'Temp']
tests = ['EtCO2', 'PTT', 'BUN', 'Lactate', 'Hgb', 'HCO3', 'BaseExcess',
       'Fibrinogen', 'Phosphate', 'WBC', 'Creatinine', 'PaCO2', 'AST', 'FiO2',
       'Platelets', 'SaO2', 'Glucose', 'Magnesium', 'Potassium', 'Calcium',
       'Alkalinephos', 'Bilirubin_direct', 'Chloride', 'Hct',
       'Bilirubin_total', 'TroponinI', 'pH']
headers_train = test_features.columns
headers_test = test_features.columns
N_patients_train = np.array(train_features.shape[0]/12).astype(int)
N_patients_test = np.array(test_features.shape[0]/12).astype(int)

# VS feature dataset:
def VS_imputation(data_set, N_patients, vital_signs, hours_obs):
    print("First imputation loop -- VS features")
    headers = data_set.columns
    h_vect = np.array([ii for ii in range(hours_obs)])
    VS_feature = data_set.loc[:,vital_signs] # .loc create a new dataframe
    VS_mean = np.nanmean(np.array(VS_feature), axis=0)
    data_clean = np.array([])
    nan_matrix = VS_feature.isna()
    bar1 = IncrementalBar('### VS features processing ### VS analyzed:', max = len(vital_signs))
    for VS in vital_signs:
        single_feature = np.array(VS_feature[VS])
        single_nan_matrix = np.array(nan_matrix[VS])
        for patient in range(N_patients):
            ii = 0
            bool_val = single_nan_matrix[patient*hours_obs:patient*hours_obs+hours_obs]
            if np.sum(np.array(bool_val)): # There exista a nan
                if np.prod(np.array(bool_val)): # The entire column have nan
                    single_feature[patient*hours_obs:patient*hours_obs+hours_obs] =  VS_mean[ii]*np.ones(h_vect.shape)
                    a = single_feature[patient*hours_obs:patient*hours_obs+hours_obs]
                    assert np.logical_not(np.sum(np.isnan(a))), "NAN FOUNDED IN VS where all data are nan, it: %r" % patient
                else:
                    X_nan = h_vect[bool_val]
                    Y_nan = single_feature[patient*hours_obs+X_nan]
                    X = h_vect[bool_val == False]
                    Y = single_feature[patient*hours_obs+X]
                    # MODEL OF IMPUTATION:
                    # regr.fit(X[:, None], Y)
                    # data_set[pat_row+X_nan, A] = regr.predict(X_nan[:, None])
                    # to increase speed: we use the mean
                    not_nan_mean = np.mean(Y)
                    single_feature[patient * hours_obs + X_nan] =  not_nan_mean
                    a = single_feature[patient*hours_obs:patient*hours_obs+hours_obs]
                    assert np.logical_not(np.sum(np.isnan(a))), "NAN FOUNDED IN VS where not all data are nan, it: %r" % patient
            ii = ii + 1
        data_clean = np.append(data_clean, single_feature)
        bar1.next()
    bar1.finish()
    data_clean = data_clean.reshape((N_patients*hours_obs, len(vital_signs)))
    data_clean = (data_clean - np.mean(data_clean, axis=0))/np.std(data_clean, axis=0)
    data_clean = pd.DataFrame(data=data_clean,columns=vital_signs)
    return data_clean  #MANCA PARTE DI AGGREGAZIONE DELLE SINGLE FEATURES
    

def test_clean_aggregation(data_set, N_patients, tests, hours_obs, min_tests, max_tests):
    print('Second imputation loop + aggregation -- tests features')
    ii = 0 
    if np.logical_not(np.any(min_tests)):
        test_min = np.nanmin(data_set[tests].loc[:], axis = 0)
    else:
        test_min = min_tests
    if np.logical_not(np.any(max_tests)):
        test_max = np.nanmax(data_set[tests].loc[:], axis = 0)
    else:
        test_max = max_tests       
    data_set_new = np.zeros([N_patients*hours_obs, len(tests)])
    bar2 = IncrementalBar('### TESTS features processing ### VS analyzed:', max = len(tests))
    for test in tests:
        test_col = np.array(data_set[test])
        test_col_mean = np.nanmean(test_col)
        for idx in range(test_col.shape[0]):
            if np.isnan(test_col[idx]):
                test_col[idx] = 0
            else:
                test_col[idx] = (test_col[idx] - test_min[ii])/(test_max[ii])
        data_set_new[:, ii] = test_col
        ii = ii + 1
        bar2.next()
        # if ii%5==0: print("Tests analyzed in loop: ", ii)
    bar2.finish()
    return data_set_new, test_max, test_min 

print()
# TRAIN FEATURES:
print("---------- TRAIN FEATURES ----------")
# VS Features:
start = time.time()
data_VS_train = VS_imputation(train_features, N_patients_train, vital_signs, hours_obs=12) 
end = time.time()
print("TRAIN FEATURES - VS features >>Execution time: ", end - start)
# Tests features:
start2 = time.time()
data_tests_train, test_max_train, test_min_train = test_clean_aggregation(train_features, N_patients_train, tests, hours_obs=12, min_tests=None, max_tests=None)
end2 = time.time()
# Saving:
data_patients = train_features[patient_characteristics]
data_set_clean = np.column_stack((data_patients, data_VS_train, data_tests_train))
col = patient_characteristics + vital_signs + tests
tests_Set = pd.DataFrame(data_set_clean, index=None, columns=col)
tests_Set.to_csv('data/train_features_clean_all_2.csv', header=True, index=False)
print("NaN still inside per columns: ", np.sum(np.isnan(np.array(tests_Set)), axis=0))
print()

# TEST FEATURES: 
print("---------- TEST FEATURES ----------")
# VS features
start3 = time.time()
data_VS_test = VS_imputation(test_features, N_patients_test, vital_signs, hours_obs=12)
end3 = time.time()
print("TEST FEATURES - VS features >> Execution time: ", end3 - start3)
# tests features
start4 = time.time()
data_tests_test, test_max, test_min = test_clean_aggregation(test_features, N_patients_test, tests, hours_obs=12, min_tests=test_min_train, max_tests=test_max_train)
end4 = time.time()
print("TEST FEATURES - tests features >> Execution time: ", end4 - start4)
# Saving 
data_patients = test_features[patient_characteristics]
data_set_clean = np.column_stack((data_patients, data_VS_test, data_tests_test))
col = patient_characteristics + vital_signs + tests
submSet = pd.DataFrame(data_set_clean, index=None, columns=col)
submSet.to_csv('data/test_features_clean_all_2.csv', header=col, index=False)
print("NaN still inside per columns: ", np.sum(np.isnan(np.array(submSet)), axis=0))