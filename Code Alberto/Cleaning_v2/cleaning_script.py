#####################  -- CLEANING DATA ALGORITHM -- #####################
#LEGEND --> VS= vital signs, tests: medical tests (from data)
#           train: training data, test: testa data to be submitted
import pandas as pd
import numpy as np
import time
from cleaning_functions import test_clean_aggregation
from cleaning_functions_v2 import VS_clean_aggregation

# Imputation and claning:
# Data import:
train_features = pd.read_csv("data/train_features.csv")
test_features = pd.read_csv("data/test_features.csv")
print(train_features.shape)
print(test_features.shape)
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

# TRAIN FEATURES:
print("---------- TRAIN FEATURES ----------")
# VS Features:
start = time.time()
data_VS_train = VS_clean_aggregation(train_features, N_patients_train, vital_signs, hours_obs=12) #!!!!!!!! CORREGGI N_PATIENTS
end = time.time()
print("TRAIN FEATURES - VS features >>Execution time: ", end - start)
# Tests features:
start2 = time.time()
data_tests_train, test_max_train, test_min_train = test_clean_aggregation(train_features, N_patients_train, tests, hours_obs=12, min_tests=None, max_tests=None)
end2 = time.time()
# Saving:
AA = [True, False, False, False, False, False, False, False, False, False, False, False]
bool_vect_train = np.tile(AA, N_patients_train)
data_patients = train_features[patient_characteristics].loc[bool_vect_train]
data_set_clean = np.column_stack((data_patients, data_VS_train, data_tests_train))
col = patient_characteristics + vital_signs + tests
tests_Set = pd.DataFrame(data_set_clean, index=None, columns=col)
tests_Set.to_csv('data/train_features_clean.csv', header=True, index=False)
print()
print()

# TEST FEATURES: 
print("---------- TEST FEATURES ----------")
# VS features
start3 = time.time()
data_VS_test = VS_clean_aggregation(test_features, N_patients_test, vital_signs, hours_obs=12)
end3 = time.time()
print("TEST FEATURES - VS features >> Execution time: ", end3 - start3)
# tests features
start4 = time.time()
data_tests_test, test_max, test_min = test_clean_aggregation(test_features, N_patients_test, tests, hours_obs=12, min_tests=test_min_train, max_tests=test_max_train)
end4 = time.time()
print("TEST FEATURES - tests features >> Execution time: ", end4 - start4)
# Saving 
bool_vect_test = np.tile(AA, N_patients_test)
data_patients = test_features[patient_characteristics].loc[bool_vect_test]
data_set_clean = np.column_stack((data_patients, data_VS_test, data_tests_test))
col = patient_characteristics + vital_signs + tests
submSet = pd.DataFrame(data_set_clean, index=None, columns=col)
submSet.to_csv('data/test_features_clean.csv', header=col, index=False)
print(data_set_clean.shape)
print(submSet.shape)