import numpy as np
import pandas as pd
from data_cleaning_functions import VS_cleaning
from data_cleaning_functions import tests_cleaning
import matplotlib.pyplot as plt
from progress.bar import IncrementalBar

# Data import:
train_features = pd.read_csv("data/train_features.csv")
test_features = pd.read_csv("data/test_features.csv")

# Informatons on the headers -- Extracting information:
head_caracteristics = ['pid', 'Time', 'Age',]
head_vitals = ['Heartrate', 'SpO2', 'ABPs', 'ABPm', 'ABPd', 'Temp', 'RRate']
head_tests = ['EtCO2', 'PTT', 'BUN', 'Lactate', 'Hgb', 'HCO3', 'BaseExcess', 'Fibrinogen', 'Phosphate', 'WBC', 'Creatinine', 'PaCO2', 'AST', 'FiO2', 'Platelets', 'SaO2', 'Glucose', 'Magnesium', 'Potassium', 'Calcium', 'Alkalinephos', 'Bilirubin_direct', 'Chloride', 'Hct','Bilirubin_total', 'TroponinI', 'pH']

######## TRAIN #######
VS_data = VS_cleaning(data_set=train_features, vitals=head_vitals, h_obs=12)
print(VS_data)
print(VS_data.shape)
tests_data, test_mean, test_std = tests_cleaning(data_set=train_features, tests=head_tests, h_obs=12, test_mean_in=None, test_std_in=None)
print(tests_data)
print(tests_data.shape)
columsn = head_caracteristics + head_vitals + head_tests
data_new  = np.column_stack((train_features[head_caracteristics].iloc[:], VS_data, tests_data))
data_new = pd.DataFrame(data=data_new, columns=columsn)
data_new.to_csv('data/train_features_clean_all_NEW.csv', header=True, index=False)








