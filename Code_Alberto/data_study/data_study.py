import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Imputation and claning:
# Data import:
train_features = pd.read_csv("data/train_features.csv")
train_features_all = pd.read_csv("data/train_features_clean_all_no_norm.csv")
train_labels = pd.read_csv("data/train_labels.csv")
test_features = pd.read_csv("data/test_features.csv")
print(train_features_all.iloc[190602:190630,:])
print(train_features['Heartrate'].loc[190602:190630])

# Informatons on the headers -- Extracting information:
patient_characteristics = ["pid", "Age"] # TIME VARIABLE IS EXCLUDED
vital_signs = ['Time', "Heartrate", "SpO2", "ABPs", "ABPm", "ABPd", "RRate", 'Temp']
tests = ['EtCO2', 'PTT', 'BUN', 'Lactate', 'Hgb', 'HCO3', 'BaseExcess',
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
headers_train = test_features.columns
headers_test = test_features.columns
N_patients_train = np.array(train_features.shape[0]/12).astype(int)
N_patients_test = np.array(test_features.shape[0]/12).astype(int)

fig = plt.figure()

#plt.subplot(6,6, ii) 
train_features = train_features_all
ii = 1
for VS in vital_signs:
       plt.subplot(4,4,ii)
       plt.hist(np.array(train_features[VS].iloc[:]))
       plt.title(VS)
       ii = ii + 1

print(train_features['WBC'].iloc[:].describe())
print(train_features['FiO2'].iloc[:].describe())
print(train_features['TroponinI'].iloc[:].describe())


plt.show()