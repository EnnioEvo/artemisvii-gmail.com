import pandas as pd
import numpy as np

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

#import data
train_raw = pd.read_csv("../data/train_features.csv", index_col=0)
train_clean_all = pd.read_csv("../data/train_features_clean_all.csv", index_col=0)
train_clean_mean = pd.read_csv("../data/train_features_clean_mean.csv", index_col=0)
train_clean_wmean = pd.read_csv("../data/train_features_clean_wmean.csv", index_col=0)
train_clean_wmean_2 = pd.read_csv("../data/train_features_clean_wmean_2.csv", index_col=0)
stored_usefulness_matrix_t1 = pd.read_csv("../data/feature_selection/usefulness_matrix_t1_sum.csv", index_col=0)
stored_usefulness_matrix_t3 = pd.read_csv("../data/feature_selection/usefulness_matrix_t3_sum.csv", index_col=0)
test_features_raw = pd.read_csv("../data/test_features.csv")
test_features_mean = pd.read_csv("../data/test_features_clean_wmean.csv")
train_labels = pd.read_csv("../data/train_labels.csv")
kernel_selection = pd.read_csv("../data/kernel_selection.csv").sort_values(by=['LABEL','score'],ascending=False)
#build train mean raw


def build_kernel_selector():
    best_kernels = pd.DataFrame(index=labels_tests + labels_sepsis, columns=['kernel', 'C'])
    for label in labels_tests + labels_sepsis:
        kernel_classific = kernel_selection[kernel_selection['LABEL']==label].sort_values(by=['score'],ascending=False)
        best_kernels.at[label,['kernel', 'C']] = kernel_classific[['kernel', 'C']].iloc[0]
    return best_kernels

best_kernels = build_kernel_selector()
best_kernels.to_csv('../data/best_kernels.csv', header=True, index=True)
print()


def data_analysis():
    usefulness_column = stored_usefulness_matrix_t1[labels_sepsis].sort_values(labels_sepsis, ascending = [0])
    useful_features_mask = np.array(usefulness_column) >= 4
    useful_features = [feature for feature,mask in zip(usefulness_column.index, useful_features_mask) if mask]
    useful_features_augmented = sum([ [feature, 'dummy_' + feature] for feature in useful_features if feature in tests], []) + \
        [feature for feature in useful_features if feature in vital_signs]
    #train_raw = train_raw[useful_features]
    #train_clean_all = train_clean_all[useful_features_augmented]
    #train_clean_mean = train_clean_mean[useful_features_augmented]
    train_dummy = train_clean_wmean[ [feature for feature in useful_features_augmented if feature in dummy_tests] ]
    relevance_train = np.dot(train_dummy,np.array(usefulness_column)[:10])
    relevance_train_frame = pd.DataFrame(index= train_clean_wmean.index, columns = ['Relevance'], data=relevance_train)
    print()
    pass

def build_usefulness_sum():
    usefulness_matrixes_t1 = []
    usefulness_matrixes_t3 = []

    usefulness_matrix_t1_sum = 0
    usefulness_matrix_t3_sum = 0

    for i in range(10):
        usefulness_matrixes_t1 = usefulness_matrixes_t1 +\
                             [pd.read_csv("../data/feature_selection/usefulness_matrix_t1_dummy_"+ str(i) + ".csv", index_col=0)]
        usefulness_matrix_t1_sum = usefulness_matrix_t1_sum + (usefulness_matrixes_t1[i] == 0)*-1 + (
                usefulness_matrixes_t1[i] == 1)*1

        usefulness_matrixes_t3 = usefulness_matrixes_t3 +\
                             [pd.read_csv("../data/feature_selection/usefulness_matrix_t3_dummy_"+ str(i) + ".csv", index_col=0)]
        usefulness_matrix_t3_sum = usefulness_matrix_t3_sum + (usefulness_matrixes_t3[i] == 0) * -1 + (
                    usefulness_matrixes_t3[i] == 1) * 1

    usefulness_matrix_t1_sum.to_csv('../data/feature_selection/usefulness_matrix_t1_sum' + '.csv', header=True, index=True)
    usefulness_matrix_t3_sum.to_csv('../data/feature_selection/usefulness_matrix_t3_sum' + '.csv', header=True, index=True)


