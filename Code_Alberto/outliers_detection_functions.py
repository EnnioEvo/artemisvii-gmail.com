import numpy as np
import pandas as pd

def _get_percentiles(data_set, min, max):
    perc = pd.DataFrame(columns=data_set.columns)
    perc.loc[0,:] = np.nanpercentile(np.array(data_set), min,
                                     axis=0, interpolation='lower')
    perc.loc[1,:] = np.nanpercentile(np.array(data_set), max,
                                     axis=0, interpolation='higher')
    return perc

percentiles = _get_percentiles(train_features, 0, 99.95)  

def clean_data_set(data_set, labels, percentiles, tests, others, h_obs):
    N_patients = np.array(data_set.shape[0]/12).astype(int)
    print(N_patients)
    non_outliers = np.zeros((1,data_set[tests].shape[1]))
    non_outliers_others = np.zeros((1, len(others)))
    outliers = np.zeros((1,data_set[tests].shape[1]))
    outliers_others = np.zeros((1, len(others)))
    labels_out = np.zeros((1, labels.shape[1]))
    labels_no_out = np.zeros((1, labels.shape[1]))
    # Generate percentiles matrices:
    perc_min = np.repeat(np.array(percentiles[tests].loc[0]), h_obs,
                         axis=0).reshape((len(tests),h_obs)).T
    perc_max = np.repeat(np.array(percentiles[tests].loc[1]), h_obs,
                         axis=0).reshape((len(tests),h_obs)).T
    patient_tot_matr = np.array(data_set[tests].iloc[:])
    patient_tot_others = np.array(data_set[others].loc[:])
    labels_np = np.array(labels)
    index = 0
    for patient in range(N_patients):
        patient_matr = patient_tot_matr[patient*12:patient*12+12, :]
        patient_matr_others = patient_tot_others[patient*12:patient*12+12, :]
        bool_min = perc_min >= patient_matr
        bool_max = patient_matr >= perc_max
        if np.sum(np.logical_or(bool_min, bool_max)):
            outliers = np.concatenate((outliers, patient_matr), axis =0)
            outliers_others = np.concatenate((outliers_others, patient_matr_others), axis=0)
            labels_out = np.append(labels_out, np.array(labels_np[index,:]).reshape(1,16), axis=0)
        else:
            non_outliers = np.concatenate((non_outliers, patient_matr), axis =0)
            non_outliers_others = np.concatenate((non_outliers_others, patient_matr_others), axis=0)
            labels_no_out = np.append(labels_no_out, np.array(labels_np[index,:]).reshape(1,16), axis=0)
        if patient%1000 == 0: print(patient)
        index = index + 1

    data_set_new_outliers = np.column_stack((outliers_others,outliers)) 
    data_set_new_non_outliers = np.column_stack((non_outliers_others,non_outliers))
    outliers = pd.DataFrame(data=data_set_new_outliers[1:,:], columns=data_set.columns)
    non_outliers = pd.DataFrame(data=data_set_new_non_outliers[1:,:], columns=data_set.columns)
    labels_no_out = pd.DataFrame(data=labels_no_out[1:,:], columns=labels.columns)
    labels_out = pd.DataFrame(data=labels_out[1:,:], columns=labels.columns)
    return outliers, non_outliers, labels_out, labels_no_out

others = patient_characteristics + vital_signs
outliers, non_outliers, labels_out, labels_no_out = clean_data_set(train_features, train_labels, percentiles, tests, others, h_obs=12)
