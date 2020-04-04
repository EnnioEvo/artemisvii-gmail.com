##################  -- CLEANING DATA ALGORITHM - data ALL -- ##################
# LEGEND --> VS= vital signs, tests: medical tests (from data)
#           train: training data, test: testa data to be submitted
#                   !cleaning with normalization!

import numpy as np
import pandas as pd


def VS_clean_aggregation(data_set, N_patients, vital_signs, hours_obs):
    print('First imputation loop + aggregation -- VS features')
    h_vect  = np.array([ii for ii in range(hours_obs)])
    VS_features = data_set.loc[:, vital_signs]
    data_set_new = np.zeros([N_patients, len(vital_signs)])
    VS_mean = np.nanmean(np.array(VS_features), axis=0)
    nan_matrix = VS_features.isna()
    for patient in range(N_patients):
        ii = 0
        for VS in vital_signs:
            if np.sum(np.array(nan_matrix[VS].iloc[patient*hours_obs:(patient+1)*hours_obs])) == hours_obs:
                data_set_new[patient, ii] = np.array(VS_mean[ii])
            else:
                data_set_new[patient, ii] = np.nanmean(np.array(VS_features[VS].iloc[patient*hours_obs:(1+patient)*hours_obs]))
            ii = ii + 1
        if patient%3000==0: print("Patients analyzed in loop: ", patient)
    return data_set_new


def test_clean_aggregation(data_set, N_patients, tests, hours_obs, min_tests, max_tests):
    print('Second imputation loop + aggregation -- tests features')
    ii = 0 
    if np.logical_not(np.any(min_tests)):
        test_min = np.nanmean(data_set[tests].loc[:], axis = 0)
    else:
        test_min = min_tests
    if np.logical_not(np.any(max_tests)):
        test_max = np.nanstd(data_set[tests].loc[:], axis = 0)
    else:
        test_max = max_tests 

    data_set_new = np.zeros([N_patients, len(tests)])

    for test in tests:
        test_col = np.array(data_set[test])
        for idx in range(test_col.shape[0]):
            if np.isnan(test_col[idx]):
                test_col[idx] = 0
            elif test_col[idx]> test_max[ii]: test_col[idx] = 2
            elif test_col[idx]< test_min[ii]: test_col[idx] = 1
            else:
                test_col[idx] = test_max[ii]*4 + (test_col[idx] - test_min[ii])/(test_max[ii])
        temp = np.reshape(test_col, (hours_obs, N_patients))
        for jj in range(N_patients):
            A = temp[:,jj]
            if np.sum(np.array(A != 0)) == 0:
                data_set_new[jj, ii] = 0
            else:
                data_set_new[jj, ii] = np.mean(A[A != 0])
        ii = ii + 1
        if ii%5==0: print("Tests analyzed in loop: ", ii)
    return data_set_new, test_max, test_min
    