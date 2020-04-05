import numpy as np
import pandas as pd
from progress.bar import IncrementalBar

def VS_cleaning(data_set, vitals, h_obs):
    print('-- VS FEATURES --')
    N_patients = np.array(data_set.shape[0]/h_obs).astype(int)
    data_set_new = np.zeros((N_patients*h_obs, len(vitals)))
    ii = 0
    bar1 = IncrementalBar('### VS features processing ### VS analyzed:', max = len(vitals))
    for VS in vitals:
        VS_col_patients = np.array(data_set[VS].loc[:])
        VS_col_new = []
        VS_mean_col = np.nanmean(VS_col_patients)
        for patient in range(N_patients):
            ind_start = patient*h_obs
            ind_finish = h_obs*patient+h_obs
            Y = VS_col_patients[ind_start:ind_finish]
            if np.prod(np.isnan(Y)):
                Y = np.ones(h_obs)*VS_mean_col
                VS_col_new = np.append(VS_col_new, Y)
            else:
                if np.sum(np.isnan(Y)):
                    mean_col = np.nanmean(Y)
                    Y[np.isnan(Y)] = np.ones(np.sum(np.isnan(Y)))*mean_col
                    VS_col_new = np.append(VS_col_new, Y)
                else:
                    VS_col_new = np.append(VS_col_new, Y)
        data_set_new[:,ii] = (VS_col_new - np.mean(VS_col_new))/np.std(VS_col_new)
        ii = ii + 1
        bar1.next()
    bar1.finish()
    return data_set_new

def tests_cleaning(data_set, tests, h_obs, test_mean_in, test_std_in):
    print('-- tests features --')
    N_patients = np.array(data_set.shape[0]/h_obs).astype(int)
    ii = 0 
    if np.logical_not(np.any(test_mean_in)):
        test_mean = np.nanmean(data_set[tests].loc[:], axis = 0)
    else:
        test_mean = test_mean_in
    if np.logical_not(np.any(test_std_in)):
        test_std = np.nanstd(data_set[tests].loc[:], axis = 0)
    else:
        test_std = test_std_in  
    data_set_new = np.zeros(data_set[tests].iloc[:].shape)
    bar2 = IncrementalBar('### TESTS features processing ### VS analyzed:', max = len(tests))
    ii = 0 
    for test in tests:
        test_col = data_set[test].loc[:]
        test_col_new = np.zeros(test_col.shape[0])
        for idx in range(test_col.shape[0]):
            if np.logical_not(np.isnan(test_col[idx])):
                test_col_new[idx] = (test_col[idx] - test_mean[ii])/test_std[ii]
            
        data_set_new[:,ii] = test_col_new
        ii = ii + 1
        bar2.next()
    bar2.finish()
    return data_set_new, test_mean, test_std 
