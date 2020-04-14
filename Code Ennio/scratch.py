import pandas as pd
import numpy as np
sample = pd.read_csv("../sample.csv")
sample.set_index('pid')


# train_raw = pd.read_csv("../data/train_features.csv", index_col=0)
# train_clean_all = pd.read_csv("../data/train_features_clean_all.csv", index_col=0)
# train_clean_mean = pd.read_csv("../data/train_features_clean_mean.csv", index_col=0)
# train_clean_wmean = pd.read_csv("../data/train_features_clean_wmean.csv", index_col=0)
# train_clean_mean_bug = pd.read_csv("../data/train_features_clean_mean_bug.csv", index_col=0)

pass

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


pass

