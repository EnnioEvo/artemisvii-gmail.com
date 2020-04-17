##################  -- GENERATE SAMPLE WEIGHTS -- ##################
import numpy as np
import pandas as pd

print("#############  -- GENERATE SAMPLE WEIGHTS - data ALL -- #############")
# Data import:
train_features = pd.read_csv("../data/train_features.csv")

X = np.array(train_features)
nanmatrix = np.isnan(X)

M = train_features.shape[0]  # 227940
N = train_features.shape[1]  # 37

nanmatrix_reduced = np.zeros([int(M / 12), N])
for j in range(3,N):
    for i in range(int(M / 12)):
        boolval = nanmatrix[12*i:12*(i+1), j]
        entry = np.all(boolval)
        nanmatrix_reduced[i, j] = entry

nans_per_column = sum(nanmatrix_reduced)  # shape 1xN
sample_weigths = np.dot(nanmatrix_reduced, nans_per_column)  # MxN * Nx1 = Mx1
pd.DataFrame(sample_weigths).to_csv('../data/sample_weights.csv', header=False, index=False)
print(nans_per_column)
