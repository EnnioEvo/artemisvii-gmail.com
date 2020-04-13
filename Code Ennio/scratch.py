import pandas as pd
import numpy as np
sample = pd.read_csv("../sample.csv")
sample.set_index('pid')

Y_test = pd.read_csv("../data/submission.csv", index_col=0)
Y_test.insert(0, 'pid', sample['pid'])
Y_test.to_csv("../data/submission.zip", header=True, index=False, float_format='%.5f', compression='zip')

X = pd.read_csv("../data/submission.csv", index_col=0)