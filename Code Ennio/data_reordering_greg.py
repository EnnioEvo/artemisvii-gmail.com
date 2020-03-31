import pandas as pd
import numpy as np
import math
#from sklearn import svm
#from sklearn import datasets


# Commento: Attualmente i dati di ogni paziente di tutte le dodici ore sono posti su solo una riga. Sono anche ripetuti i valori ridondanti di 
# "pid" e "age", che si possono togliere semplicemente cambiando il range di "ind2" nell'inner "for loop"

# Data import from folder

data_set_x_train = np.array(pd.read_csv("../data/data_train_clean_entire_dataset.csv"))
data_set_y_train = np.array(pd.read_csv("../data/train_labels.csv"))
data_set_x_test = np.array(pd.read_csv("../data/data_test_clean_entire_dataset.csv"))



#Concatenation of all the data for the 12 hours in data_set_x_train to obtain one single raw for each patient
N = data_set_x_train.shape[0] #227940
X_train=np.zeros((math.ceil(N/12),36+34*11))

# put "227940" in the range function to stop the iteration of the inner "for cycle" exactly at the last row of tha last patient
for ind1 in range(0, N, 12):
    print(ind1)
    X_temp = np.array([data_set_x_train[ind1][:2]])
    for ind2 in range(ind1,ind1+12):
        X_temp=np.array([np.concatenate((X_temp, np.array([data_set_x_train[ind2][2:]])), axis=None)])
    X_train[int(ind1/12), :] = X_temp

#X_train=X_train[1:, :]
print(X_train.shape)


#Concatenation of all the data for the 12 hours in data_set_x_test to obtain one single raw for each patient

X_test=np.zeros((1,36+34*11))

# put "15960" in the range function to stop the iteration of the inner "for cycle" exactly at the last row of tha last patient
for ind1 in range(0, 15960, 12):
    X_temp = np.array([data_set_x_train[ind1][0:2]])
    for ind2 in range(ind1,ind1+12):
        X_temp=np.array([np.concatenate((X_temp, np.array([data_set_x_test[ind2][2:]])), axis=None)])
    X_test=np.row_stack((X_test,X_temp))

X_test=X_test[1:, :]
print(X_test.shape)

#example of just one label
Y_train = np.array([data_set_y_train[:, 1]]).T
print(Y_train)
print(Y_train.shape)

# these labels are useful for saving back arrays to dataframes
labels = ['EtCO2', 'PTT', 'BUN', 'Lactate', 'Temp', 'Hgb', 'HCO3', 'BaseExcess',
       "RRate", 'Fibrinogen', 'Phosphate', 'WBC', 'Creatinine', 'PaCO2', 'AST', 'FiO2',
       'Platelets', 'SaO2', 'Glucose', "ABPm", 'Magnesium', 'Potassium', "ABPd", 'Calcium',
       'Alkalinephos', "SpO2", 'Bilirubin_direct', 'Chloride', 'Hct',
       'Heartrate', 'Bilirubin_total', 'TroponinI', "ABPs", 'pH']

all_labels = ['pid', 'Age'] + sum([ [label + str(i+1) for label in labels] for i in range(12)],[])

df = pd.DataFrame(X_test, columns=all_labels)
df.to_csv('../data/data_test_clean_columned_dataset.csv', header=True, index=False)

