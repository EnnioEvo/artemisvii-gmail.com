import pandas as pd
import numpy as np
from utilities.kernels import LinearKernel, PolynomialKernel, GaussianKernel, PeriodicKernel, LaplacianKernel, SumKernel

#from sklearn import svm
#from sklearn import datasets


# Commento: Attualmente i dati di ogni paziente di tutte le dodici ore sono posti su solo una riga. Sono anche ripetuti i valori ridondanti di 
# "pid" e "age", che si possono togliere semplicemente cambiando il range di "ind2" nell'inner "for loop"

# Data import from folder
data_set_x_train = np.array(pd.read_csv("../data/train_features.csv"))
data_set_y_train = np.array(pd.read_csv("../data/train_labels.csv"))
data_set_x_test = np.array(pd.read_csv("../data/test_features.csv"))

data_set_x_train=np.nan_to_num(data_set_x_train)
data_set_x_test=np.nan_to_num(data_set_x_test)
#print(data_set_x_test[1][:])


#Concatenation of all the data for the 12 hours in data_set_x_train to obtain one single raw for each patient
X_train=np.zeros((1,37*12))
X_temp=[]

# put "227940" in the range function to stop the iteration of the inner "for cycle" exactly at the last row of tha last patient
for ind1 in range(0, 227940, 12):

    for ind2 in range(ind1,ind1+12):
        X_temp=np.array([np.concatenate((X_temp, np.array([data_set_x_train[ind2][:]])), axis=None)])

    X_train=np.row_stack((X_train,X_temp))
    X_temp=[]

X_train=X_train[1:, :]
print(X_train.shape)


#Concatenation of all the data for the 12 hours in data_set_x_test to obtain one single raw for each patient
X_test=np.zeros((1,37*12))
X_temp=[]

# put "15960" in the range function to stop the iteration of the inner "for cycle" exactly at the last row of tha last patient
for ind1 in range(0, 15960 , 12):

    for ind2 in range(ind1,ind1+12):
        X_temp=np.array([np.concatenate((X_temp, np.array([data_set_x_test[ind2][:]])), axis=None)])

    X_test=np.row_stack((X_test,X_temp))
    X_temp=[]

X_test=X_test[1:, :]
print(X_test.shape)

#example of just one label
Y_train = np.array([data_set_y_train[:, 1]]).T
print(Y_train)
print(Y_train.shape)



# Parte mancante ancora da modificare

#bw=0.2
#regressor = GaussianKernel(X_train, Y_train, reg=0.0, bw=bw)
#regressor = PolynomialKernel(X_train, Y_train, deg=3, reg=0.0)
#regressor.calculate_alpha(Y_train)
#ypr = regressor.predict(X_test)
#ycl=1/(1+np.exp(-ypr))
#print(ycl)