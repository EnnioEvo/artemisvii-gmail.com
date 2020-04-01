import pandas as pd
import numpy as np
import math
from sklearn import svm

# Data import from folder. Remember: train 'train_features_clean_all' has 36 columns since the column of the hours is not included 
data_set_x_train = np.array(pd.read_csv("../data/train_features_clean_all.csv"))
data_set_y_train = np.array(pd.read_csv("../data/train_labels.csv"))
data_set_x_test = np.array(pd.read_csv("../data/test_features_clean_all.csv"))

#print(data_set_x_test.shape)
#print(data_set_x_train.shape)
#print(data_set_y_train.shape)

#Concatenation of all the data for the 12 hours in data_set_x_train to obtain one single raw for each patient
N = 227940
X_train=np.zeros((math.ceil(N/12),36*12))
X_temp=[]

# put "227940" in the range function to stop the iteration of the inner "for cycle" exactly at the last row of tha last patient
for ind1 in range(0, N, 12):
#    print(ind1)
    for ind2 in range(ind1,ind1+12):
        X_temp=np.array([np.concatenate((X_temp, np.array([data_set_x_train[ind2][:]])), axis=None)])
    X_train[int(ind1/12), :] = X_temp
    X_temp=[]

print(X_train.shape)


#Concatenation of all the data for the 12 hours in data_set_x_test to obtain one single raw for each patient
N = 151968
X_test=np.zeros((math.ceil(N/12),36*12))
X_temp=[]

# put "151968" in the range function to stop the iteration of the inner "for cycle" exactly at the last row of tha last patient
for ind1 in range(0, N , 12):
    for ind2 in range(ind1,ind1+12):
        X_temp=np.array([np.concatenate((X_temp, np.array([data_set_x_test[ind2][:]])), axis=None)])
    X_test[int(ind1/12), :] = X_temp
    X_temp=[]

print(X_test.shape)

X_tn=X_train
X_tt=X_test

print()
print()
print()
print('##########          ##########')
print('##########          ##########')
print('##########          ##########')
print()
print()
print()

#print(data_set_x_train[:,0])
#print(data_set_x_train)

#X_train = data_set_x_train[0:1000,:]
#print(X_train.shape)
#X_test= data_set_x_test[0:1000,:]

#define number of patient in train and test set which are later taken into account:

Ntrain=500
Ntest=500

X_train = X_tn[0:Ntrain,:]
X_test= X_tt[0:Ntest,:]

from time import time
start = time()


#Subtask 1
Y_test_1=np.zeros((Ntest,11))
Y_test_1[0:Ntest,0]=X_test[0:Ntest,0]
Y_temp_1=np.zeros((Ntest,2))
print(Y_test_1.shape)
for ind1 in range(1,11):
    Y_train = np.array([data_set_y_train[0:Ntrain, ind1]]).T
    print(Y_train.shape)
    clf = svm.SVC(gamma='scale',probability=True,class_weight='balanced')
    #clf = svm.SVC(probability=True,class_weight='balanced')
    clf.fit(X_train, Y_train)
    Y_temp_1[:,:]=np.array([clf.predict_proba(X_test)])
    Y_test_1[:,ind1]=Y_temp_1[:,1]
    Y_temp_1=np.zeros((Ntest,2))
#print(Y_train)
#print(Y_train.shape)
print(Y_test_1)
print('timer:', (time()-start)/Ntrain * 1000)


#Subtask 2
Y_test_2=np.zeros((Ntest,2))
Y_test_2[0:Ntest,0]=X_test[0:Ntest,0]
Y_temp_2=np.zeros((Ntest,2))
print(Y_test_2.shape)
Y_train = np.array([data_set_y_train[0:Ntrain, 11]]).T
print(Y_train.shape)
clf = svm.SVC(gamma='scale',probability=True,class_weight='balanced')
clf.fit(X_train, Y_train)
Y_temp_2[:,:]=np.array([clf.predict_proba(X_test)])
Y_test_2[:,1]=Y_temp_2[:,1]
#print(Y_train)
#print(Y_train.shape)
print(Y_test_2)

#Subtask 3
Y_test_3=np.zeros((Ntest,5))
Y_test_2[0:Ntest,0]=X_test[0:Ntest,0]
print(Y_test_3.shape)
for ind2 in range(1,5):
    Y_train = np.array([data_set_y_train[0:Ntrain, ind2+11]]).T
    print(Y_train.shape)
    clf = svm.SVR(gamma='scale')
    clf.fit(X_train, Y_train)
    Y_test_3[0:Ntest,ind2]=np.array([clf.predict(X_test)])
#print(Y_train)
#print(Y_train.shape)
print(Y_test_3)

print()
print()
print()


print(Y_test_1.shape)
print(Y_test_2.shape)
print(Y_test_3.shape)

Y_test=np.column_stack((Y_test_1, Y_test_2[:,1:], Y_test_3[:,1:])) 

# let's add labels
labels = ['pid','LABEL_BaseExcess','LABEL_Fibrinogen','LABEL_AST','LABEL_Alkalinephos','LABEL_Bilirubin_total','LABEL_Lactate','LABEL_TroponinI','LABEL_SaO2','LABEL_Bilirubin_direct','LABEL_EtCO2','LABEL_Sepsis','LABEL_RRate','LABEL_ABPm','LABEL_SpO2','LABEL_Heartrate']
df = pd.DataFrame(Y_test, columns=labels)

# suppose df is a pandas dataframe containing the result
#df.to_csv('prediction.csv', index=False, float_format='%.3f')

# suppose df is a pandas dataframe containing the result
df.to_csv('prediction.csv', index=False, float_format='%.3f')