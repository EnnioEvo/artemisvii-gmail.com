import pandas as pd
import numpy as np
import math
from sklearn import svm
from sklearn import linear_model
import sklearn.metrics as skmetrics

from sklearn.preprocessing import StandardScaler

# Data import from folder. Remember: train 'train_features_clean_all' has 36 columns since the column of the hours is not included 
data_set_x_train_in = np.array(pd.read_csv("../data/train_features_clean_all.csv"))
data_set_y_train_fake = np.array(pd.read_csv("../data/train_labels.csv"))


#Concatenation of all the data for the 12 hours in data_set_x_train to obtain one single raw for each patient
N = 227940
X_train_in=np.zeros((math.ceil(N/12),36*12))
X_temp=[]

# put "227940" in the range function to stop the iteration of the inner "for cycle" exactly at the last row of tha last patient
for ind1 in range(0, N, 12):
#    print(ind1)
    for ind2 in range(ind1,ind1+12):
        X_temp=np.array([np.concatenate((X_temp, np.array([data_set_x_train_in[ind2][:]])), axis=None)])
    X_train_in[int(ind1/12), :] = X_temp
    X_temp=[]

print(X_train_in.shape)


data_set_x_train=X_train_in[0:15000,:]
data_set_x_test=X_train_in[15000:,:]
data_set_y_train=data_set_y_train_fake[0:15000,:]
data_set_y_test=data_set_y_train_fake[15000:,:]


print()
print()
print()
print('##########          ##########')
print('##########          ##########')
print('##########          ##########')
print()
print()
print()


#define number of patient in train and test set which are later taken into account:

Ntrain=15000
Ntest=18995-15000

X_train = data_set_x_train
X_test = data_set_x_test

#from time import time
#start = time()


#Subtask 1
Y_test_1=np.zeros((Ntest,11))
Y_test_1[0:Ntest,0]=X_test[0:Ntest,0]
Y_temp_1=np.zeros((Ntest,1))
Y_temp_2=np.zeros((Ntest,1))
print(Y_test_1.shape)
for ind1 in range(1,11):
    Y_train = np.array([data_set_y_train[0:Ntrain, ind1]]).T
    print(Y_train.shape)

    clf = svm.LinearSVC(C=0.001, loss='hinge', class_weight='balanced',max_iter=10000)
    clf.fit(X_train, Y_train)
    Y_temp_1[:,0]=np.array([clf.decision_function(X_test)])

    #clf = linear_model.SGDClassifier(class_weight='balanced')
    #clf.fit(X_train, Y_train)
    #Y_temp_1[:,0]=np.array([clf.decision_function(X_test)])

    Y_temp_2=1/(1+np.exp(-Y_temp_1))
    Y_test_1[:,ind1]=Y_temp_2[:,0]
    Y_temp_1=np.zeros((Ntest,1))
    Y_temp_1=np.zeros((Ntest,1))
#print(Y_train)
#print(Y_train.shape)
print(Y_test_1)
#print('timer:', (time()-start)/Ntrain * 1000)


#Subtask 2
Y_test_2=np.zeros((Ntest,2))
Y_test_2[0:Ntest,0]=X_test[0:Ntest,0]
Y_temp_3=np.zeros((Ntest,1))
Y_temp_4=np.zeros((Ntest,1))
print(Y_test_2.shape)
Y_train = np.array([data_set_y_train[0:Ntrain, 11]]).T
print(Y_train.shape)
clf = svm.LinearSVC(C=0.001, loss='hinge', class_weight='balanced',max_iter=1000)
clf.fit(X_train, Y_train)
Y_temp_3[:,0]=np.array([clf.decision_function(X_test)])
Y_temp_4=1/(1+np.exp(-Y_temp_3))
Y_test_2[:,1]=Y_temp_4[:,0]
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
    clf = svm.LinearSVR(C=0.001,max_iter=1000)
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
df.to_csv('prediction.csv', index=False, float_format='%.3f')


# Now the final score is computed

task1 = 0
for ii in range(1,11):
       task1 = task1 + 1/10*(skmetrics.roc_auc_score(data_set_y_test[:,ii],Y_test[:,ii]))
print("AOC score -- task 1: ", task1)

print()
print()

task2 = skmetrics.roc_auc_score(data_set_y_test[:,11], Y_test_2[:,1])

print("AOC score -- task 2: ", task2)

print()
print()

task3 = np.zeros(4)
for ii in range(4):
       task3[ii] = 0.5 + 0.5 * np.maximum(0, skmetrics.r2_score(data_set_y_test[:,12+ii], Y_test[:,12+ii]))
task3_mean = np.mean(task3)
print("R2 score -- task 3: ", task3)

print()
print()

print("-------------------------------------")
score = np.mean([task1, task2, task3_mean])
print("TOTAL SCORE: ", score)