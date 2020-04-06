import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn import svm
from sklearn.linear_model import LinearRegression
import sklearn.metrics as skmetrics
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.utils import to_categorical
from keras import regularizers

train_features = pd.read_csv('data/train_features_clean_all_no_norm.csv')
train_labels = pd.read_csv('data/train_labels.csv')
train_data = 15000

real_solution = train_labels.loc[train_data:,:]
train_labels = train_labels.loc[0:train_data,:]

test_features = pd.DataFrame(data=np.array(train_features.iloc[train_data*12:, :]), columns=train_features.columns)
train_features = train_features.iloc[0:train_data*12, :]

# Informatons on the headers:
patient_characteristics = ['pid', 'Age'] 
time_car = ['Time']
vital_signs = ["Heartrate", "SpO2", "ABPs", "ABPm", "ABPd", "RRate", 'Temp']
tests = ['EtCO2', 'PTT', 'BUN', 'Lactate', 'Hgb', 'HCO3', 'BaseExcess',
       'Fibrinogen', 'Phosphate', 'WBC', 'Creatinine', 'PaCO2', 'AST', 'FiO2',
       'Platelets', 'SaO2', 'Glucose', 'Magnesium', 'Potassium', 'Calcium',
       'Alkalinephos', 'Bilirubin_direct', 'Chloride', 'Hct',
       'Bilirubin_total', 'TroponinI', 'pH']
labels_tests = ['LABEL_BaseExcess', 'LABEL_Fibrinogen', 'LABEL_AST',
       'LABEL_Alkalinephos', 'LABEL_Bilirubin_total', 'LABEL_Lactate',
       'LABEL_TroponinI', 'LABEL_SaO2', 'LABEL_Bilirubin_direct',
       'LABEL_EtCO2' ]
labels_sepsis = ['LABEL_Sepsis']
labels_VS_mean = ['LABEL_RRate', 'LABEL_ABPm', 'LABEL_SpO2', 'LABEL_Heartrate']

h_obs = 12
N_patients_train = np.array(train_features.shape[0]/h_obs).astype(int)
N_patients_test = np.array(test_features.shape[0]/h_obs).astype(int)

# Scaling data: ######### CHABGE POSITION
feature_selected =  vital_signs + tests
X = np.array(train_features[feature_selected].iloc[:])
scaler = preprocessing.StandardScaler().fit(X)
X = scaler.transform(X)
X_test = scaler.transform(np.array(test_features[feature_selected].iloc[:]))


# Flatten hourly features
X = np.concatenate((np.array(train_features[['Age']].loc[range(0,N_patients_train*12,12)]), X.reshape((N_patients_train, h_obs*len(feature_selected)))), axis=1)

X_test = np.concatenate(( np.array(test_features[['Age']].loc[range(0,N_patients_test*12,12)]), X_test.reshape((N_patients_test, h_obs*len(feature_selected)))), axis=1)


Y_t1_test = real_solution
Y_t1 = train_labels

def sigmfun(x):
       return 1/(1+np.exp(-x))

labels_prediction = ['pid'] +labels_tests + labels_sepsis + labels_VS_mean
prediction = np.zeros((N_patients_test, len(labels_prediction)))
prediction[:,0] = np.array(test_features['pid'].iloc[:]).reshape((N_patients_test, h_obs))[:,0]
prediction_ind = 1


###################### TASK 1 ##########################################################################
for test in labels_tests:
       print('Test considered : ', test)
       Y_t1 = np.array(train_labels[test].iloc[:])[0:train_data]
       #model_1 = SVC(C=0.0001, kernel='rbf', degree=1, class_weight='balanced', verbose=1)
       model_1 = svm.LinearSVC(C=0.0001, class_weight='balanced',max_iter=1000)
       model_1.fit(X, Y_t1)
       prediction[:,prediction_ind] = sigmfun(model_1.decision_function(X_test))
       prediction_ind = prediction_ind + 1


###################### TASK 2 ##########################################################################
Y_t2 = np.array(train_labels[labels_sepsis].iloc[:])[0:train_data]
print('Testing ', labels_sepsis)
#model_2 = SVC(C=10, kernel='rbf', class_weight='balanced')
#model_2 = svm.LinearSVC(C=0.0001, class_weight='balanced',max_iter=10000)
#model_2.fit(X, np.ravel(Y_t2))
#prediction[:,prediction_ind] = np.ravel(sigmfun(model_2.decision_function(X_test)))

model_2 =Sequential()
model_2.add(Dense(units = 500, input_dim= X.shape[1]))
model_2.add(Dropout(0.5))
model_2.add(BatchNormalization())
model_2.add(Dense(100, activation = "tanh")) #tanh
model_2.add(Dropout(0.5))
model_2.add(BatchNormalization())
model_2.add(Dense(2, activation = "softmax"))
model_2.compile(optimizer=keras.optimizers.Adadelta(),
              loss=keras.losses.binary_crossentropy,
              metrics=[keras.metrics.categorical_accuracy])

model_2.fit(X, to_categorical(Y_t2), epochs=10)
prediction[:,prediction_ind] = np.ravel(model_2.predict(X_test)[:,1])
prediction_ind = prediction_ind + 1


###################### TASK 3 ##########################################################################
model_3 =Sequential()
model_3.add(Dense(units = 500, input_dim= X.shape[1]))
model_3.add(Dropout(0.5))
model_3.add(BatchNormalization())
model_3.add(Dense(100, activation = "relu",)) #relu
model_3.add(Dropout(0.5))
model_3.add(BatchNormalization())
model_3.add(Dense(1, activation = "linear"))
model_3.compile(optimizer=keras.optimizers.Adadelta(),
              loss=keras.losses.mean_squared_error)

for VS in labels_VS_mean:
       print('Testing ', VS)
       Y_t3 = np.array(train_labels[VS].iloc[:])[0:train_data]
       Y_t3_mean = np.mean(Y_t3)
       Y_t3_std = np.std(Y_t3)
       Y_t3 = (Y_t3-Y_t3_mean)/Y_t3_std
       #model_3 = LinearRegression()
       model_3.fit(X, Y_t3, epochs=10)
       Y_predictions_3 = model_3.predict(X_test)*Y_t3_std + Y_t3_mean
       #prediction[:,prediction_ind] = np.ravel(model_3.predict(X_test))
       prediction[:,prediction_ind] = np.ravel(Y_predictions_3)
       prediction_ind = prediction_ind + 1

feature_selected =  vital_signs + tests
X = np.array(train_features[feature_selected].iloc[:])
scaler = preprocessing.StandardScaler().fit(X)
X = scaler.transform(X)
Y = scaler.inverse_transform(X, copy=None)
X_test = scaler.transform(np.array(test_features[feature_selected].iloc[:]))


###########################  SAVING #####################
df_submission = pd.DataFrame(data=prediction, columns=labels_prediction)


############################################# RESULTS ##################################################
print()
print()
print('####################               RESULTS              ####################')
VITALS = ['LABEL_RRate', 'LABEL_ABPm', 'LABEL_SpO2', 'LABEL_Heartrate']
TESTS = ['LABEL_BaseExcess', 'LABEL_Fibrinogen', 'LABEL_AST', 'LABEL_Alkalinephos', 'LABEL_Bilirubin_total',
         'LABEL_Lactate', 'LABEL_TroponinI', 'LABEL_SaO2',
         'LABEL_Bilirubin_direct', 'LABEL_EtCO2']

def get_score(df_true, df_submission):
    df_submission = df_submission.sort_values('pid')
    df_true = df_true.sort_values('pid')
    task1 = np.mean([skmetrics.roc_auc_score(df_true[entry], df_submission[entry]) for entry in TESTS])
    task2 = skmetrics.roc_auc_score(df_true['LABEL_Sepsis'], df_submission['LABEL_Sepsis'])
    print('task3 individual scores:')
    print([0.5 + 0.5 * np.maximum(0, skmetrics.r2_score(df_true[entry], df_submission[entry])) for entry in VITALS])
    task3 = np.mean([0.5 + 0.5 * np.maximum(0, skmetrics.r2_score(df_true[entry], df_submission[entry])) for entry in VITALS])
    score = np.mean([task1, task2, task3])
    print(task1, task2, task3)
    return score

df_true = real_solution

#for label in TESTS + ['LABEL_Sepsis']:
#    # round classification labels
#    df_true[label] = np.around(df_true[label].values)
print()
print('Score: ', get_score(df_true, df_submission))
