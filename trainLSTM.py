
from imutils import paths
import random
import os
import numpy as np
from model import MiniVGG
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
import pickle
import matplotlib.pyplot as plt


import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from keras.utils import np_utils
import pickle
from sklearn.model_selection import train_test_split


data=pd.read_csv('balanced2.csv')
print(data.head(10))
# data = data.drop(data.columns[[5]], axis=1)
# print(data.head(10))
tlabel=data['label'].values
print(tlabel)
tdata=data.drop(['label'],axis=1)

X_train, X_test, y_train, y_test = train_test_split(tdata, tlabel, test_size=0.2, random_state=42)

print(X_train)
print(y_train)
# print(X_train.shape)
# labels=y_train.to_numpy()
# print(data.label.unique())
# labels = np_utils.to_categorical(labels)
# features=X_train
# X_train=X_train.to_numpy()
# # X_test=X_test.to_numpy()
# print(X_train.shape)
# print(X_test.shape)
# # x_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
# # print(x_train.shape)
# # x_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
# # print(x_test.shape)

from sklearn.preprocessing import MinMaxScaler

lstmscaler = MinMaxScaler().fit(X_train)
pickle.dump(lstmscaler,open('minmaxlstm.pkl','wb'))

X_train=lstmscaler.transform(X_train)
X_test=lstmscaler.transform(X_test)
X_train_series = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_valid_series = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
print('Train set shape', X_train_series.shape)
print('Validation set shape', X_valid_series.shape)

# from sklearn.preprocessing import MinMaxScaler

# norm = MinMaxScaler().fit(X_train)
# pickle.dump(norm,open('norm.pkl','wb'))

# features=norm.transform(features)
# print(features.shape)
# features=np.reshape(features,(len(features),20,2,1))


# print("***************")
# print(features.shape)
# print(labels)
# print(labels.shape)


# # # test['label']=newlabeldf_test
# # print(y_test.unique())
# t_labels=y_test.to_numpy()
# t_labels = np_utils.to_categorical(t_labels)
# t_features=X_test.to_numpy()
# t_features=norm.transform(t_features)
# t_features=np.reshape(t_features,(len(t_features),20,2,1))

# print(t_features.shape)
# print(t_labels.shape)
# y_train=y_train.to_numpy()
# y_test=y_test.to_numpy()
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
from sklearn.metrics import accuracy_score
import json
from tensorflow.keras.models import model_from_json


model_lstm = Sequential()
model_lstm.add(LSTM(50, activation='relu', input_shape=(X_train_series.shape[1], X_train_series.shape[2])))
model_lstm.add(Dense(1))
model_lstm.compile(loss='mean_squared_error', optimizer='adam')
model_lstm.summary()
lstm_history = model_lstm.fit(X_train_series, y_train, validation_data=(X_valid_series, y_test), epochs=20, verbose=1)
# serialize model to JSON
model_json = model_lstm.to_json()
with open("model_lstm1.json", "w") as json_file:
        json_file.write(model_json)
# serialize weights to HDF5
model_lstm.save_weights("lstm_weight1.h5")
print("[INFO] Saved model to disk")
ypred=model_lstm.predict(X_valid_series)
print(ypred)
# for i in ypred:
#     result=np.argmax(i)
#     print(result)
# acc=accuracy_score(y_test, ypred)
# print("score==>",acc)
# # create and fit the LSTM network
# look_back = 15
# model = Sequential()
# model.add(LSTM(20, input_shape=(X_train_series.shape[1], X_train_series.shape[2])))
# model.add(Dense(1))
# model.compile(loss='mean_squared_error', optimizer='adam')
# model.fit(X_train_series, y_train, epochs=20, batch_size=1, verbose=1)
