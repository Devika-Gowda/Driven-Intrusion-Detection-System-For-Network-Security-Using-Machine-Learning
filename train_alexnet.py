
from pyexpat import model
import numpy as np
from keras import layers
from keras.models import Sequential
from keras.layers import Input, Dense, Activation,BatchNormalization, Flatten, Conv2D, MaxPooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.layers.core import Dropout

from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
def AlexNet():
    inputShape = (20,2, 1)
    chanDim = -1
    


    model = Sequential()
    model.add(Conv2D(32, (3,3), input_shape = inputShape ,padding='same'))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size = (2,2), strides=(2,2),padding='same'))

    model.add(Conv2D(32, (3,3), input_shape = inputShape ,padding='same'))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size = (2,2), strides=(2,2),padding='same'))

    model.add(Conv2D(32, (3,3), input_shape = inputShape ,padding='same'))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Activation("relu"))
    
    model.add(Conv2D(32, (3,3), input_shape = inputShape ,padding='same'))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Activation("relu"))

    model.add(Conv2D(32, (3,3), input_shape = inputShape ,padding='same'))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size = (2,2), strides=(2,2),padding='same'))

    model.add(Dropout(0.25))
    model.add(Flatten())

    model.add(Dense(512))
    model.add(Dense(1024))
    model.add(Dense(5))
    model.add(Activation('softmax'))
   
    return model


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
tlabel=data['label']
tdata=data.drop(['label'],axis=1)

X_train, X_test, y_train, y_test = train_test_split(tdata, tlabel, test_size=0.2, random_state=42)

print(X_train)
labels=y_train.to_numpy()
print(data.label.unique())
labels = np_utils.to_categorical(labels)
features=X_train
features=features.to_numpy()
print(features.shape)

from sklearn.preprocessing import MinMaxScaler

norm = MinMaxScaler().fit(features)
pickle.dump(norm,open('norm.pkl','wb'))

features=norm.transform(features)
print(features.shape)
features=np.reshape(features,(len(features),20,2,1))


print("***************")
print(features.shape)
print(labels)
print(labels.shape)


# # test['label']=newlabeldf_test
# print(y_test.unique())
t_labels=y_test.to_numpy()
t_labels = np_utils.to_categorical(t_labels)
t_features=X_test.to_numpy()
t_features=norm.transform(t_features)
t_features=np.reshape(t_features,(len(t_features),20,2,1))

print(t_features.shape)
print(t_labels.shape)
model=AlexNet()
model.compile(loss="categorical_crossentropy", optimizer='adam',
	metrics=["accuracy"])

H = model.fit(features,labels,validation_data=(t_features, t_labels),epochs=5, verbose=1,batch_size=256)


# save the model to disk
print("[INFO] serializing network...")
model.save('alexmodel.model')