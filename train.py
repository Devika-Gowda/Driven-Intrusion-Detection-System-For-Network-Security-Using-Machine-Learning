
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
pickle.dump(norm,open('vggscale.pkl','wb'))

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




model = MiniVGG.build(
	width=2, height=20,
	depth=1, classes=5)

# initialize the optimizer (SGD is sufficient)
#opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
EPOCHS=10

model.compile(loss="categorical_crossentropy", optimizer='adam',
	metrics=["accuracy"])

H = model.fit(features,labels,validation_data=(t_features, t_labels),epochs=EPOCHS, verbose=1,batch_size=256)

# save the model to disk
print("[INFO] serializing network...")
model.save('model.model')

# save the multi-label binarizer to disk
# print("[INFO] serializing label binarizer...")
# f = open('mlb.pkl', "wb")
# f.write(pickle.dumps(mlb))
# f.close()

plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper left")
plt.savefig("plot_ep.jpg")