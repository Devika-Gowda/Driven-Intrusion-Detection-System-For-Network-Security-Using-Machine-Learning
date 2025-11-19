
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
from tensorflow.keras.models import model_from_json

json_file = open('model_lstm1.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("lstm_weight1.h5")
print("Loaded model from disk")
text=[[79.81768149763768,1.0,19.0,9.0,116.97260870176454,442.8949274852955,0.0,0.0,0.0,2.0,0.0,1.0,0.0,0.0,0.0,0.0,1.0,0.0,0.6757970994118212,0.0,0.0,1.0,1.0,1.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,83.34753674939742,1.0,0.6757970994118212,0.0032420290058817877,0.6757970994118212,0.0,0.0,0.0,0.01296811602352715]]
print(len(text))
lstm_trans=pickle.load( open( "minmaxlstm.pkl", "rb" ) )
X_test=lstm_trans.transform(text)
print(X_test)
feat=np.array(X_test)
print(feat.shape)
feat=np.reshape(feat,(1,40,1))
y=model.predict(feat)
print(y)
result=round(y[0][0])
print(result)