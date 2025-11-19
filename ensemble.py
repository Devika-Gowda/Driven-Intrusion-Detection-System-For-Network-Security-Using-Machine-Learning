from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.models import model_from_json
import pickle
attack_list=['Normal','DOS','Probe','R2L','U2R']
loaded_model1=load_model('alexmodel.model')
print("Loaded Alexnet model from disk")

relist=[]
text=[[79.81768149763768,1.0,19.0,9.0,116.97260870176454,442.8949274852955,0.0,0.0,0.0,2.0,0.0,1.0,0.0,0.0,0.0,0.0,1.0,0.0,0.6757970994118212,0.0,0.0,1.0,1.0,1.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,83.34753674939742,1.0,0.6757970994118212,0.0032420290058817877,0.6757970994118212,0.0,0.0,0.0,0.01296811602352715]]
print(len(text))
featalex=np.array(text)
alex_scale=pickle.load( open( "norm.pkl", "rb" ) )
featalex=alex_scale.transform(featalex)
featalex=np.reshape(featalex,(1,20,2,1))
preds = loaded_model1.predict(featalex)[0]
alex_result=np.argmax(preds)
print("alexnet result==>",alex_result)
relist.append(alex_result)


json_file = open('model_lstm1.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model_lstm = model_from_json(loaded_model_json)
# load weights into new model
model_lstm.load_weights("lstm_weight1.h5")
print("Loaded LSTM model from disk")

lstm_trans=pickle.load( open( "minmaxlstm.pkl", "rb" ) )
X_test=lstm_trans.transform(text)
print(X_test)
feat=np.array(X_test)
print(feat.shape)
feat=np.reshape(feat,(1,40,1))
y=model_lstm.predict(feat)
print(y)
lstm_result=round(y[0][0])
print("LSTM result==>",lstm_result)
relist.append(lstm_result)

minivgg_model=load_model('model.model')
featvgg=np.array(text)
vgg_scale=pickle.load( open( "norm.pkl", "rb" ) )
featvgg=vgg_scale.transform(featvgg)
featvgg=np.reshape(featvgg,(1,20,2,1))
preds = minivgg_model.predict(featvgg)[0]
result_vgg=np.argmax(preds)
print("Mini VGG result==>",result_vgg)

relist.append(result_vgg)

print(relist)
finalindex=max(relist, key = relist.count)
print(finalindex)

print("Intrusion type==>",attack_list[finalindex])