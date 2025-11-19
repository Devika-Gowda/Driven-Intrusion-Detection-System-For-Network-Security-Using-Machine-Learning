from tensorflow.keras.models import load_model
import numpy as np
attack_list=['Normal','DOS','Probe','R2L','U2R']
loaded_model=load_model('alexmodel.model')

import pickle
text=[[79.81768149763768,1.0,19.0,9.0,116.97260870176454,442.8949274852955,0.0,0.0,0.0,2.0,0.0,1.0,0.0,0.0,0.0,0.0,1.0,0.0,0.6757970994118212,0.0,0.0,1.0,1.0,1.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,83.34753674939742,1.0,0.6757970994118212,0.0032420290058817877,0.6757970994118212,0.0,0.0,0.0,0.01296811602352715]]
print(len(text))
feat=np.array(text)
alex_scale=pickle.load( open( "norm.pkl", "rb" ) )
feat=alex_scale.transform(feat)
feat=np.reshape(feat,(1,20,2,1))
preds = loaded_model.predict(feat)[0]
result=np.argmax(preds)
print(result)