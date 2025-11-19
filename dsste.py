
from imutils import paths
import random
import cv2
import os
import numpy as np


from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split

import pickle
import matplotlib.pyplot as plt


import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

import pickle



data=pd.read_csv('data.csv')




def dot_replace(x):
    x1=x.replace('.','')
    return x1

data['label']=data['label'].apply(dot_replace)


print(data.label.unique())

labeldf=data['label']



newlabeldf=labeldf.replace({ 'normal' : 0, 'neptune' : 1 ,'back': 1, 'land': 1, 'pod': 1, 'smurf': 1, 'teardrop': 1,'mailbomb': 1, 'apache2': 1, 'processtable': 1, 'udpstorm': 1, 'worm': 1,
                           'ipsweep' : 2,'nmap' : 2,'portsweep' : 2,'satan' : 2,'mscan' : 2,'saint' : 2
                           ,'ftp_write': 3,'guess_passwd': 3,'imap': 3,'multihop': 3,'phf': 3,'spy': 3,'warezclient': 3,'warezmaster': 3,'sendmail': 3,'named': 3,'snmpgetattack': 3,'snmpguess': 3,'xlock': 3,'xsnoop': 3,'httptunnel': 3,
                           'buffer_overflow': 4,'loadmodule': 4,'perl': 4,'rootkit': 4,'ps': 4,'sqlattack': 4,'xterm': 4})



data['label']=newlabeldf

print(data['label'].value_counts())


data1=data[data['label']==0]
data1=data1.sample(frac=1,replace=True)
data1.to_csv('0.csv')





from keras.utils import np_utils





def most_frequent(List):
    counter = 0
    num = List[0]
     
    for i in List:
        curr_frequency = List.count(i)
        if(curr_frequency> counter):
            counter = curr_frequency
            num = i
 
    return num




from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans  
def dsste(S):
    SE=[]
    SD=[]
    feats=S.drop(['label'],axis=1)
    labels=S['label'].values
    nn = NearestNeighbors(n_neighbors=50, algorithm='ball_tree')
    nn.fit(feats)
    #nn=pickle.load(open('nn.pkl','rb'))

    for i in range(len(feats)):
      f1=feats.iloc[i]
      lbl=labels[i]
      _,nbr=nn.kneighbors(np.array([f1]))
      nbr_lbl=[labels[j] for j in nbr]

      mf=most_frequent(nbr_lbl)

      print(lbl)
      print(mf)

      if mf==lbl:
        print('e')
        SE.append(i)
      else:
        print('d')
        SD.append(i)  
    
    Smaj=[]
    Smin=[]
    for i in SD:
        if labels[i]==0:
            Smaj.append(i)
        else:
            Smin.append(i)

    Smaj1=[features.iloc[i] for i in Smaj]
    Smin1=[features.iloc[i] for i in Smin]
    kmeans = KMeans(50)     
    print('clustering....')   
    kmeans.fit(np.array(Smaj1))
    
    Smaj1=list(kmeans.cluster_centers_)
    cnt='protocol_type,service,flag'
    try:
        XD=Smin1.drop(cnt)
        XC=Smin[[cnt]]
        Y=Smin['label']
        K=len(Smin1)
        SZ=[]
        for n in range(0,K):
            XD1=XD
            XC1=XC*(1-1/n)
            XD2=XD
            XC2=XC*(1+1/n)
            SZ.append(pd.concate(XD1,XC1,Y))
            SZ.append(pd.concate(XD2,XC2,Y))
        final=pd.concat([SE,Smaj1,Smin1,SZ])
        final.to_csv('balanced1.csv')

    except:
        pass
                        

if __name__=="__main__":
    dsste(data)





          




         
      
      



# x_train=data.drop(['label'],axis=1).astype('float64')    
# columns=x_train.columns
# y_train=data['label'].values

# from imblearn.over_sampling import SMOTE
# print('b')
# oversample = SMOTE()
# x_train, y_train= oversample.fit_resample(x_train, y_train)
# print('b2')
# df=pd.DataFrame(x_train,columns=columns)
# df['label']=y_train

# df.to_csv('balanced.csv')

# print(x_train.value_counts())


    











