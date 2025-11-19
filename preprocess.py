import pandas as pd

# names="""duration
# protocol_type
# service
# flag
# src_bytes
# dst_bytes
# land
# wrong_fragment
# urgent
# hot
# num_failed_logins
# logged_in
# num_compromised
# root_shell
# su_attempted
# num_root
# num_file_creations
# num_shells
# num_access_files
# num_outbound_cmds
# is_host_login
# is_guest_login
# count
# srv_count
# serror_rate
# srv_serror_rate
# rerror_rate
# srv_rerror_rate
# same_srv_rate
# diff_srv_rate
# srv_diff_host_rate
# dst_host_count
# dst_host_srv_count
# dst_host_same_srv_rate
# dst_host_diff_srv_rate
# dst_host_same_src_port_rate
# dst_host_srv_diff_host_rate
# dst_host_serror_rate
# dst_host_srv_serror_rate
# dst_host_rerror_rate
# dst_host_srv_rerror_rate
# class
# diff"""


# columns=names.split('\n')
# print(columns)
# print(len(columns))

# df = pd.read_csv("KDDTrain+.txt")
# df.columns = columns
# df=df.drop(['diff'],axis=1)
# df.to_csv("dataset.csv",index=False)


df = pd.read_csv("dataset.csv")
# print(df.head(10))
print(df.shape)
# # print(df['label'].value_counts())
# drop_values = ['rootkit','warezclient','spy','back','nmap','perl','guess_passwd','pod','teardrop','imap','land','buffer_overflow','warezmaster','phf','multihop','ftp_write','loadmodule']
# # df=df[~df.label.str.contains("warezclient")]
# df=df[~df['label'].str.contains('|'.join(drop_values))]

print(df['label'].value_counts())
print(df.shape)

from sklearn.preprocessing import LabelEncoder
import pickle
# label_encoder object knows how to understand word labels.
protocol_enc = LabelEncoder()
 
# Encode labels in column 'species'.
df['protocol_type']= protocol_enc.fit_transform(df['protocol_type'])
with open('protocol_enc.pickle', 'wb') as f:
    pickle.dump(protocol_enc, f)

service_enc = LabelEncoder()
 
# Encode labels in column 'species'.
df['service']= protocol_enc.fit_transform(df['service'])
with open('service_enc.pickle', 'wb') as f:
    pickle.dump(service_enc, f)

flag_enc = LabelEncoder()
 
# Encode labels in column 'species'.
df['flag']= protocol_enc.fit_transform(df['flag'])
with open('flag_enc.pickle', 'wb') as f:
    pickle.dump(flag_enc, f)
print(df.head(10))

labeldf=df['label']



newlabeldf=labeldf.replace({ 'normal' : 0, 'neptune' : 1 ,'back': 1, 'land': 1, 'pod': 1, 'smurf': 1, 'teardrop': 1,'mailbomb': 1, 'apache2': 1, 'processtable': 1, 'udpstorm': 1, 'worm': 1,
                           'ipsweep' : 2,'nmap' : 2,'portsweep' : 2,'satan' : 2,'mscan' : 2,'saint' : 2
                           ,'ftp_write': 3,'guess_passwd': 3,'imap': 3,'multihop': 3,'phf': 3,'spy': 3,'warezclient': 3,'warezmaster': 3,'sendmail': 3,'named': 3,'snmpgetattack': 3,'snmpguess': 3,'xlock': 3,'xsnoop': 3,'httptunnel': 3,
                           'buffer_overflow': 4,'loadmodule': 4,'perl': 4,'rootkit': 4,'ps': 4,'sqlattack': 4,'xterm': 4})



df['label']=newlabeldf

print(df['label'].value_counts())


df.to_csv("final_dataset.csv",index=False)
