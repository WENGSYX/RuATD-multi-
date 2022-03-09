import pandas as pd
import os
os.chdir('./data')
df = pd.read_csv('train.csv')
status_dict = df['Class'].unique().tolist()
pd.to_pickle(status_dict,'../status.pk')
df['label']=df['Class'].apply(lambda x : status_dict.index(x))
data = []
for i in range(len(df)):
    data.append({'sen':df.iloc[i]['Text'],'label':df.iloc[i]['label']})
pd.to_pickle(data,'../train.pk')

df = pd.read_csv('val.csv')
df['label']=df['Class'].apply(lambda x : status_dict.index(x))
data = []
for i in range(len(df)):
    data.append({'sen':df.iloc[i]['Text'],'label':df.iloc[i]['label']})
pd.to_pickle(data,'../valid.pk')

df = pd.read_csv('test.csv')
data = []
for i in range(len(df)):
    data.append({'sen':df.iloc[i]['Text'],'label':0})
pd.to_pickle(data,'../test.pk')
