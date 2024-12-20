import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, dataset, random_split
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import cv2
import itertools
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
#from google.colab.patches import cv2_imshow
import collections
from PIL import ImageFile
import torch.nn.functional as F
from os import listdir
from os.path import isfile, join
import json
#0:org_text	1:clean_text 2:start_time 3:signer_id	4:signer	5:start	6:end	
#7:file	8:label	9:height	10:fps	 11:end_time	12:url	13:text	14:box	15:width 16:review
st=[]
st1=[]
st2=[]
class SpatialDataset(datasets.ImageFolder):

    def __init__(self,path,df, classes, transform):
      self.path = path
      self.df = df
      self.transform = transform
      self.classes = classes

    def __len__(self):
      return len(self.df)

    def __getitem__(self, idx):

      url = self.df.iloc[idx][12]
      video = cv2.VideoCapture(self.path + url)
    
      frame = self.df.iloc[idx][6]  - self.df.iloc[idx][5]
      video.set(cv2.CAP_PROP_POS_FRAMES, self.df.iloc[idx][5])
      ret, image = video.read()
      
      if self.transform:
        image = self.transform(image)

      label = self.classes[self.df.iloc[idx][1]]
      return  (image, label)
def getdataloader_sizes(batchsize): 
  train =  SpatialDataset( path = train_videos, df = train_df, classes = classes, transform = transform)
  validation = SpatialDataset( path = val_videos, df = val_df,  classes = classes, transform = transform)
  test = SpatialDataset( path = test_videos, df = test_df,  classes = classes, transform = transform)

  
  all_datasets = {'train' : train, 'validation' : validation, 'test' : test}

  dataloaders = {x: torch.utils.data.DataLoader(all_datasets[x], batch_size=batchsize,
                                               shuffle=True)
                for x in ['train', 'validation' ,'test']}
  dataset_sizes = {x: len(all_datasets[x]) for x in ['train', 'validation','test']}

  class_names = list(classes.keys())
  print(class_names)
  print(dataset_sizes)
  return dataloaders,dataset_sizes,class_names   
def extract_frames(source, dest, df):
  files = [f for f in listdir(source) if isfile(join(source, f))]
  for i in range(0, df.shape[0]):
    start_frame = df.iloc[i][5]
    end_frame = df.iloc[i][6]
    video = cv2.VideoCapture(source + df.iloc[i][12])
    for j in range(start_frame, end_frame):     
      video.set(cv2.CAP_PROP_POS_FRAMES, j)   
      ret, image = video.read()
      if(ret):
        file_path = dest + df.iloc[i][1] + '/' + df.iloc[i][12] + '_' + str(j) + '.png'
        cv2.imwrite(file_path, image)
      else:
        print(df.iloc[i][12], j)
def check_device():
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    print(torch.version.cuda)
#check_device()
def load_data(path,is_test):
  df = pd.read_json(path)
  # 過濾掉 clean_text 欄位中不在 classes 字典中的值
  for value in df['clean_text'].values:
    if value not in classes.keys():
      df.drop(df[df['clean_text']== value].index, inplace = True)
      st.append(value)
    else:
      st1.append(value) 
  # 將 url 欄位值的開頭替換為 'w'，並移除 'https://www.youtube.com'
  df['url'] = df['url'].apply(lambda x :  x.lstrip('https://www.youtube.com/watch?v='))
  return df
def clean_data(path, df):
  #問題:會清掉所有影片   因為json的url欄位名字跟影片名字完全不同
  # 取得目錄中所有檔案的名稱
  files = [os.path.splitext(f)[0] for f in listdir(path) if isfile(join(path, f))]
  #print(files)
  with_ext = [f for f in listdir(path) if isfile(join(path, f))]# 取得 DataFrame 中 url 欄位的所有值
  #print(with_ext)
  org_values = df['url'].values# 過濾掉 url 欄位中不在檔案清單中的值
  #print(len(org_values))
  for value in org_values:
    if value not in files:
      df.drop(df[df['url']== value].index, inplace = True)

  
  # 替換 url 欄位中的值為具有副檔名的檔案名
  for i in range(0, df.shape[0]):
    #print(df.iloc[i,12])#第12欄是url
    if df.iloc[i,12] in files:# 檢查第 12 欄(網址)是否在檔案名清單中
        index = files.index(df.iloc[i,12])
        df = df.replace(df.iloc[i,12], with_ext[index])# 替換為完整檔名
        #print(df.iloc[i,12])
    else:
        st2.append(df.iloc[i,12])
 
  return df
classes ={'hello' : 0, 'nice' :1, 'teacher':2,  'eat':3 , 'no':4, 'happy':5, 'like':6, 'orange':7, 'want' :8, 'deaf':9}
#classes = "MSASL_classes.json"
train_json = "MSASL_train.json"
val_json = "MSASL_val.json"
test_json = "MSASL_test.json"
train_videos = "train_videos/"
val_videos = "val_videos/"
test_videos = "test_videos/" 
train_df = load_data(train_json,False)
#train_df做完load_data會剩下{'hello' : 0, 'nice' :1, 'teacher':2,  'eat':3 , 'no':4, 'happy':5, 'like':6, 'orange':7, 'want' :8, 'deaf':9}的json
#print(len(train_df['text'].values))#剩下472筆json
#print(train_df['url'].values)
c={}
for text in train_df["text"].values:
    if c.get(text)==None:
        c[text]=1
    else:
        c[text]+=1
        #print(i)
#print(c['door'])
c_sorted = sorted(c.items(), key = lambda x:(-(x[1]), x[0]))
print(c_sorted)
"""
org_text                                                   like
clean_text                                                 like
start_time                                                  0.0
signer_id                                                   269
signer                                                       53
start                                                         0
end                                                          52
file                                     SignSchool really like
label                                                         6
height                                                      360
fps                                                       29.97
end_time                                                  1.735
url                                                       7y5Ye-2-ZBs   #7y5Ye-2-ZBs是影片的名字
text                                                       like
box           [0.040461480617523006, 0.335311889648437, 0.99...
width                                                       640
review                                                      NaN
Name: 15, dtype: object
"""
#print(len(listdir(train_videos)))#4712部影片
#print(len(list(set(st))))#拿掉1038個字
#st1=list(set(st1))
#print(len(st1))#472

train_df = clean_data(train_videos,train_df)
c1={}
for text in train_df["text"].values:
    if c1.get(text)==None:
        c1[text]=1
    else:
        c1[text]+=1
        #print(i)
#print(c['door'])
c1_sorted = sorted(c1.items(), key = lambda x:(-(x[1]), x[0]))
print(c1_sorted)
#print(len(st2))#108
#print(train_df)#243
#print(train_df.iloc[5,12])
"""
洗資料前後
[('eat', 57), ('nice', 54), ('want', 53), ('orange', 50), ('teacher', 50), ('like', 48), ('deaf', 46), ('no', 46), ('happy', 38), ('hello', 30)]
[('eat', 31), ('teacher', 29), ('want', 29), ('like', 27), ('orange', 26), ('no', 23), ('deaf', 22), ('happy', 22), ('nice', 22), ('hello', 12)]
"""
print(train_df.shape[0])
video = cv2.VideoCapture("train_videos/"+train_df.iloc[2,12])
video.set(cv2.CAP_PROP_POS_FRAMES,30)
success, image = video.read()
if success:
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.show()
else:
    print("error")



transform = transforms.Compose([
  transforms.ToPILImage(),
  transforms.ToTensor(),
  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),                            
])
     