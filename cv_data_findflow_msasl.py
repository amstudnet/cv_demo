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
from model import *
from torchinfo import summary
def extract_flows(source, dest, df):
  files = [f for f in listdir(source) if isfile(join(source, f))]
  if not os.path.exists(dest):
        os.mkdir(dest) 
  for i in range(0, df.shape[0]):
    start_frame = df.iloc[i,5]
    video = cv2.VideoCapture(source + df.iloc[i,12]) 
    file_path = dest  + df.iloc[i,12] + '_' + df.iloc[i,1]
    video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    ret, first_frame = video.read()

    first_frame= cv2.resize(first_frame, (224,224), interpolation = cv2.INTER_AREA)
    mask = np.zeros_like(first_frame)
    mask[..., 1] = 255


    prev = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    flows = prev

    for j in range(start_frame+1, start_frame+10):     
      if(ret):
        video.set(cv2.CAP_PROP_POS_FRAMES, j)
        ret, frame = video.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        current = cv2.resize(gray, (224,224), interpolation = cv2.INTER_AREA)
        flow = cv2.calcOpticalFlowFarneback(prev, current, None, pyr_scale = 0.5, levels = 5, winsize = 11, iterations = 5, poly_n = 5, poly_sigma = 1.1, flags = 0)
        
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        mask[..., 0] = angle * 180 / np.pi / 2
      
        mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

        rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)
        flow_image = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        flows = np.dstack((flows, flow_image))
        prev = current
      else:
        print(df.iloc[i,12], j)
    #print(file_path)    
    np.save(dest+"/"+file_path, flows)
def load_data(path,is_test):
  df = pd.read_json(path)
  # 過濾掉 clean_text 欄位中不在 classes 字典中的值
  for value in df['clean_text'].values:
    if value not in classes.keys():
      df.drop(df[df['clean_text']== value].index, inplace = True)

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
 
  return df
classes ={'teacher':0,  'eat':1 ,'hello':2 }
#classes ={'hello' : 0, 'nice' :1, 'teacher':2,  'eat':3 , 'no':4, 'happy':5, 'like':6, 'orange':7, 'want' :8, 'deaf':9}
#classes = "MSASL_classes.json"
train_json = "MSASL_train.json"
val_json = "MSASL_val.json"
test_json = "MSASL_test.json"
train_videos = "train_videos/"
val_videos = "val_videos/"
test_videos = "test_videos/" 

train_df = load_data(train_json,False)
val_df =load_data(val_json,False)
test_df = load_data(test_json,True)

train_df = clean_data(train_videos,train_df)
val_df = clean_data(val_videos,val_df)
test_df =  clean_data(test_videos,test_df)


extract_flows(train_videos, '3train_flows10', train_df)#(224,224,5)
extract_flows(val_videos,'3val_flows10',val_df)#(224,224,5)
extract_flows(test_videos, '3test_flows10', test_df)#(224,224,5)
"""
#test channel
a=np.load("train_flows5/train_flows5_8t-Avfk310.mp4_deaf.npy")
print(a.shape) #(224,224,5)
print(len(os.listdir('train_flows5')))#194
print(len(os.listdir('train_flows')))#194
print(len(os.listdir('val_flows5')))#63
print(len(os.listdir('val_flows')))#63
print(len(os.listdir('test_flows5')))#46
print(len(os.listdir('test_flows')))#46
"""