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
import mediapipe as mp

def extract_frames_with_mediapipe(source, dest, df):
    # 初始化 Mediapipe 模組
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils
    
    # 創建類別資料夾
    for class_name in classes.keys():
        class_path = os.path.join(dest, class_name)
        os.makedirs(class_path, exist_ok=True)

    # 處理每部影片
    for i in range(df.shape[0]):
        start_frame = df.iloc[i]['start']
        end_frame = df.iloc[i]['end']
        class_name = df.iloc[i]['clean_text']
        video_file = df.iloc[i]['url']
        box = df.iloc[i]['box']  # [x_min, y_min, x_max, y_max]

        video_path = os.path.join(source, video_file)
        if not os.path.exists(video_path):
            print(f"影片檔案不存在: {video_path}")
            continue

        # 加載影片
        video = cv2.VideoCapture(video_path)
        video.set(cv2.CAP_PROP_FPS, 30)
        if not video.isOpened():
            print(f"無法加載影片: {video_path}")
            continue

        # 初始化 Mediapipe Holistic
        with mp_holistic.Holistic(static_image_mode=True) as holistic:
            # 循環提取幀
            for j in range(start_frame, end_frame):
                video.set(cv2.CAP_PROP_POS_FRAMES, j)
                ret, image = video.read()
                if ret:
                    # 取得圖片尺寸
                    h, w, _ = image.shape

                    # 計算裁剪的像素座標
                    x_min = int(box[0] * w)
                    y_min = int(box[1] * h)
                    x_max = int(box[2] * w)
                    y_max = int(box[3] * h)

                    # 裁剪圖片
                    cropped_image = image[y_min:y_max, x_min:x_max]

                    # Mediapipe 處理
                    results = holistic.process(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
                    
                    # 在圖片上標記特徵
                    if results.left_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            cropped_image, 
                            results.left_hand_landmarks, 
                            mp_holistic.HAND_CONNECTIONS
                        )
                    if results.right_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            cropped_image, 
                            results.right_hand_landmarks, 
                            mp_holistic.HAND_CONNECTIONS
                        )

                    # 保存帶標記的圖片
                    output_path = os.path.join(dest, class_name, f"{video_file}_{j}.png")
                    cv2.imwrite(output_path, cropped_image)
                else:
                    print(f"無法提取幀: {video_file} 第 {j} 幀")
            
            video.release()

def extract_frames(source, dest, df):
    # 創建類別資料夾
    for class_name in classes.keys():
        class_path = os.path.join(dest, class_name)
        os.makedirs(class_path, exist_ok=True)

    # 處理每部影片
    for i in range(df.shape[0]):
        start_frame = df.iloc[i]['start']
        end_frame = df.iloc[i]['end']
        class_name = df.iloc[i]['clean_text']
        video_file = df.iloc[i]['url']
        box = df.iloc[i]['box']  # [x_min, y_min, x_max, y_max]
        
        
        video_path = os.path.join(source, video_file)
        if not os.path.exists(video_path):
            print(f"影片檔案不存在: {video_path}")
            continue

        # 加載影片
       
        video = cv2.VideoCapture(video_path)
        video.set(cv2.CAP_PROP_FPS,30)
        if not video.isOpened():
            print(f"無法加載影片: {video_path}")
            continue

        # 循環提取幀
        for j in range(start_frame, end_frame):
            video.set(cv2.CAP_PROP_POS_FRAMES, j)
            ret, image = video.read()
            if ret:
                # 取得圖片尺寸
                h, w, _ = image.shape

                # 計算裁剪的像素座標
                x_min = int(box[0] * w)
                y_min = int(box[1] * h)
                x_max = int(box[2] * w)
                y_max = int(box[3] * h)

                # 裁剪圖片
                cropped_image = image[y_min:y_max, x_min:x_max]

                # 保存裁剪後的圖片
                output_path = os.path.join(dest, class_name, f"{video_file}_{j}.png")
                cv2.imwrite(output_path, cropped_image)
            else:
                print(f"無法提取幀: {video_file} 第 {j} 幀")
        
        video.release()
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
classes ={'teacher':0,  'happy':1 }
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

extract_frames_with_mediapipe(train_videos, 'train_png', train_df)
extract_frames_with_mediapipe(val_videos,'val_png',val_df)
extract_frames_with_mediapipe(test_videos, 'test_png', test_df)

