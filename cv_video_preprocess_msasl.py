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
from torchvision.models import resnet18
from PIL import Image

#0:org_text	1:clean_text 2:start_time 3:signer_id	4:signer	5:start	6:end	
#7:file	8:label	9:height	10:fps	 11:end_time	12:url	13:text	14:box	15:width 16:review
temporal_transform = transforms.Compose([
 
  
  transforms.ToTensor(),
  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),                   
])
#transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
#transforms.RandomHorizontalFlip(),
#transforms.RandomRotation(10),
#transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),

spatial_transform = transforms.Compose([
  transforms.ToPILImage(),
  transforms.Resize(256),
  transforms.CenterCrop(224),
  transforms.ToTensor(),
  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),                            
])#transforms.CenterCrop(224),
class SpatialDataset(datasets.ImageFolder):
    def __init__(self, path, df, classes, transform, dir):
        """
        初始化 SpatialDataset。

        Args:
            path (str): 圖片所在的根目錄。
            df (pd.DataFrame): 包含影片和標籤資訊的資料框。
            classes (dict): 類別字典，將文字標籤對應到數值標籤。
            transform (callable): 用於處理圖片的 transform。
            dir (str): 儲存已裁剪圖片的資料夾路徑。
        """
        self.path = path  # 根目錄路徑
        self.df = df  # 資料框
        self.transform = transform  # 圖片處理方式
        self.classes = classes  # 類別字典
        self.dir = dir  # 已裁剪圖片的資料夾路徑
        # 過濾掉不存在的檔案(因為生成出來的時候有的圖片超爛，我手動刪除)
        valid_indices = []
        for idx in range(len(df)):
            url = df.iloc[idx]['url']
            start_frame = df.iloc[idx]['start']
            class_name = df.iloc[idx]['clean_text']
            image_name = f"{url}_{start_frame}.png"
            image_path = os.path.join(self.dir, class_name, image_name)
            if os.path.exists(image_path):
                valid_indices.append(idx)

        # 更新 DataFrame，只保留存在的檔案
        self.df = df.iloc[valid_indices].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """
        根據索引返回一張圖片及其標籤。

        Args:
            idx (int): 索引值。

        Returns:
            tuple: 圖片張量及其標籤。
        """
        # 從資料框獲取對應的檔案名稱和類別
        url = self.df.iloc[idx]['url']
        start_frame = self.df.iloc[idx]['start']
        class_name = self.df.iloc[idx]['clean_text']
        label = self.classes[class_name]

        # 組合圖片檔案的名稱
        image_name = f"{url}_{start_frame}.png"
        image_path = os.path.join(self.dir, class_name, image_name)
        print(image_path)
 
        # 檢查圖片是否存在
        if not os.path.exists(image_path):
          print(f"圖片檔案不存在: {image_path}")
          raise FileNotFoundError(f"圖片檔案不存在: {image_path}")

        # 讀取圖片並應用 transform
        image = cv2.imread(image_path)
        
        #image = Image.open(image_path).convert("RGB")
        if self.transform:
            #imgae = torch.from_numpy(image)
            image =  self.transform(image)

        return (image, label)
class TemporalDataset(datasets.ImageFolder):

  def __init__(self,path,df, classes, transform,dir):
    self.path = path
    self.df = df
    self.transform = transform
    self.classes = classes
    self.dir = dir
  def __len__(self):
    return len(self.df)

  def __getitem__(self, idx):

    path =  self.dir +"/"+self.path + self.df.iloc[idx,12] + '_' +  self.df.iloc[idx,1] + ".npy"
    print(path)
    flow = np.load(path)
    flow = np.transpose(flow)
  


    label = self.classes[self.df.iloc[idx,1]]

    return  (flow, label) 
     
def getdataloader_sizes(net, batchsize): 
  if(net == 'temporal'):
    train =  TemporalDataset( path = '2train_flows10', df = train_df, classes = classes, transform = temporal_transform , dir="2train_flows10")
    validation = TemporalDataset( path = '2val_flows10', df = val_df,  classes = classes, transform = temporal_transform ,dir="2val_flows10")
    test = TemporalDataset( path = '2test_flows10', df = test_df,  classes = classes, transform = temporal_transform,dir ="2test_flows10")
  else:
    train =  SpatialDataset( path = "train_png", df = train_df, classes = classes, transform = spatial_transform,dir ="train_png")
    validation = SpatialDataset( path = "val_png", df = val_df,  classes = classes, transform = spatial_transform, dir="val_png")
    test = SpatialDataset( path = "test_png", df = test_df,  classes = classes, transform = spatial_transform,dir ="test_png")

  
  all_datasets = {'train' : train, 'validation' : validation, 'test' : test}

  dataloaders = {x: torch.utils.data.DataLoader(all_datasets[x], batch_size=batchsize,
                                               shuffle=True)
                for x in ['train', 'validation' ,'test']}
  dataset_sizes = {x: len(all_datasets[x]) for x in ['train', 'validation','test']}

  class_names = list(classes.keys())
  print(class_names)
  print(dataset_sizes)
  return dataloaders,dataset_sizes,class_names

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
#classes ={'teacher':0,  'eat':1 ,'hello':2 }
classes ={'teacher':0,  'happy':1 }
#classes ={'teacher':0,  'eat':1 }
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
#train_df做完load_data會剩下{'hello' : 0, 'nice' :1, 'teacher':2,  'eat':3 , 'no':4, 'happy':5, 'like':6, 'orange':7, 'want' :8, 'deaf':9}的json
#print(len(train_df['text'].values))#剩下472筆json
#print(train_df['url'].values)
"""
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
val_df = clean_data(val_videos,val_df)
test_df =  clean_data(test_videos,test_df)
"""
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
"""
洗資料前後
[('eat', 57), ('nice', 54), ('want', 53), ('orange', 50), ('teacher', 50), ('like', 48), ('deaf', 46), ('no', 46), ('happy', 38), ('hello', 30)]
[('eat', 31), ('teacher', 29), ('want', 29), ('like', 27), ('orange', 26), ('no', 23), ('deaf', 22), ('happy', 22), ('nice', 22), ('hello', 12)]
"""
"""
print(train_df.shape[0])
video = cv2.VideoCapture("train_videos/"+train_df.iloc[2,12])
start_frame = train_df.iloc[2,5]
end_frame = train_df.iloc[2,6]
video.set(cv2.CAP_PROP_POS_FRAMES,start_frame+1)
success, image = video.read()
if success:
    image = cv2.resize(image,(225,225),interpolation = cv2.INTER_AREA)
    mask = np.zeros_like(image)
    mask[...,1]=255 #chanel 1 255
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    video.set(cv2.CAP_PROP_POS_FRAMES, start_frame+5)
    sucess2,image2 = video.read()
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    image2 = cv2.resize(image2, (225,225), interpolation = cv2.INTER_AREA)
    flow = cv2.calcOpticalFlowFarneback(image, image2, None, pyr_scale = 0.5, levels = 5, winsize = 11, iterations = 5, poly_n = 5, poly_sigma = 1.1, flags = 0)
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    mask[..., 0] = angle * 180 / np.pi / 2 #chanel 1 255

    mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)#chanel 1 255

    rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)
    flow_image = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    plt.imshow(image)
    plt.show()
    plt.imshow(image2)
    plt.show()
    plt.imshow(flow_image)
    #plt.imshow(image)
    plt.show()

else:
    print("error")
"""

#extract_flows(train_videos, 'train_flows', train_df)#(224,224,10)
#extract_flows(val_videos,'val_flows',val_df)#(224,224,10)
#extract_flows(test_videos, 'test_flows', test_df)#(224,224,10)
#a=np.load("train_flows/train_flows_8t-Avfk310.mp4_deaf.npy")
#print(a.shape) #(224,224,10)


device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(device)
def plot_confusion_matrix(cm, classes,
                          cmap=plt.cm.Blues):
  
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    threshold = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j]),horizontalalignment="center",color="white" if cm[i, j] > threshold else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
def plot_graph(plotlist1,plotlist2,ylabel):
   
    plt.xlabel("Training Epochs")
    plt.ylabel(ylabel)
    plt.plot(plotlist1, color="green")
    plt.plot(plotlist2, color="red")
    
    plt.gca().legend(('Train', 'Validation'))
    plt.show()
     
def train_model(model, criterion, optimizer, epoch_number,device,earlystopping):
   
    model.to(device)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_train_acc = 0.0
    best_val_acc = 0.0
    best_test_acc = 0.0
    train_acc_history = list()
    train_loss_history =list()
    val_acc_history = list()
    val_loss_history =list()
    
    counter = 0
    stop =False
    best_loss = None
    
   
    n_epochs_stop = 1
    min_val_loss = np.inf
    epochs_no_improve = 0
    
    for epoch in range(epoch_number):
        if stop:
          break
        print('Epoch {}/{}'.format(epoch, epoch_number - 1))
        
        # Train and validation for each epoch
        for part in ['train', 'validation']:
            if part == 'train':
                
                model.train()  
            else:
                model.eval()  

            current_loss = 0.0
            current_phase_correct_outputnumber = 0
            # For each phase in datasets are iterated
            for inputs, labels in dataloaders[part]:
                inputs = inputs.to(torch.float32)
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                with torch.set_grad_enabled(part == 'train'):
                    
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    # Backpropagate and opitimize Training part
                    if part == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                current_loss += loss.item() * inputs.size(0)
                current_phase_correct_outputnumber += torch.sum(preds == labels.data)

            current_loss = current_loss / dataset_sizes[part]
            epoch_acc = 100*current_phase_correct_outputnumber.double() / dataset_sizes[part]

            if part == 'validation':
                val_acc_history.append(epoch_acc)
                val_loss_history.append(current_loss)
                if earlystopping:
                  # If the validation loss is at a minimum
                  if current_loss < min_val_loss:
                    # Save the model
                    epochs_no_improve = 0
                    min_val_loss = current_loss

                  else:
                    epochs_no_improve += 1
                    # Check early stopping condition
                    if epochs_no_improve == n_epochs_stop:
                      print('Early stopping!')
                      
                      #Printed best accuracies
                      print('Best train Acc: {:4f}'.format(best_train_acc))
                      print('Best validation Acc: {:4f}'.format(best_val_acc))

                      print()

                      #Printed best accuracies
                      print('Best train Acc: {:4f}'.format(best_train_acc))
                      print('Best validation Acc: {:4f}'.format(best_val_acc))

                      # load best model weights
                      model.load_state_dict(best_model_wts)
                      #Plot accuracy graph 
                      plot_graph(train_acc_history,val_acc_history,"Accuracy")
                      plot_graph(train_loss_history,val_loss_history,"Loss")
                      
                      return model                  
            else:
                train_acc_history.append(epoch_acc)
                train_loss_history.append(current_loss)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                part, current_loss, epoch_acc))

            # deep copy the model
            if part == 'train' and epoch_acc > best_train_acc:
                  best_train_acc = epoch_acc
                
            if part == 'validation' and epoch_acc > best_val_acc:             
                best_val_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
              
        print() 
    
              
    print('Best train Acc: {:4f}'.format(best_train_acc))
    print('Best validation Acc: {:4f}'.format(best_val_acc))
    
  
    model.load_state_dict(best_model_wts)
   
    plot_graph(train_acc_history,val_acc_history,"Accuracy")
    plot_graph(train_loss_history,val_loss_history,"Loss")
  
    return model
ImageFile.LOAD_TRUNCATED_IMAGES = True 
#spatial_net = SpatialNet()
#temporal_net = TemporalNet()
#spatial_model = SpatialNet()
spatial_model = models.resnet50(pretrained=True)
in_features = spatial_model.fc.in_features
fc = nn.Linear(in_features=in_features, out_features=len(classes))
spatial_model.fc = fc
#print(spatial_model )
temporal_model = TemporalNet()


#初始化參數
learning_rate = 0.001#0.00002
epoch = 50
batchsize = 32
criterion = nn.CrossEntropyLoss()
spatial_optimizer = torch.optim.Adam(spatial_model.parameters(), lr=learning_rate)
#spatial_lr_scheduler = torch.optim.lr_scheduler.StepLR(spatial_optimizer , step_size=7, gamma=0.1)

temporal_optimizer = torch.optim.Adam(temporal_model.parameters(), lr=learning_rate)
temporal_lr_scheduler = torch.optim.lr_scheduler.StepLR(temporal_optimizer , step_size=7, gamma=0.1)
earlystoping = False


dataloaders,dataset_sizes,class_names = getdataloader_sizes('spatial',batchsize)#temporal or else(空間)


trained_spatial_model = train_model(spatial_model, criterion, spatial_optimizer,epoch,device,earlystoping)
#trained_temporal_model = train_model(temporal_model, criterion, temporal_optimizer,epoch,device,earlystoping)

#fusion_model = FusionNet(trained_spatial_model,trained_temporal_model)
#fusion_optimizer = torch.optim.Adam(fusion_model.parameters(), lr=learning_rate)
#trained_fusion_model = train_model(fusion_model, criterion, fusion_optimizer,epoch,device,earlystoping)
def calculateTestAcc(trained_model,dataloaders,dataset_sizes,class_names):
  confusion_matrixx = torch.zeros(len(class_names), len(class_names))
  np.set_printoptions(precision=2)
  current_phase_correct_outputnumber = 0

  with torch.no_grad():
    # 使用 torch.no_grad() 表示禁用自動梯度計算。  
    # 減少記憶體使用，並加速推論過程，因為不需要計算反向傳播。
    for i, (inputs, classes) in enumerate(dataloaders['test']):
        inputs = inputs.to(torch.float32)
        inputs = inputs.to(device)
        # 將輸入數據移動到設備（例如 GPU 或 CPU）上。
        classes = classes.to(device)
        # 將標籤移動到同一設備上，確保輸入和標籤的設備一致。

        outputs = trained_model(inputs)
        # 將輸入數據通過已訓練的模型進行推論，獲取模型的輸出結果。
        _, preds = torch.max(outputs, 1)
        # 使用 `torch.max` 獲取每個樣本的預測類別。
        # `outputs` 是概率分佈，`preds` 是每個樣本預測的最大概率所對應的類別索引。
        print(classes.data)
        print(preds)
        current_phase_correct_outputnumber += torch.sum(preds == classes.data)
        # 比較預測類別 `preds` 與真實標籤 `classes.data`。
        # 將正確分類的樣本數加到 `current_phase_correct_outputnumber` 中。

        for t, p in zip(classes.view(-1), preds.view(-1)):
          confusion_matrixx[t.long(), p.long()] += 1
        
        #cm=confusion_matrix(classes.view(-1),preds.view(-1))
        #print(confusion_matrixx)
        # 更新混淆矩陣
       
    test_acc = 100*current_phase_correct_outputnumber.double() / dataset_sizes['test']
    confusion_matrix_normalized = confusion_matrixx / confusion_matrixx.sum(1, keepdim=True)
    print('Test Acc: {:4f}'.format(test_acc))
   
  plt.figure(figsize = (10,10))#(10,10)
  plot_confusion_matrix(confusion_matrix_normalized,classes=class_names)
  plt.show()
calculateTestAcc(trained_spatial_model  ,dataloaders,dataset_sizes,list(classes.keys()))#原本:trained_fusion_model
test_input = torch.randn(64, 3, 224, 224)#(batch_size=64, 通道數=10, 圖像尺寸=224x224)
summary(trained_spatial_model , input_data=test_input)#原本:trained_fusion_model


