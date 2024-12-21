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
#0:org_text	1:clean_text 2:start_time 3:signer_id	4:signer	5:start	6:end	
#7:file	8:label	9:height	10:fps	 11:end_time	12:url	13:text	14:box	15:width 16:review
st=[]
st1=[]
st2=[]
temporal_transform = transforms.Compose([
  transforms.RandomHorizontalFlip(),
  transforms.RandomRotation(10),
  transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
  transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
  transforms.ToTensor(),
  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),                   
])

spatial_transform = transforms.Compose([
  transforms.ToPILImage(),
  transforms.Resize(256),
  transforms.CenterCrop(224),
  transforms.ToTensor(),
  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),                            
])

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
class SpatialDataset(datasets.ImageFolder):

    def __init__(self,path,df, classes, transform,dir):
      self.path = path
      self.df = df
      self.transform = transform
      self.classes = classes
      self.dir = dir
    def __len__(self):
      return len(self.df)

    def __getitem__(self, idx):

      url = self.df.iloc[idx,12]
      video = cv2.VideoCapture(self.dir+"/"+self.path + url)
    
      frame = self.df.iloc[idx,6]  - self.df.iloc[idx,5]
      video.set(cv2.CAP_PROP_POS_FRAMES, self.df.iloc[idx,5])
      ret, image = video.read()
      
      if self.transform:
        image = self.transform(image)

      label = self.classes[self.df.iloc[idx,1]]
      return  (image, label)
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
    train =  TemporalDataset( path = 'train_flows', df = train_df, classes = classes, transform = temporal_transform , dir="train_flows")
    validation = TemporalDataset( path = 'val_flows', df = val_df,  classes = classes, transform = temporal_transform ,dir="val_flows")
    test = TemporalDataset( path = 'test_flows', df = test_df,  classes = classes, transform = temporal_transform,dir ="test_flows")
  else:
    train =  SpatialDataset( path = train_videos, df = train_df, classes = classes, transform = spatial_transform,dir ="train_videos")
    validation = SpatialDataset( path = val_videos, df = val_df,  classes = classes, transform = spatial_transform, dir="val_videos")
    test = SpatialDataset( path = test_videos, df = test_df,  classes = classes, transform = spatial_transform,dir ="test_videos")

  
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
    start_frame = df.iloc[i,5]
    end_frame = df.iloc[i,6]
    video = cv2.VideoCapture(source + df.iloc[i,12])
    for j in range(start_frame, end_frame):     
      video.set(cv2.CAP_PROP_POS_FRAMES, j)   
      ret, image = video.read()
      if(ret):
        file_path = dest + df.iloc[i,1] + '/' + df.iloc[i,12] + '_' + str(j) + '.png'
        cv2.imwrite(file_path, image)
      else:
        print(df.iloc[i,12], j)
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
a=np.load("train_flows/train_flows_8t-Avfk310.mp4_deaf.npy")
print(a.shape) #(224,224,10)


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
    plt.plot(plotlist2, color="yellow")
    
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
spatial_model = SpatialNet()
temporal_model = TemporalNet()


#初始化參數
learning_rate = 0.001#0.00002
epoch = 80
batchsize = 64
criterion = nn.CrossEntropyLoss()
spatial_optimizer = torch.optim.Adam(spatial_model.parameters(), lr=learning_rate)
temporal_optimizer = torch.optim.Adam(temporal_model.parameters(), lr=learning_rate)
earlystoping = False


dataloaders,dataset_sizes,class_names = getdataloader_sizes('temporal',batchsize)#temporal or else(空間)


trained_spatial_model = train_model(spatial_model, criterion, spatial_optimizer,epoch,device,earlystoping)
trained_temporal_model = train_model(temporal_model, criterion, temporal_optimizer,epoch,device,earlystoping)

fusion_model = FusionNet(trained_spatial_model,trained_temporal_model)
fusion_optimizer = torch.optim.Adam(fusion_model.parameters(), lr=learning_rate)
trained_fusion_model = train_model(fusion_model, criterion, fusion_optimizer,epoch,device,earlystoping)
def calculateTestAcc(trained_model,dataloaders,dataset_sizes):
  confusion_matrixx = torch.zeros(10, 10)
  np.set_printoptions(precision=2)
  current_phase_correct_outputnumber = 0

  with torch.no_grad():
    for i, (inputs, classes) in enumerate(dataloaders['test']):
        inputs = inputs.to(torch.float32)
        inputs = inputs.to(device)
        classes = classes.to(device)
        outputs = trained_model(inputs)
        _, preds = torch.max(outputs, 1)
        current_phase_correct_outputnumber += torch.sum(preds == classes.data)
          
        for t, p in zip(classes.view(-1), preds.view(-1)):
            confusion_matrixx[t.long(), p.long()] += 1
   
    test_acc = 100*current_phase_correct_outputnumber.double() / dataset_sizes['test']

    print('Test Acc: {:4f}'.format(test_acc))
    
  plt.figure(figsize = (10,10))
  plot_confusion_matrix(confusion_matrixx,classes=class_names)
  plt.show()
calculateTestAcc(trained_fusion_model,dataloaders,dataset_sizes)
test_input = torch.randn(64, 10, 224, 224)#(batch_size=64, 通道數=10, 圖像尺寸=224x224)
summary(trained_fusion_model, input_data=test_input)


