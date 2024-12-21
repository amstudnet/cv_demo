from torch import nn
import torch
import torch.nn.functional as F
class SpatialNet(nn.Module):
  def __init__(self):
      super(SpatialNet, self).__init__()
      self.conv1 = nn.Conv2d(10, 96, kernel_size=7, stride=2)#3
      self.bn1 = nn.BatchNorm2d(96)
      self.relu1 = nn.ReLU()
      self.pool1 = nn.MaxPool2d(3, stride=2)
      self.norm1 = nn.LocalResponseNorm(2)
      self.conv2 = nn.Conv2d(96, 256, kernel_size=5, stride=2)
      self.bn2 = nn.BatchNorm2d(256)
      self.relu2 = nn.ReLU()
      self.pool2 = nn.MaxPool2d(3, stride=2)
      self.norm2 = nn.LocalResponseNorm(2)
      self.conv3 = nn.Conv2d(256, 512, kernel_size=3)
      self.bn3 = nn.BatchNorm2d(512)
      self.relu3 = nn.ReLU()
      self.conv4 = nn.Conv2d(512, 512, kernel_size=3)
      self.bn4 = nn.BatchNorm2d(512)
      self.relu4= nn.ReLU()
      self.conv5 = nn.Conv2d(512, 512, kernel_size=3)
      self.bn5 = nn.BatchNorm2d(512)       
      self.relu5 = nn.ReLU()
      self.pool3 =nn.MaxPool2d(3, stride=2)
  
  
      self.l1 = nn.Linear(2048, 4096)
      self.drop1 = nn.Dropout(p=0.4)
      self.l2 = nn.Linear(4096, 2048)
      self.drop2 = nn.Dropout(p=0.3)
      self.l3 =nn.Linear(2048, 10)
        
  def forward(self, x):
      x = self.conv1(x)
      x = self.relu1(x)
      x = self.pool1(x)
      x = self.norm1(x)

      x = self.conv2(x)
      x = self.relu2(x)
      x = self.pool2(x)
      x = self.norm2(x)

      x = self.conv3(x)
      x = self.relu3(x)

      x = self.conv4(x)
      x = self.relu4(x)


      x = self.conv5(x)
      x = self.relu5(x)
      x = self.pool3(x)
      print("spatial Shape before flattening:", x.shape)  # 打印形狀
      x = x.view(x.size(0), -1)
      print("spatial Shape after flattening:", x.shape)  # 打印形狀
      x = self.l1(x)
      x = self.drop1(x)
      x = self.l2(x)
      x = self.drop2(x)
      x = self.l3(x)

      return x

class TemporalNet(nn.Module):
  def __init__(self):
      super(TemporalNet, self).__init__()
      
      self.conv1 = nn.Conv2d(10, 96, kernel_size=7, stride=2)
      self.bn1 = nn.BatchNorm2d(96)
      self.relu1 = nn.ReLU()
      self.pool1 = nn.MaxPool2d(3, stride=2)
      self.norm1 = nn.LocalResponseNorm(2)
      self.conv2 = nn.Conv2d(96, 256, kernel_size=5, stride=2)
      self.bn2 = nn.BatchNorm2d(256)
      self.relu2 = nn.ReLU()
      self.pool2 = nn.MaxPool2d(3, stride=2)
      self.norm2 = nn.LocalResponseNorm(2)
      self.conv3 = nn.Conv2d(256, 512, kernel_size=3)
      self.bn3 = nn.BatchNorm2d(512)
      self.relu3 = nn.ReLU()
      self.conv4 = nn.Conv2d(512, 512, kernel_size=3)
      self.bn4 = nn.BatchNorm2d(512)
      self.relu4= nn.ReLU()
      self.conv5 = nn.Conv2d(512, 512, kernel_size=3) 
      self.bn5 = nn.BatchNorm2d(512)          
      self.relu5 = nn.ReLU()
      self.pool3 =nn.MaxPool2d(3, stride=2)

  
      self.l1 = nn.Linear(2048, 4096)
      self.drop1 = nn.Dropout(p=0.5)
      self.l2 = nn.Linear(4096, 2048)
      self.drop2 = nn.Dropout(p=0.5)
      self.l3 =nn.Linear(2048, 10)
        
      
      

  def forward(self, x):
      x = self.conv1(x.float())
      x = self.relu1(x)
      x = self.pool1(x)
      x = self.norm1(x)

      x = self.conv2(x)
      x = self.relu2(x)
      x = self.pool2(x)
      x = self.norm2(x)

      x = self.conv3(x)
      x = self.relu3(x)

      x = self.conv4(x)
      x = self.relu4(x)


      x = self.conv5(x)
      x = self.relu5(x)
      x = self.pool3(x)
      print("temporal Shape before flattening:", x.shape)  # 打印形狀
      x = x.view(x.size(0), -1)
      print("temporal Shape before flattening:", x.shape)  # 打印形狀
      x = self.l1(x)
      x = self.drop1(x)
      x = self.l2(x)
      x = self.drop2(x)
      x = self.l3(x)

      return x

class FusionNet(nn.Module):
    def __init__(self, spatial_net, temporal_net):
        super(FusionNet, self).__init__()
        self.spatial = spatial_net
        self.temporal = temporal_net
        
        # 替換最後一層為 Identity
        self.spatial.l3 = nn.Identity()
        self.temporal.l3 = nn.Identity()
        
        self.classifier = nn.Linear(2048+2048, 10)#2048+512
        
    def forward(self, x):
        x1 = self.spatial(x.clone())  
        print("spatial Shape before flattening:", x1.shape)  # 打印形狀
        x1 = x1.view(x1.size(0), -1)
        print("spatial Shape after flattening:", x1.shape)  # 打印形狀
        x2 = self.temporal(x)
        print("temporal Shape before flattening:", x2.shape)  # 打印形狀
        x2 = x2.view(x2.size(0), -1)
        print("temporal Shape after flattening:", x2.shape)  # 打印形狀
        x = torch.cat((x1, x2), dim=1)# (64,2048) (64,2048) =(64,4096)
        print("x1+x2",x.shape)
        x = self.classifier(x)#F.relu(x)
        #x = F.softmax(x, dim=1)  # 添加 Softmax 激活函數
        return x    
spatial_net = SpatialNet()
temporal_net = TemporalNet()