from torch import nn
import torch
import torch.nn.functional as F
class SpatialNet(nn.Module):
  def __init__(self):
      super(SpatialNet, self).__init__()
      #Alexnet
      self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=11, padding=2, stride=4)
      self.conv2 = nn.Conv2d(in_channels=64, out_channels=192, kernel_size=5, padding=2)
      self.conv3 = nn.Conv2d(in_channels=192, out_channels=384, kernel_size=3, padding=1)
      self.conv4 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1)
      self.conv5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)

      self.fc1 = nn.Linear(in_features=256*6*6, out_features=4096)
      self.fc2 = nn.Linear(in_features=4096, out_features=1024)
      self.fc3 = nn.Linear(in_features=1024, out_features=2)

  def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # x = x.view(x.size(0), -1)
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=0.5)
        x = self.fc3(x)

        return x

class TemporalNet(nn.Module):
  def __init__(self):
      super(TemporalNet, self).__init__()
      
      self.conv1 = nn.Conv2d(in_channels=10, out_channels=64, kernel_size=11, padding=2, stride=4)
      self.conv2 = nn.Conv2d(in_channels=64, out_channels=192, kernel_size=5, padding=2)
      self.conv3 = nn.Conv2d(in_channels=192, out_channels=384, kernel_size=3, padding=1)
      self.conv4 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1)
      self.conv5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)

      self.fc1 = nn.Linear(in_features=256*6*6, out_features=4096)
      self.fc2 = nn.Linear(in_features=4096, out_features=1024)
      self.fc3 = nn.Linear(in_features=1024, out_features=2)

  

        
      
      

  def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # x = x.view(x.size(0), -1)
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=0.5)
        x = self.fc3(x)

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