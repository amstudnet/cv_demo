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
      self.fc3 = nn.Linear(in_features=1024, out_features=29)

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
spatial_net = SpatialNet()