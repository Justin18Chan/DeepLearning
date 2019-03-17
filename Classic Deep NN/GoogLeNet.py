import torch.nn as nn
import torch as torch
import torch.nn.functional as F
import torchvision.models.inception
class BasicConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,**kwargs):
        super(BasicConv2d,self).__init__()
        self.conv = nn.Conv2d(in_channels,out_channels,bias=False,**kwargs)
        self.bn = nn.BatchNorm2d(out_channels,eps=0.001)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x,inplace=True)

class Inception(nn.Module):
    def __init__(self,in_channels,pool_features):
        super(Inception,self).__init__()
        self.branch1X1 = BasicConv2d(in_channels,64,kernel_size = 1)

        self.branch5X5_1 = BasicConv2d(in_channels,48,kernel_size = 1)
        self.branch5X5_2 = BasicConv2d(48,64,kernel_size=5,padding = 2)

        self.branch3X3_1 = BasicConv2d(in_channels,64,kernel_size = 1)
        self.branch3X3_2 = BasicConv2d(64,96,kernel_size = 3,padding = 1)
        # self.branch3X3_2 = BasicConv2d(96, 96, kernel_size=1,padding = 1)

        self.branch_pool = BasicConv2d(in_channels,pool_features,kernel_size = 1)
    def forward(self, x):
        branch1X1 = self.branch1X1(x)

        branch5X5 = self.branch5X5_1(x)
        branch5X5 = self.branch5X5_2(branch5X5)

        branch3X3 = self.branch3X3_1(x)
        branch3X3 = self.branch3X3_2(branch3X3)

        branch_pool = F.avg_pool2d(x,kernel_size = 3,stride = 1,padding = 1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1X1,branch3X3,branch5X5,branch_pool]
        return torch.cat(outputs,1)
