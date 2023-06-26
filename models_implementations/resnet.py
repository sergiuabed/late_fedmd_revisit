import torch.nn as nn
from typing import Union

# 3x3 convolution
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=1, bias=False)

# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, stride:int=1, downsample:Union[None, nn.Sequential]=None, norm_layer:str="bn", groups:int=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        if norm_layer == "gn":
            self.bn1 = nn.GroupNorm(groups, out_channels)
            self.bn2 = nn.GroupNorm(groups, out_channels)
        else:
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

# ResNet
class ResNet(nn.Module):
    def __init__(self, layers:list[int], num_classes:int=10, norm_layer:str="bn", groups:int=1):
        super().__init__()
        self.in_channels = 16
        self.norm_layer = norm_layer
        self.groups = groups
        self.conv = conv3x3(3, self.in_channels, 2)
        if self.norm_layer == "gn":
            self.bn = nn.GroupNorm(groups, self.in_channels)    
        else:
            self.bn = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(16, layers[0])
        self.layer2 = self.make_layer(32, layers[1], 2)
        self.layer3 = self.make_layer(64, layers[2], 2)
        self.avg_pool = nn.AvgPool2d(2, 2)
        self.fc = nn.Linear(256, num_classes)

    def make_layer(self, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels) if self.norm_layer == "bn" else nn.GroupNorm(self.groups, out_channels))
        layers = []
        layers.append(ResidualBlock(self.in_channels, out_channels, stride, downsample, self.norm_layer, self.groups))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels, norm_layer = self.norm_layer, groups=self.groups))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
    

