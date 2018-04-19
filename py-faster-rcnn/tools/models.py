# -*- coding: utf-8 -*-
import torch.nn as nn
from torchvision import models
import torch
import torch.nn.functional as F
import torch.nn.init as init

class VGG(nn.Module):
    def __init__(self, name='vgg16', cls_num=2):
        super(VGG, self).__init__()
        if name=='vgg16':
            net = models.vgg16(True).features
        elif name=='vgg16_bn':
            net = models.vgg16_bn(True).features
        elif name=='vgg19':
            net = models.vgg19(True).features
        elif name=='vgg19_bn':
            net = models.vgg19_bn(True).features
        else:
            print 'Wrong Name'
            quit()
        self.features = net
        self.classifier = nn.Sequential(
            nn.Linear(in_features=2048, out_features=512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(in_features=512, out_features=512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(in_features=512, out_features=2),
        )
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class ResNet(nn.Module):
    def __init__(self, name='resnet50', cls_num=2):
        super(ResNet, self).__init__()
        if name=='resnet50':
            net = models.resnet50(True)
        elif name=='resnet18':
            net = models.resnet18(True)
        elif name=='resnet34':
            net = models.resnet34(True)
        else:
            print 'Wrong Name'
            quit()
        self.conv1 = net.conv1
        self.bn1 = net.bn1
        self.relu = net.relu
        #self.maxpool = net.maxpool
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4
        self.classifier = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            nn.Linear(in_features=2048, out_features=2)
        )
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def Net(name='vgg16', cls_num=2):
    if 'vgg' in name:
        return VGG(name, cls_num)
    elif 'res' in name:
        return ResNet(name, cls_num)
