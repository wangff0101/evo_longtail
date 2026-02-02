import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
from torch.nn import Parameter
import math
import os
import sys
import importlib.util

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride


    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        out = self.relu(out)

        return out


class LT_DECODER(nn.Module):
    def __init__(self, feat_dim=512, cls_num_tensor=None):
        super(LT_DECODER, self).__init__()
        self.fc1 = nn.Linear(feat_dim, 64) 
        self.fc2 = nn.Linear(64, feat_dim)  
        self.bn1 = nn.BatchNorm1d(64)  
        self.bn2 = nn.BatchNorm1d(feat_dim) 
        self.relu = nn.ReLU(inplace=True)  

    def forward(self, feats):
        x = self.fc1(feats)
        x = self.bn1(x) 
        x = self.relu(x)  
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x=x+feats

        return x 


class ResNet(nn.Module):
    def __init__(self, block, layers, pool_size=7, num_classes=100, task_list=None):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.task_list = task_list if task_list else ['default'] # Default to one head if no tasks given
        self.num_heads = len(self.task_list)

        # --- Define layers locally or within Sequential --- 
        conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        bn1 = nn.BatchNorm2d(64)
        relu = nn.ReLU(inplace=True)
        # Use self._make_layer which updates self.inplanes correctly
        layer1 = self._make_layer(block, 64, layers[0])
        layer2 = self._make_layer(block, 128, layers[1], stride=2)
        layer3 = self._make_layer(block, 256, layers[2], stride=2)
        layer4 = self._make_layer(block, 512, layers[3], stride=2)
        avgpool = nn.AvgPool2d(pool_size)
        flatten = nn.Flatten(1)
        feat_dim = 512 * block.expansion
        # --- Define Shared Encoder using the layers above --- 
        self.encoder = nn.Sequential(
            conv1,
            bn1,
            relu,
            layer1,
            layer2,
            layer3,
            layer4,
            avgpool,
            flatten, # Use the flatten layer instance
            LT_DECODER(feat_dim=feat_dim)
        )
        
        # --- Define Classifier Heads --- (One per task)
        self.fc_heads = nn.ModuleDict()
         # Calculate feature dimension after encoder
        for task_name in self.task_list:
            self.fc_heads[task_name] = nn.Linear(feat_dim, num_classes)

        # Initialize weights (optional but good practice)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                     nn.init.constant_(m.bias, 0)


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        features = self.encoder(x)
        outputs = {}
        for task_name, head in self.fc_heads.items():
            outputs[task_name] = head(features)

        # Keep the logic to return single tensor if only one head, dict otherwise
        if self.num_heads == 1:
             return outputs[self.task_list[0]]
        else:
             return outputs


def resnet34(**kwargs):
    """Constructs a ResNet-34 model.
    Pass task_list=['task1', 'task2'] to create multiple heads.
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def resnet18(**kwargs):
    """ return a ResNet 18 object
    """
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet50(**kwargs):
    """ return a ResNet 50 object
    """
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


def resnet101(**kwargs):
    """ return a ResNet 101 object
    """
    return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)


def resnet152(**kwargs):
    """ return a ResNet 152 object
    """
    return ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
