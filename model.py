"""Headers"""

from __future__ import print_function
from PIL import Image
import os
import os.path
import numpy as np
import sys
import math
import time
import pickle
import numpy.random

import torch.utils.data as data
from torchvision.datasets.utils import download_url, check_integrity

import csv
import copy

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os.path
import sys
import torch
import torch.utils.data
import torchvision
import torchvision.transforms as transforms

import cv2

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F

import numpy.random

from coco_dict import *
from datasetloader import *
from boxplot import *
from classifyNet import *


class PreTrainedResNet(nn.Module):
    def __init__(self, feature_extracting = True, model = 'resnet18', IS_GPU = False):
        super(PreTrainedResNet, self).__init__()

        self.IS_GPU = IS_GPU
        #TODO1: Load pre-trained ResNet Model
        if model == 'resnet50':
            self.resnet = models.resnet50(pretrained=True)
            #Set gradients to false
            modules=list(self.resnet.children())[:-3]
            self.shortResnet=nn.Sequential(*modules)
            finalFeatures = 1024
        
        elif model == 'resnet101':
            self.resnet = models.resnet101(pretrained=True)
            #Set gradients to false
            modules=list(self.resnet.children())[:-3]
            self.shortResnet=nn.Sequential(*modules)
            finalFeatures = 1024
        
        elif model == 'resnet18':
            self.resnet = models.resnet18(pretrained=True)
            #Set gradients to false
            modules=list(self.resnet.children())[:-3]
            self.shortResnet=nn.Sequential(*modules)
            finalFeatures = 256
        
        else:
            print("Oops, couldn't find that model!")

        if feature_extracting:
            for param in self.shortResnet.parameters():
                param.requires_grad = False

        self.clNet = ClassifyNetwork(finalFeatures = finalFeatures, N_CLASSES = len(list(id2Label.keys())))
    
        self.interLayer = nn.Conv2d(finalFeatures, 256, 3, stride = 1, padding = 1)
    
        self.normResNet = nn.BatchNorm2d(finalFeatures)
        self.normInter = nn.BatchNorm2d(256)
    
        self.reg = nn.Conv2d(256, 4 * 9, 1, stride = 1)
        self.cls_logit = nn.Conv2d(256, 9, 1, stride = 1)
    
        torch.nn.init.normal_(self.normInter.weight.data, 0.0, 0.01)
        torch.nn.init.normal_(self.reg.weight.data, 0.0, 0.01)
        torch.nn.init.normal_(self.cls_logit.weight.data, 0.0, 0.01)
    
        torch.nn.init.constant_(self.normInter.bias.data, 0.0)
        torch.nn.init.constant_(self.reg.bias.data, 0.0)
        torch.nn.init.constant_(self.cls_logit.bias.data, 0.0)
    
        #Losses
        self.valLoss = []
        self.trainLoss = []
        self.fastLoss = []
        self.rpnLoss = []
    
    def forward(self, x):
        #TODO3: Forward pass x through the model
        x = self.shortResnet.forward(x)
        x = F.relu(self.interLayer(x))
        cls_logit = self.cls_logit(x)
        reg = self.reg(x)
        return cls_logit, reg
    
    
    '''
    save:
        saves parameters to a file
        dirName: directory to save parameters in
    '''
    def save(self, dirName):
        if not os.path.isdir(dirName):
            os.mkdir(dirName)
        torch.save(self.state_dict(), dirName + '/RPN.pt')
        torch.save(self.clNet.state_dict(), dirName + '/classifyNet.pt')
    
    '''
    load:
        loads parameters from file.
        dirName: directory to load parameters from
    '''
    def load(self, dirName):
        try:
            self.load_state_dict(torch.load(dirName + '/RPN.pt'))
            self.clNet.load_state_dict(torch.load(dirName + '/classifyNet.pt'))
        except:
            print('Error loading file')
    
    

    '''
    predict
    predicts bounding boxes and classifies the contents
    
    intput:
        img = image as (1, C, H, W) tensor
        cutoff = the cls score cutoff to use for defining an object.
    output:
        None (displays an image. Can modify this...)
    '''
    def predict(self, img, cutoff = 0.90, boxParams = None, dummyLabel = False):
        input_features = self.shortResnet.forward(img)
        cls, reg = self.forward(img)
        self.clNet.predict(img, input_features, cls, reg, cutoff = cutoff, boxParams = boxParams, dummyLabel = dummyLabel)
        
    
    '''
    compRPNLoss
    computes the loss for the RPN
    Inputs: 
      inputs = input image as (1, C, H, W) tensor
      target = annotation dictionary containing ground truth info
    Outputs:
      torch tensor for the loss.
    '''
    def compRPNLoss(self, inputs, target, boxParams, lam = 10.0):
        
        totalRegLoss = 0.0
        totalClsLoss = 0.0
    
        clsLoss = torch.nn.BCEWithLogitsLoss()
        regLoss = torch.nn.SmoothL1Loss()
    
        cls, reg = self.forward(inputs)
        
        l = np.argwhere(target['posNegTensor'].numpy() > 0)
        np.random.shuffle(l)
        for i, dex in enumerate(l):
            #get anchor params...
            bboxIndex = target['posNegTensor'].numpy()[dex[0], dex[1], dex[2], dex[3]]
            anchorParams = boxParams[dex[3]]
            
            anchorCtr = [(dex[2] + 0.5) * target['featureScale'], (dex[1] + 0.5) * target['featureScale']]
            anchorWH = (anchorParams[0] * np.sqrt(1/anchorParams[1]), anchorParams[0] * np.sqrt(anchorParams[1]))
            
            #get actual ground truth params
            actualBbox = target['boxes'][0][int(bboxIndex) - 1]
            actualCtr = [(actualBbox[0] + actualBbox[2]) / 2.0, (actualBbox[1] + actualBbox[3]) / 2.0]
            actualWH = [(actualBbox[2] - actualBbox[0]), (actualBbox[3] - actualBbox[1])]
        
            if actualWH[0] < 1 or actualWH[1] < 1:
                continue
            
            #calculate t parameters for box
            t = torch.tensor([(actualCtr[0] - anchorCtr[0]) / anchorWH[0], (actualCtr[1] - anchorCtr[1]) / anchorWH[1],
                np.log(actualWH[0] / anchorWH[0]), np.log(actualWH[1] / anchorWH[1])])
            
            #compute loss for system
            totalRegLoss += regLoss(t, reg[dex[0], (dex[3] * 4): (dex[3] * 4 + 4), dex[1], dex[2]])
            totalClsLoss += clsLoss(cls[dex[0], dex[3], dex[1], dex[2]], torch.tensor(1.))
            if i > 128:
                break
        
        i = len(l)
        
        l = np.argwhere(target['posNegTensor'].numpy() == 0)
        np.random.shuffle(l)
        l = l[0:(256 - i)]
        for j, dex in enumerate(l):
            totalClsLoss += clsLoss(cls[dex[0],dex[3], dex[1], dex[2]], torch.tensor(0.))
            
        
        return totalClsLoss / 256 + lam * totalRegLoss / (target["featureSize"][0] * target["featureSize"][1])
    
    '''
      train
      Inputs: 
          trainLoader = dataLoader for training set data
          valLoader = dataLoader for validation set data
      Outputs:
          None
    '''
    def train(self, trainLoader, valLoader, box_params, lamb = 10.,
              class_start = 26, NUM_EPOCHS = 20, learnRate = 0.003):
        optimizer = torch.optim.Adam(self.parameters(), lr=learnRate)
        fastRCNN_opt = torch.optim.Adam(self.clNet.parameters(), lr=learnRate)
        
        for n in range(NUM_EPOCHS):
            st_time = time.time()
            epochValLoss = 0.0
            epochTrainLoss = 0.0
            epochFastLoss = 0.0
            epochRPNLoss = 0.0
            totalLoss = 0.0
            
            #computer training losses...
            #and backward propegate
        
            for k, data in enumerate(trainLoader, 0):
                if self.IS_GPU:
                    inputs = Variable(data[0].cuda())
                    target = data[1]
                else:
                    inputs = Variable(data[0])
                    target = data[1]
                
                totalRegLoss = 0
                totalClsLoss = 0
        
                optimizer.zero_grad()
        
                totalLoss = self.compRPNLoss(inputs, target, box_params, lam = lamb)
        
                epochTrainLoss += totalLoss.item()
        
                totalLoss.backward()
                optimizer.step()
                
                if len(self.valLoss) > class_start:
                    input_features = self.shortResnet.forward(inputs)
                    cls, reg = self.forward(inputs)
            
                    fastRCNN_opt.zero_grad()
                    fastLoss = self.clNet.compLoss(inputs, input_features, cls,
                                               reg, target, boxParams = box_params)
                    fastLoss.backward()
                    fastRCNN_opt.step()
                else:
                    fastLoss = torch.tensor([0.0])
                
                epochTrainLoss += fastLoss.item()
                epochFastLoss += fastLoss.item()
                epochRPNLoss += totalLoss.item()
                
            #compute performance on validation set
            
            for k, data in enumerate(valLoader, 0):
                if self.IS_GPU:
                    inputs = Variable(data[0].cuda())
                    target = data[1]
                else:
                    inputs = Variable(data[0])
                    target = data[1]
                
                totalRegLoss = 0
                totalClsLoss = 0
        
                epochValLoss += self.compRPNLoss(inputs, target, box_params, lam = lamb).item()
                
                if len(self.valLoss) > class_start:
                    input_features = self.shortResnet.forward(inputs)
                    cls, reg = self.forward(inputs)
                
                    fastLoss = self.clNet.compLoss(inputs, input_features, cls, reg, target).item()
                else:
                    fastLoss = torch.tensor([0.0])
                
                epochValLoss += fastLoss
            
                
            #Here, we need to get proposals from RPN, non-max suppress, ROI pool, and feed into classifier...
            #See predict method here for idea on how to do that
    
            self.valLoss.append(epochValLoss)
            if len(self.valLoss) > class_start:
                self.fastLoss.append(epochFastLoss)
            self.trainLoss.append(epochTrainLoss)
            self.rpnLoss.append(epochRPNLoss)
            print('Epoch ', n,' : Time required -> ', time.time() - st_time)
