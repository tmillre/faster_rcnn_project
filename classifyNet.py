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


class ClassifyNetwork(nn.Module):
    def __init__(self, feature_extracting = True, IS_GPU = False, finalFeatures = 256,
                 N_CLASSES = 81, roiSize = 2):
        super(ClassifyNetwork, self).__init__()
        
        self.N_CLASSES = N_CLASSES

        self.IS_GPU = IS_GPU
        
        self.interLayer = nn.Conv2d(finalFeatures, 256, 3, stride = 1, padding = 1)
    
        self.choice = nn.Linear(finalFeatures * roiSize * roiSize, N_CLASSES + 1)
        self.bbox = nn.Linear(finalFeatures * roiSize * roiSize, 4 * (N_CLASSES + 1))
        
        self.roiSize = roiSize
    
    def forward(self, x):
        y = x#F.relu(self.interLayer(x))
        y = y.view(y.shape[0], -1)
        return self.choice(y), self.bbox(y).view(x.shape[0], self.N_CLASSES + 1, -1)
    
    '''
    compLoss: computes loss for training classification network
    inputs:
        img: the image in question
        imgFeatures: features of img from feature extractor (ResNet, VGG, etc.)
        cls, reg: inputs from RPN
        target: ground truth information
    returns:
        Loss (1x1 torch tensor)
    '''
    
    def compLoss(self, img, imgFeatures, cls, reg, target, boxParams = None,
              minPos = 0.5, posPercent = 0.5, batchSize = 128, lam = 1.0):
        minDim = 8
        pObj = torch.empty((cls.shape[0], cls.shape[1], cls.shape[2], cls.shape[3]))
        sig = torch.nn.Sigmoid()

        if boxParams == None:
            boxParams = [[8, 0.5], [8, 1.], [8, 2], [16, 0.5], [16, 1.], [16, 2.], [32, 0.5], [32, 1.], [32, 2.]]
        
        for i in range(9):
            pObj[:,i,:,:] = sig(cls[:,i, :, :].detach())
        
        if self.IS_GPU:
          pObj = pObj.cpu()
        pObj = pObj.numpy()
        #now we have a numpy array with probabilities for each anchor box...
        originalP = pObj.copy()
        
        #Below, sort first 12000 proposals (or fewer if there aren't that many)
        sortLength = min(12000, pObj.flatten().shape[0] - 1)
        
        cutoff = pObj.flatten()[np.argpartition(-1 * pObj.flatten(), sortLength)[sortLength]]
        
        #Below, sort first 12000 proposals (or fewer if there aren't that many)
        goodBox = np.where(pObj >= cutoff)
    
        fScale = img.shape[2] / cls.shape[2]
        
        predBoxes = []
        predScores = []
    
        #Below, we get our "good" boxes in standard form
        for i in range(len(goodBox[0])):
            featureCoords = [goodBox[0][i], goodBox[1][i], goodBox[2][i], goodBox[3][i]]
            
            t = reg[featureCoords[0], (4 * featureCoords[1]):(4 * featureCoords[1] + 4), featureCoords[2], featureCoords[3]].detach()
            if self.IS_GPU:
              t = t.cpu()
            anchorParams = boxParams[featureCoords[1]]
            anchorCtr = [(featureCoords[3] + 0.5) * fScale, (featureCoords[2] + 0.5) * fScale]
            anchorWH = (anchorParams[0] * np.sqrt(1/anchorParams[1]), anchorParams[0] * np.sqrt(anchorParams[1]))
        
            predCtr = [(t[0] * anchorWH[0] + anchorCtr[0]).item(), (t[1] * anchorWH[1] + anchorCtr[1]).item()]
            predWH = [(torch.exp(t[2]) * anchorWH[0]).item(), (torch.exp(t[3]) * anchorWH[1]).item()]
        
            tBox = [predCtr[0] - 0.5 * predWH[0], predCtr[1] - 0.5 * predWH[1], predCtr[0] + 0.5 * predWH[0], predCtr[1] + 0.5 * predWH[1]]
            
            if tBox[0] < 0:
                tBox[0] = 0.
            if tBox[1] < 0:
                tBox[1] = 0.
            if tBox[2] > img.shape[3]:
                tBox[2] = img.shape[3]
            if tBox[3] > img.shape[2]:
                tBox[3] = img.shape[2]
            
            if np.abs(tBox[2] - tBox[0]) > minDim and np.abs(tBox[3] - tBox[1]) > minDim:
                predBoxes.append(tBox)
                predScores.append(originalP[featureCoords[0], featureCoords[1], featureCoords[2], featureCoords[3]])
                
        
        #Below, we look at our boxes and do non-maximum suppression.
        #that is, for a given box, nearby boxes that overlap significantly
        #and have smaller score will be removed if any are present
        keep = torchvision.ops.nms(torch.tensor(predBoxes), torch.tensor(predScores), 0.7)
        
        predScores = torch.tensor(predScores)
        
        sortLength = min(2000, predScores[keep].numpy().flatten().shape[0] - 1)
        
        finalKeep = keep[np.argpartition(-1 * predScores[keep].numpy(), sortLength)[0:sortLength]]
        
        finalBoxes = []
        coords = []
            
        for i in range(finalKeep.shape[0]):
            coords.append([goodBox[0][i], goodBox[1][i], goodBox[2][i], goodBox[3][i]])
            finalBoxes.append(predBoxes[finalKeep[i]])
                
        finalBoxes = torch.tensor(finalBoxes)
        
        actualBbox = target['boxes'][0]
        
        if actualBbox.shape[0] < 1:
            print(actualBbox)
            return torch.tensor(0.0)
        '''
        hold = finalBoxes.clone()
        finalBoxes[:, 0] = hold[:,1]
        finalBoxes[:, 1] = hold[:,0]
        finalBoxes[:, 2] = hold[:,3]
        finalBoxes[:, 3] = hold[:,2]
        '''
        IOUMatrix = torchvision.ops.box_iou(actualBbox, finalBoxes).numpy()
        
        maxes = np.max(IOUMatrix, axis = 0)
        argmaxes = np.argmax(IOUMatrix, axis = 0)
        
        gtLabels = []
        for idx in argmaxes:
            gtLabels.append(idx)
            #gtLabels.append(target['label'][idx])
            
        gtLabels = np.asarray(gtLabels)
        
        highIOU = np.where(maxes > 0.7)
        highIOU = highIOU[0]
        numpy.random.shuffle(highIOU)
        lowIOU = np.where(np.logical_and(maxes < 0.3, maxes >= 0.1))
        lowIOU = lowIOU[0]
        numpy.random.shuffle(lowIOU)
        noIOU = np.where(maxes < 0.1)
        noIOU = noIOU[0]
        numpy.random.shuffle(noIOU)
        
        gtLabels[lowIOU] = -1
        gtLabels[noIOU] = -1
        
        numPos = min(highIOU.shape[0], int(posPercent * batchSize))
        numNeg = min(lowIOU.shape[0], batchSize - numPos)
        numRest = batchSize - (numPos + numNeg)
        
        roi_annotations = torch.zeros(batchSize, 5)
        finalLabels = torch.zeros(batchSize)
        finalCoords = []
        
        for i in range(numPos):
            roi_annotations[i, 1:] = finalBoxes[highIOU[i]]
            finalLabels[i] = gtLabels[highIOU[i]]
            finalCoords.append(coords[highIOU[i]])
        for i in range(numPos, numNeg + numPos):
            roi_annotations[i, 1:] = finalBoxes[lowIOU[i - numPos]]
            finalLabels[i] = gtLabels[lowIOU[i - numPos]]
            finalCoords.append(coords[lowIOU[i - numPos]])
        for i in range(numPos + numNeg, numNeg + numPos + numRest):
            roi_annotations[i, 1:] = finalBoxes[noIOU[i - numPos - numNeg]]
            finalLabels[i] = gtLabels[noIOU[i - numPos - numNeg]]
            finalCoords.append(coords[noIOU[i - numPos - numNeg]])
            
        '''
        It turns out our boxes have their coordinates flipped. We fix that here
        '''
        
        hold = roi_annotations.clone()
        roi_annotations[:, 1] = hold[:,2]
        roi_annotations[:, 2] = hold[:,1]
        roi_annotations[:, 3] = hold[:,4]
        roi_annotations[:, 4] = hold[:,3]
        
        
        output_size = self.roiSize
        
        if self.IS_GPU:
          roi_pooled = torchvision.ops.roi_pool(imgFeatures, roi_annotations.cuda(), output_size = output_size, spatial_scale = 1/fScale)
        else:
          roi_pooled = torchvision.ops.roi_pool(imgFeatures, roi_annotations, output_size = output_size, spatial_scale = 1/fScale)
        roi_pooled = torch.clamp(roi_pooled, min = -1000, max = 1000)
        #clsLoss = torch.nn.functional.cross_entropy()
        regLoss = torch.nn.SmoothL1Loss()
        
        #implement loss and backwards step!
        totalRegLoss = 0.
        totalClsLoss = 0.
        
        catch, bboxes = self.forward(roi_pooled)
        
        j = 0
        for i in range(len(finalCoords)):
            #get anchor params...
            if finalLabels[i] >= 0:
                targetLabel = target['label'][int(finalLabels[i].item())]
                
                anchorCtr = [(roi_annotations[i][1] + roi_annotations[i][3] ) / 2.0,
                             (roi_annotations[i][2] + roi_annotations[i][4] ) / 2.0]
                anchorWH = [(roi_annotations[i][3] - roi_annotations[i][1] ) / 2.0,
                             (roi_annotations[i][4] - roi_annotations[i][2] ) / 2.0]
                
                #get actual ground truth params
                actualBbox = target['boxes'][0][int(finalLabels[i].item())]
                actualCtr = [(actualBbox[0] + actualBbox[2]) / 2.0, (actualBbox[1] + actualBbox[3]) / 2.0]
                actualWH = [(actualBbox[2] - actualBbox[0]), (actualBbox[3] - actualBbox[1])]
                #print('BOX')
                #print(roi_annotations[i])
                #print(actualBbox)
                #print(targetLabel)
        
                if actualWH[0] < 1 or actualWH[1] < 1:
                    continue
                
                if anchorWH[0] < 1 or anchorWH[1] < 1:
                    continue
            
                #calculate t parameters for box
                '''
                Note we flipped the coordinates!
                '''
                t = torch.tensor([(actualCtr[0] - anchorCtr[1]) / anchorWH[1], (actualCtr[1] - anchorCtr[0]) / anchorWH[0],
                                  np.log(actualWH[0] / anchorWH[1]), np.log(actualWH[1] / anchorWH[0])])
            
                #compute loss for system
                #totalRegLoss += regLoss(t, bboxes[i, targetLabel, :].squeeze(0))
                #totalClsLoss += ((batchSize - numPos) / numPos) * torch.nn.functional.cross_entropy(catch[i, :].unsqueeze(0), targetLabel)
                if self.IS_GPU:
                  totalClsLoss += torch.nn.functional.cross_entropy(catch[i, :].unsqueeze(0), targetLabel.cuda())
                else:
                  totalClsLoss += torch.nn.functional.cross_entropy(catch[i, :].unsqueeze(0), targetLabel)
            else:
                if j > numPos:
                    continue
                j += 1
                if self.IS_GPU:
                  totalClsLoss += torch.nn.functional.cross_entropy(catch[i, :].unsqueeze(0), torch.tensor([self.N_CLASSES]).cuda())
                else:
                  totalClsLoss += torch.nn.functional.cross_entropy(catch[i, :].unsqueeze(0), torch.tensor([self.N_CLASSES]))
        return totalClsLoss + lam * totalRegLoss
        
        
    '''
    predict -- predicts bounding boxes and labels based on RoI
    inputs:
        img -- the original image
        imgFeatures -- the feature map (ResNet, VGG, etc. output for img)
        cls, reg -- the outputs of the RPN
    output:
        None (shows boxes on img)
    '''
        
    
    def predict(self, img, imgFeatures, cls, reg, cutoff = 0.90, boxParams = None,
                minDim = 8, dummyLabel = False, secondLabel = False):
        pObj = torch.empty((cls.shape[0], cls.shape[1], cls.shape[2], cls.shape[3]))
        sig = torch.nn.Sigmoid()

        if boxParams == None:
            boxParams = [[8, 0.5], [8, 1.], [8, 2], [16, 0.5], [16, 1.], [16, 2.], [32, 0.5], [32, 1.], [32, 2.]]
        
        for i in range(9):
            pObj[:,i,:,:] = sig(cls[:,i, :, :].detach())
            
        pObj = pObj.numpy()
        #now we have a numpy array with probabilities for each anchor box...
        originalP = pObj.copy()
    
        goodBox = np.where(pObj >= cutoff)
        
        #below, we make a map to show where objects are
        pObj[goodBox] = 1000
        pObj = np.sum(pObj, axis = 1)
        goodArea = np.where(pObj >= 1000)
        badArea = np.where(pObj < 1000)
        pObj[goodArea] = 1
        pObj[badArea] = 0
    
        #show objects of feature scale!
        plt.imshow(pObj[0,:,:])
        plt.show()
    
        fScale = img.shape[2] / cls.shape[2]
    
        predAnnotation = {}
        predBoxes = []
        predLabs = []
        predScores = []
    
        #Below, we get our "good" boxes in standard form
        for i in range(len(goodBox[0])):
            featureCoords = [goodBox[0][i], goodBox[1][i], goodBox[2][i], goodBox[3][i]]
            #print(featureCoords)
            t = reg[featureCoords[0], (4 * featureCoords[1]):(4 * featureCoords[1] + 4), featureCoords[2], featureCoords[3]].detach()

            if self.IS_GPU:
              t = t.cpu()
            anchorParams = boxParams[featureCoords[1]]
            anchorCtr = [(featureCoords[3] + 0.5) * fScale, (featureCoords[2] + 0.5) * fScale]
            anchorWH = (anchorParams[0] * np.sqrt(1/anchorParams[1]), anchorParams[0] * np.sqrt(anchorParams[1]))
        
            predCtr = [(t[0] * anchorWH[0] + anchorCtr[0]).item(), (t[1] * anchorWH[1] + anchorCtr[1]).item()]
            predWH = [(torch.exp(t[2]) * anchorWH[0]).item(), (torch.exp(t[3]) * anchorWH[1]).item()]
        
            tBox = [predCtr[0] - 0.5 * predWH[0], predCtr[1] - 0.5 * predWH[1], predCtr[0] + 0.5 * predWH[0], predCtr[1] + 0.5 * predWH[1]]
            
            if tBox[0] < 0:
                tBox[0] = 0.
            if tBox[1] < 0:
                tBox[1] = 0.
            if tBox[2] > img.shape[3]:
                tBox[2] = img.shape[3]
            if tBox[3] > img.shape[2]:
                tBox[3] = img.shape[2]
            
            if np.abs(tBox[2] - tBox[0]) > minDim and np.abs(tBox[3] - tBox[1]) > minDim:
                predBoxes.append(tBox)
                predScores.append(originalP[featureCoords[0], featureCoords[1], featureCoords[2], featureCoords[3]])

        #Below, we look at our boxes and do non-maximum suppression.
        #that is, for a given box, nearby boxes that overlap significantly
        #and have smaller score will be removed if any are present
        if len(predBoxes) > 0:
            keep = torchvision.ops.nms(torch.tensor(predBoxes), torch.tensor(predScores), 0.7)
        
            finalBoxes = []
            predLabs = []
            
            for i in range(keep.shape[0]):
                finalBoxes.append(predBoxes[keep[i]])
                predLabs.append(0)
                
            predAnnotation['boxes'] = torch.tensor([finalBoxes])
            
            output_size = self.roiSize
            roi_annotations = torch.zeros(predAnnotation['boxes'][0,:,:].shape[0], 5)
            roi_annotations[:,1:] = predAnnotation['boxes'][0,:,:]
            
            '''
            It turns out our boxes have their coordinates flipped. We fix that here
            '''
            
            hold = roi_annotations.clone()
            roi_annotations[:, 1] = hold[:,2]
            roi_annotations[:, 2] = hold[:,1]
            roi_annotations[:, 3] = hold[:,4]
            roi_annotations[:, 4] = hold[:,3]
            
            if self.IS_GPU:
              roi_pooled = torchvision.ops.roi_pool(imgFeatures, roi_annotations.cuda(), output_size = output_size, spatial_scale = 1/fScale)
            else:
              roi_pooled = torchvision.ops.roi_pool(imgFeatures, roi_annotations, output_size = output_size, spatial_scale = 1/fScale)
            
            #roi_pooled = roi_pool(imgFeatures, roi_annotations)
            #print(roi_annotations)
            #print(roi_pooled[0,0,:,:])
            #print(roi_pooled[1,0,:,:])
            roi_pooled = torch.clamp(roi_pooled, min = -100, max = 100)
            #print(roi_pooled)
            
            #TODO: modify for double classification
            catch, bboxes = self.forward(roi_pooled)
            if self.IS_GPU:
              catch = catch.cpu()
              bboxes = bboxes.cpu()
            #print(catch)
            
            if dummyLabel:
                predLabs = torch.tensor(predLabs).unsqueeze(1)
            elif secondLabel:
                predLabs = torch.argmax(catch, dim = 1).unsqueeze(1)
                for i, lab in enumerate(predLabs):
                    catch[i, lab] = -10000
                print(catch.shape)
                predLabs = torch.argmax(catch, dim = 1).unsqueeze(1)
            else:
                predLabs = torch.argmax(catch, dim = 1).unsqueeze(1)
                
            
            predAnnotation['label'] = predLabs
            
            showBoxes(img, predAnnotation)
            