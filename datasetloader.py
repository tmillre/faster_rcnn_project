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

from coco_dict import *

'''
Two helper functions for the dataset class.
'''

'''
IOU
computer intersection over union for two boxes
'''

def IOU(box1, box2):
    #find Union area
    xDiff = (min(box1[3], box2[3]) - max(box1[1], box2[1]))
    yDiff = (min(box1[2], box2[2]) - max(box1[0], box2[0]))
    if xDiff < 0 or yDiff < 0:
        return 0
    inter = xDiff * yDiff
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    return inter / (area1 + area2 - inter)

'''
boxLabels
labels anchor boxes as positive, negative, or neither based on IOU with ground truth.
'''

def boxLabels(imgSize, featureSize, gtBoxes, boxParams = None):
    featureScale = (imgSize[0] / featureSize[0]) * 0.5 + (imgSize[1] / featureSize[1]) * 0.5
    
    labels = np.zeros((featureSize[0], featureSize[1], 9))
    if boxParams == None:
        boxParams = [[8, 0.5], [8, 1], [8, 2], [16, 0.5], [16, 1], [16, 2], [32, 0.5], [32, 1], [32, 2]]
    maxes = []
    for i in range(len(gtBoxes)):
        maxes.append([-1, (0,0,0)])
    for i in range(featureSize[0]):
        for j in range(featureSize[1]):
            for k, boxP in enumerate(boxParams):
                '''
                If any bounding boxes cross img boundary, they are not used in training
                '''
                ctrX = featureScale * i + 0.5 * featureScale
                ctrY = featureScale * j + 0.5 * featureScale
                height = boxP[0] * np.sqrt(1 / boxP[1])
                width = boxP[0] * np.sqrt(boxP[1])
                box = []
                box.append(ctrY - height / 2)
                if box[-1] < 0:
                    continue
                box.append(ctrX - width / 2)
                if box[-1] < 0:
                    continue
                box.append(ctrY + height / 2)
                if box[-1] > imgSize[1]:
                    continue
                box.append(ctrX + width / 2)
                if box[-1] > imgSize[0]:
                    continue
                #print(ctrX, ctrY, width, height)
                """
                Here, we look at the IOU of bounding boxes with ground truth
                IOU > 0.7 leads to positive classification (labelled with bounding box #)
                Any 0.7 > IOU > 0.3 means we know it's not a negative.
                We also upate the largest IOU with each gt box.
                """
                skip = False
                for m, gtBox in enumerate(gtBoxes):
                    iou = IOU(box, gtBox)
                    if iou > maxes[m][0]:
                        maxes[m] = [iou, (i,j,k)]
                    if skip == False:
                        if iou > 0.7:
                            labels[i,j,k] = m + 1
                            skip = True
                        elif iou > 0.3:
                            labels[i,j,k] = -1
    '''
    If we find the max IOU is less than 0.7, we promote it to a positive.
    '''
    
    for m, maxIOU in enumerate(maxes):
        if maxIOU[0] < 0.7:
            #print(maxIOU[0])
            labels[maxIOU[1]] = m + 1
    return labels

'''
Below is our dataset class.
This loads the dataset and annotations.
'''

class DataSet:
    """
    Args:
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.

    """
    

    def __init__(self, fold="train", path=os.path.join("DataRaw","roboflow"),
                 transform=None, target_transform=None, max_img = None, boxParams = None, maxDim = 300):
        
        fold = fold.lower()

        self.train = False
        self.valid = False
        self.test = False

        if fold == "train":
            self.train = True
            inputFolder = "train"
        elif fold == "valid":
            self.valid = True
            inputFolder = "valid"
        elif fold == "test":
            self.test = True
            inputFolder = "test"
        else:
            raise RuntimeError("Not train-val-test")

        self.transform = transform
        self.target_transform = target_transform

        # now load the picked numpy arrays

        annotationPath = os.path.join(os.path.join(path, inputFolder),"_annotations.csv")
        
        annotations = pd.read_csv(annotationPath)
        
        self.annotations = []
                                   
        self.data = []
        
        fileNum = 0
        
        fileList = os.listdir(os.path.join(path, inputFolder))
        
        np.random.shuffle(fileList)
        
        for fileName in fileList:
            if fileName.split('.')[-1] not in ['png', 'jpeg', 'jpg']:
                continue
            fileInfo = {}
            
            loadImg = Image.open(os.path.join(os.path.join(path, inputFolder), fileName))
            w, h = loadImg.size
            
            scale = maxDim / max(w, h) #resize s.t. larger size is 600
            
            fileInfo["scale"] = torch.tensor(scale, dtype=torch.float32)
            
            loadImg = loadImg.resize((int(w * scale), int(h * scale)))
            
            currImgAnnotation = annotations[annotations["filename"] == fileName]
            
            tempBox = []
            tempLabel = []
            
            for index, row in currImgAnnotation.iterrows():
                holdBox = [row["xmin"] * scale, row['ymin'] * scale, row["xmax"] * scale, row["ymax"] * scale]
                if (holdBox[3] - holdBox[1]) < 1 or (holdBox[2] - holdBox[0]) < 1:
                    continue
                tempBox.append(holdBox)
                tempLabel.append(label2id[row["class"]])
                
            if len(tempBox) < 1 or len(np.asarray(loadImg).shape) < 3:
                continue
                
            self.data.append(np.asarray(loadImg))
            
            fileInfo["boxes"] = torch.as_tensor(tempBox, dtype = torch.float32)
            fileInfo["label"] = tempLabel
            fileInfo["featureSize"] = (int(h * scale / 16), int(w * scale / 16))
            if boxParams is not None:
                fileInfo["posNegTensor"] = boxLabels((int(h * scale), int(w * scale)), fileInfo["featureSize"], fileInfo["boxes"], boxParams = boxParams)
            else:
                fileInfo["posNegTensor"] = boxLabels((int(h * scale), int(w * scale)), fileInfo["featureSize"], fileInfo["boxes"])
            fileInfo["featureScale"] = torch.tensor(w*scale / fileInfo["featureSize"][1], dtype = torch.float32)
            
            self.annotations.append(fileInfo)
            
            fileNum += 1
            if max_img is not None:
                if fileNum > max_img:
                    break
            
        self.data = np.asarray(self.data)#.transpose((0, 1, 2, 3))
            
        

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, annotations).
        """
        img = np.uint8(self.data[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        target = copy.deepcopy(self.annotations[index])

        if self.transform is not None:
            #TODO: random resize crop
            
            #here, we randomly flip our bounding boxes
            '''
            if numpy.random.rand() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                print(img.size)
                for i in range(len(target["boxes"])):
                    temp = target["boxes"][i][0].clone()
                    target["boxes"][i][0] = img.size[0] - target["boxes"][i][2]
                    target["boxes"][i][2] = img.size[0] - temp
                    temp = target["boxes"][i][1].clone()
                    target["boxes"][i][1] = target["boxes"][i][3]
                    target["boxes"][i][3] = temp
            '''
                    
            img = self.transform(img)
            
                
                

        return img, target

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
