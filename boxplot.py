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
showBoxes

Will display ground truth boxes over image
'''

def showBoxes(img, annotations):
    asNp = img.numpy()
    fig, ax = plt.subplots(dpi=300)
    asNp = np.float32(np.swapaxes(np.swapaxes(asNp[0,:,:,:], 0, 2), 0, 1)) * 0.5 + 0.5
    asNp = cv2.cvtColor(asNp, cv2.COLOR_BGR2BGRA)
    for i in range(len(annotations["boxes"][0])):
        box = annotations["boxes"][0][i]
        start_point = (int(box[0]), int(box[1]))
        end_point = (int(box[2]), int(box[3]))
        color = (1,0,0)
        thickness = 1
        fontScale = 0.2
        try:
            asNp = cv2.rectangle(asNp, start_point, end_point, color, thickness)
            asNp = cv2.putText(asNp, id2Label[annotations["label"][i][0].item()], start_point , cv2.FONT_HERSHEY_SIMPLEX, fontScale, color , thickness,  cv2.LINE_AA)
        except TypeError:
            print(start_point, end_point)
    plt.imshow(asNp)

'''
showAllBoxes

For debugging, will plot all anchor boxes over the image
'''
def showAllBoxes(img, featureScale = 15.6):
    asNp = img.numpy()
    imgSize = img[0,0,:,:].size()
    fig, ax = plt.subplots(dpi=300)
    asNp = np.float32(np.swapaxes(np.swapaxes(asNp[0,:,:,:], 0, 2), 0, 1)) * 0.5 + 0.5
    asNp = cv2.cvtColor(asNp, cv2.COLOR_BGR2BGRA)
    
    featureSize = (math.ceil(imgSize[0] / 16), math.ceil(imgSize[1] / 16))
    
    #boxParams = [[8, 0.5], [8, 1], [8, 2], [16, 0.5], [16, 1], [16, 2], [32, 0.5], [32, 1], [32, 2]]
    boxParams = [[8, 0.5]]
    #boxParams = [[64, 0.5], [64, 1], [64, 2], [16, 0.5], [16, 1], [16, 2], [32, 0.5], [32, 1], [32, 2]]
    #featureSize = (38, 18)
    for i in range(featureSize[1]):
        for j in range(featureSize[0]):
            for k, boxP in enumerate(boxParams):
                '''
                If any bounding boxes cross img boundary, they are not used in training
                '''
                ctrX = featureScale * i + 0.5 * featureScale
                ctrY = featureScale * j + 0.5 * featureScale
                height = boxP[0] * np.sqrt(1 / boxP[1])
                width = boxP[0] * np.sqrt(boxP[1])
                box = []
                box.append(int(ctrX - width / 2))
                if box[-1] < 0:
                    print(box[-1])
                    continue
                box.append(int(ctrY - height / 2))
                if box[-1] < 0:
                    print(box[-1])
                    continue
                box.append(int(ctrX + width / 2))
                if box[-1] > imgSize[1]:
                    print(box[-1])
                    continue
                box.append(int(ctrY + height / 2))
                if box[-1] > imgSize[0]:
                    print(box[-1])
                    continue
                color = (1,0,0)
                thickness = 1
                asNp = cv2.rectangle(asNp, (box[0], box[1]), (box[2], box[3]), color, thickness)
    plt.imshow(asNp)
