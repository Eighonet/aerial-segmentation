import os
from os import listdir
from tqdm import tqdm

import numpy as np

import torch
from torch.utils import data
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

class SegDataset(data.Dataset):
    def __init__(self,
                 inputs_path: list,
                 targets_path: list,
                 transform_input=None,
                 transform_mask=None,
                 ):
        self.inputs = [inputs_path+f for f in listdir(inputs_path) if f.split('.')[-1] == 'png']
        self.targets = [targets_path+f for f in listdir(targets_path) if f.split('.')[-1] == 'png']
        self.transform_input = transform_input
        self.transform_mask = transform_mask
        
    def __len__(self):
        return len(self.inputs)

    def __getitem__(self,
                    index: int):
        input_ID = self.inputs[index]
        target_ID = self.targets[index]
        
        x, y = plt.imread(input_ID), plt.imread(target_ID)
        
        if self.transform_input is not None:
            x = self.transform_input(x)

        if self.transform_mask is not None:
            y = self.transform_mask(y)

        return x.type(torch.float32), y[np.newaxis, 0, :, :].type(torch.int64), input_ID, target_ID