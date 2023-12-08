from os import listdir

import torch
from torch.utils import data
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np

import matplotlib.pyplot as plt

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input, dim=-1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()


class mIoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True, n_classes=2):
        super(mIoULoss, self).__init__()
        self.classes = n_classes

    def to_one_hot(self, tensor):
        n,h,w = tensor.size()
        one_hot = torch.zeros(n,self.classes,h,w).to(tensor.device).scatter_(1,tensor.view(n,1,h,w),1)
        return one_hot

    def forward(self, inputs, target):
        # inputs => N x Classes x H x W
        # target_oneHot => N x Classes x H x W

        N = inputs.size()[0]

        # predicted probabilities for each pixel along channel
        inputs = F.softmax(inputs,dim=1)

        # Numerator Product
        target_oneHot = self.to_one_hot(target)
        inter = inputs * target_oneHot
        ## Sum over all pixels N x C x H x W => N x C
        inter = inter.view(N,self.classes,-1).sum(2)

        #Denominator
        union= inputs + target_oneHot - (inputs*target_oneHot)
        ## Sum over all pixels N x C x H x W => N x C
        union = union.view(N,self.classes,-1).sum(2)

        loss = inter/union

        ## Return average loss over classes and batch
        return 1-loss.mean()
    
def seg_acc(y, predicted):
    result = (y.cpu() == torch.argmax(predicted, axis=1).cpu()).sum() / torch.numel(y.cpu())
    return result

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
            
#        x, y = x.type(self.inputs_dtype), y.type(self.targets_dtype)

        return x.type(torch.float32), y[np.newaxis, 0, :, :].type(torch.int64), input_ID, target_ID



def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(10, 2.5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()