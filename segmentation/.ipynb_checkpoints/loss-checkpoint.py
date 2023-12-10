import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average
    
    def __str__(self):
        return 'FocalLoss'
    
    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)
            input = input.transpose(1,2)
            input = input.contiguous().view(-1,input.size(2))
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


class JaccardLoss(nn.Module):
    def __init__(self, weight=None, size_average=True, n_classes=2):
        super(JaccardLoss, self).__init__()
        self.classes = n_classes
    
    def __str__(self):
        return 'JaccardLoss'
    
    def to_one_hot(self, tensor):
        n,h,w = tensor.size()
        one_hot = torch.zeros(n,self.classes,h,w).to(tensor.device).scatter_(1,tensor.view(n,1,h,w),1)
        return one_hot

    def forward(self, inputs, target):

        N = inputs.size()[0]

        inputs = F.softmax(inputs,dim=1)

        target_oneHot = self.to_one_hot(target)
        inter = inputs * target_oneHot
        inter = inter.view(N,self.classes,-1).sum(2)

        union= inputs + target_oneHot - (inputs*target_oneHot)
        union = union.view(N,self.classes,-1).sum(2)

        loss = inter/union

        return 1-loss.mean()