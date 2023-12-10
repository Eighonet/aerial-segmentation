import torch

def s_acc_score(y, predicted):
    mask = torch.argmax(predicted, axis=1).cpu()
    result = (y.cpu() == mask).sum() / torch.numel(y.cpu())
    return result

def s_precision_score(y, predicted):
    mask = torch.argmax(predicted, axis=1).cpu()
    tp = ((y.cpu() == 1) & (mask == 1)).sum()
    fp = ((y.cpu() == 0) & (mask == 1)).sum()
    result = tp/(tp+fp)
    return result

def s_recall_score(y, predicted):
    mask = torch.argmax(predicted, axis=1).cpu()
    tp = ((y.cpu() == 1) & (mask == 1)).sum()
    fn = ((y.cpu() == 1) & (mask == 0)).sum()
    result = tp/(tp+fn)
    return result

def dice_score(y, predicted):
    p = s_precision_score(y, predicted)
    r = s_recall_score(y, predicted)
    result = (2*p*r)/(p+r)
    return result