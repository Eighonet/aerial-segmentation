import os
from os import listdir
from tqdm import tqdm

import torch
from torch.utils import data
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
import mlflow

import numpy as np

import matplotlib.pyplot as plt

from .loss import FocalLoss, JaccardLoss
from .metric import s_acc_score, s_precision_score, s_recall_score, dice_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

focal_criterion = FocalLoss(gamma=1.75).to(device)
jaccard_criterion = JaccardLoss(n_classes=2).to(device)    

def train(run_name,
          model,
          optimizer,
          lr_scheduler,
          criterion,
          train_dataloader,
          val_dataloader,
          epochs,
          device):
        
    os.makedirs('./saved_models', exist_ok=True)
    min_loss = torch.tensor(float('inf'))
    best_model = None
    scheduler_counter = 0

    for epoch in tqdm(range(epochs)):
        # ======= TRAIN PART ==========
        model.train()

        jac_train, dice_train, pre_train, rec_train, acc_train, loss_train = [], [], [], [], [], []

        for batch_i, (x, y, _, _) in enumerate(train_dataloader):
            optimizer.zero_grad()

            y = y[:, 0, :]
            pred_mask = model(x.to(device))

#            print(x.shape, pred_mask.shape, y.shape)

            loss = criterion(pred_mask, y.to(device))
            jac = 1 - jaccard_criterion(pred_mask, y.to(device))
            acc = s_acc_score(y, pred_mask)
            pre = s_precision_score(y, pred_mask)
            rec = s_recall_score(y, pred_mask)
            dice = dice_score(y, pred_mask)
            
            loss.backward()
            optimizer.step()

            loss_train.append(loss.cpu().detach().numpy())
            jac_train.append(jac.cpu().detach().numpy())
            acc_train.append(acc.numpy())
            pre_train.append(pre.numpy())
            rec_train.append(rec.numpy())
            dice_train.append(dice.numpy())
                

        mlflow.log_metrics(
            {
                'train/loss':np.mean(loss_train),
                'train/jaccard':np.mean(jac_train),
                'train/acc':np.mean(acc_train),
                'train/precision':np.mean(pre_train),
                'train/recall':np.mean(rec_train),
                'train/dice':np.mean(dice_train)
            },
            step=epoch)

        scheduler_counter += 1

        # ======= VALIDATION PART ==========
        model.eval()
        jac_val, dice_val, pre_val, rec_val, acc_val, loss_val = [], [], [], [], [], []
        for batch_i, (x, y, _, _) in enumerate(val_dataloader):

            y = y[:, 0, :]

            with torch.no_grad():
                pred_mask = model(x.to(device))

            loss = criterion(pred_mask, y.to(device))
            jac = 1 - jaccard_criterion(pred_mask, y.to(device))
            acc = s_acc_score(y, pred_mask)
            pre = s_precision_score(y, pred_mask)
            rec = s_recall_score(y, pred_mask)
            dice = dice_score(y, pred_mask)

            loss_val.append(loss.cpu().detach().numpy())
            jac_val.append(jac.cpu().detach().numpy())
            acc_val.append(acc.numpy())
            pre_val.append(pre.numpy())
            rec_val.append(rec.numpy())
            dice_val.append(dice.numpy())


        mlflow.log_metrics(
            {
                'val/loss':np.mean(loss_val),
                'val/jaccard':np.mean(jac_val),
                'val/acc':np.mean(acc_val),
                'val/precision':np.mean(pre_val),
                'val/recall':np.mean(rec_val),
                'val/dice':np.mean(dice_val)
            }, step=epoch)

        compare_loss = np.mean(loss_val)
        if compare_loss < min_loss:
            scheduler_counter = 0
            torch.save(model.state_dict(), f'./saved_models/{run_name}_best.pt')
            best_model = model

        min_loss = min(compare_loss, min_loss)  
        mlflow.log_metrics({'val/min_loss':min_loss}, step=epoch)

        if scheduler_counter > 5:
            lr_scheduler.step()
            print(f"lowering learning rate to {optimizer.param_groups[0]['lr']}")
            scheduler_counter = 0
        # dynamic params cannot be logged in mlflow, what a shame ¯\_(ツ)_/¯
        mlflow.log_metrics({'lr':optimizer.param_groups[0]['lr']}, step=epoch)
        
    return best_model

def test(model,
         test_dataloader,
         device):
    
    model.eval()
    
    jac_test, dice_test, pre_test, rec_test, acc_test = [], [], [], [], []
    for batch_i, (x, y, _, _) in enumerate(test_dataloader):

        y = y[:, 0, :]

        with torch.no_grad():
            pred_mask = model(x.to(device))
        
        jac = 1 - jaccard_criterion(pred_mask, y.to(device))
        acc = s_acc_score(y, pred_mask)
        pre = s_precision_score(y, pred_mask)
        rec = s_recall_score(y, pred_mask)
        dice = dice_score(y, pred_mask)

        jac_test.append(jac.cpu().detach().numpy())
        acc_test.append(acc.numpy())
        pre_test.append(pre.numpy())
        rec_test.append(rec.numpy())
        dice_test.append(dice.numpy())
    
    jac_mean = np.mean(jac_test) 
    acc_mean = np.mean(acc_test)
    pre_mean = np.nanmean(pre_test)
    rec_mean = np.nanmean(rec_test)
    dice_mean = np.nanmean(dice_test)

    mlflow.log_metrics(
        {
            'test/jaccard':jac_mean,
            'test/acc':acc_mean,
            'test/precision':pre_mean,
            'test/recall':rec_mean,
            'test/dice':dice_mean
        }, 
        step=0)
    
    return jac_mean, acc_mean, pre_mean, rec_mean, dice_mean

def visualize(**images):
    n = len(images)
    plt.figure(figsize=(10, 2.5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()