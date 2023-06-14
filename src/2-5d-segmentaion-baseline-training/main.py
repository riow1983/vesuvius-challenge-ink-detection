#########################################
################ main.py ################
#########################################

import os
os.system("pip install segmentation_models_pytorch")
os.system("pip install warmup_scheduler")
os.system("pip install wandb")

from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, log_loss
import pickle
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import warnings
import sys
import pandas as pd
import gc
import math
import time
import random
import shutil
from pathlib import Path
from contextlib import contextmanager
from collections import defaultdict, Counter
import cv2
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from functools import partial
import argparse
import importlib
import torch
import torch.nn as nn
from torch.optim import Adam, SGD, AdamW
import datetime
import segmentation_models_pytorch as smp
import numpy as np
from torch.utils.data import DataLoader, Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations import ImageOnlyTransform
import wandb

from utils import (
    init_logger, 
    cfg_init,
    get_scheduler,
    scheduler_step,
    train_fn,
    valid_fn,
    calc_fbeta,
    calc_cv,
    send_line_notification
)
from dataset import get_train_valid_dataset, get_transforms, CustomDataset
from model import build_model
from loss import criterion
from config import CFG
from _wandb import build_wandb

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

if CFG.exp_name == "debug":
    CFG.epochs = 2

cfg_init(CFG)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
Logger = init_logger(log_file=CFG.log_path)
Logger.info('\n\n-------- exp_info -----------------')
# Logger.info(datetime.datetime.now().strftime('%Y年%m月%d日 %H:%M:%S'))

#### W&B
wandbrun = build_wandb(wandb_json_path=CFG.wandb_json_path, 
                       kaggle_env=False, 
                       dir=CFG.outputs_path, 
                       project=CFG.comp_name, 
                       name=CFG.exp_name, 
                       config=CFG, 
                       group=CFG.proj_name)
print(f"wandb run id: {wandbrun.id}")
send_line_notification(f"Training of {CFG.proj_name} has been started. \nSee {wandbrun.url}", CFG.line_json_path)
#### W&BW&B

train_images, train_masks, valid_images, valid_masks, valid_xyxys = get_train_valid_dataset()
valid_xyxys = np.stack(valid_xyxys)


train_dataset = CustomDataset(train_images, CFG, labels=train_masks, transform=get_transforms(data='train', cfg=CFG))
valid_dataset = CustomDataset(valid_images, CFG, labels=valid_masks, transform=get_transforms(data='valid', cfg=CFG))

train_loader = DataLoader(train_dataset,
                          batch_size=CFG.train_batch_size,
                          shuffle=True,
                          num_workers=CFG.num_workers, 
                          pin_memory=True, 
                          drop_last=True)

valid_loader = DataLoader(valid_dataset,
                          batch_size=CFG.valid_batch_size,
                          shuffle=False,
                          num_workers=CFG.num_workers, 
                          pin_memory=True, 
                          drop_last=False)


model = build_model(CFG)
model.to(device)

optimizer = AdamW(model.parameters(), lr=CFG.lr)
scheduler = get_scheduler(CFG, optimizer)



if __name__ == '__main__':
    fragment_id = CFG.valid_id
    valid_mask_gt = cv2.imread(CFG.comp_dataset_path + f"train/{fragment_id}/inklabels.png", 0)
    valid_mask_gt = valid_mask_gt / 255
    pad0 = (CFG.tile_size - valid_mask_gt.shape[0] % CFG.tile_size)
    pad1 = (CFG.tile_size - valid_mask_gt.shape[1] % CFG.tile_size)
    valid_mask_gt = np.pad(valid_mask_gt, [(0, pad0), (0, pad1)], constant_values=0)
    
    fold = CFG.valid_id

    if CFG.metric_direction == 'minimize':
        best_score = np.inf
    elif CFG.metric_direction == 'maximize':
        best_score = -1

    best_loss = np.inf

    for epoch in range(CFG.epochs):

        start_time = time.time()

        # train
        avg_loss = train_fn(train_loader, model, criterion, optimizer, device)

        # eval
        avg_val_loss, mask_pred = valid_fn(
            valid_loader, model, criterion, device, valid_xyxys, valid_mask_gt)

        scheduler_step(scheduler, avg_val_loss, epoch)

        best_dice, best_th = calc_cv(valid_mask_gt, mask_pred, Logger)

        # score = avg_val_loss
        score = best_dice

        elapsed = time.time() - start_time

        Logger.info(
            f'Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s')
        # Logger.info(f'Epoch {epoch+1} - avgScore: {avg_score:.4f}')
        Logger.info(
            f'Epoch {epoch+1} - avgScore: {score:.4f}')
        
        #### W&B
        metric_dict = {"Epoch": epoch+1, "avg_train_loss": avg_loss, "avg_val_loss": avg_val_loss, "avgScore": score, "time": elapsed}
        wandb.log(metric_dict)
        #### W&BW&B

        if CFG.metric_direction == 'minimize':
            update_best = score < best_score
        elif CFG.metric_direction == 'maximize':
            update_best = score > best_score

        if update_best:
            best_loss = avg_val_loss
            best_score = score

            Logger.info(
                f'Epoch {epoch+1} - Save Best Score: {best_score:.4f} Model')
            Logger.info(
                f'Epoch {epoch+1} - Save Best Loss: {best_loss:.4f} Model')
            
            torch.save({'model': model.state_dict(),
                        'preds': mask_pred},
                        CFG.model_dir + f'{CFG.model_name}_fold{fold}_best.pth')


    check_point = torch.load(CFG.model_dir + f'{CFG.model_name}_fold{fold}_{CFG.inf_weight}.pth', map_location=torch.device('cpu'))
    mask_pred = check_point['preds']
    best_dice, best_th  = calc_fbeta(valid_mask_gt, mask_pred, Logger)
    print(f"best_dice: {best_dice}; best_th: {best_th}")

    #### W&B
    send_line_notification(f"Training of {CFG.proj_name} has been done. \nbest_dice: {best_dice}; best_th: {best_th}. \nSee {wandbrun.url}", CFG.line_json_path)
    #### W&BW&B