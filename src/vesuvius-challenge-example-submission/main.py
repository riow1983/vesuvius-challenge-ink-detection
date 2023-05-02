import os
import gc
import glob
import json
from collections import defaultdict
import multiprocessing as mp
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple
import warnings

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import PIL.Image as Image
from sklearn.metrics import fbeta_score
from sklearn.exceptions import UndefinedMetricWarning
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as thd
from tqdm import tqdm

from dataset import SubvolumeDataset
from model import InkDetector
from utils import rle
warnings.simplefilter('ignore', UndefinedMetricWarning)

import yaml

class CFG(object):
    def __init__(self, filepath):
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        with open(filepath) as file:
            args = yaml.load(file, Loader=yaml.FullLoader) # Loader is recommended

        self.EXP = args["EXP"]
        self.TRAINING_STEPS = args["TRAINING_STEPS"]
        self.LEARNING_RATE = args["LEARNING_RATE"]
        self.TRAIN_RUN = args["TRAIN_RUN"] # To avoid re-running when saving the notebook
        self.BATCH_SIZE = args["BATCH_SIZE"]

        if type(self.EXP) == str:
            self.TRAINING_STEPS = 60

    

comp_name = "vesuvius-challenge-ink-detection"
proj_name = "vesuvius-challenge-example-submission"

KAGGLE_ENV = True if 'KAGGLE_URL_BASE' in set(os.environ.keys()) else False
if KAGGLE_ENV:
    base_path = Path(f"../input/{comp_name}/")
else:
    base_path = Path(f"/content/drive/MyDrive/colab_notebooks/kaggle/{comp_name}/input/{comp_name}/")
# base_path = Path("/kaggle/input/vesuvius-challenge/")
train_path = base_path / "train"
test_path = base_path / "test"
args = CFG(f"/content/drive/MyDrive/colab_notebooks/kaggle/{comp_name}/src/{proj_name}/config.yaml")
print("######################################################################")
print(f"############ EXP: {args.EXP}, TRAINING_STEPS: {args.TRAINING_STEPS}, DEVICE: {args.DEVICE} ############")
print("######################################################################")
if KAGGLE_ENV:
    out_path = Path("")
else:
    out_path = Path(f"/content/drive/MyDrive/colab_notebooks/kaggle/{comp_name}/output/{proj_name}/EXP_{args.EXP}/")
    os.makedirs(out_path.as_posix(), exist_ok=True)

all_fragments = sorted([f.name for f in train_path.iterdir()])
print("All fragments:", all_fragments)
# Due to limited memory on Kaggle, we can only load 1 full fragment
if type(args.EXP) == str:
    train_fragments = [train_path / fragment_name for fragment_name in ["1"]]
else:
    train_fragments = [train_path / fragment_name for fragment_name in ["1", "2", "3"]]

if type(args.EXP) == str:
    train_dset = SubvolumeDataset(fragments=train_fragments, voxel_shape=(6, 16, 16), filter_edge_pixels=True)
else:
    train_dset = SubvolumeDataset(fragments=train_fragments, voxel_shape=(48, 64, 64), filter_edge_pixels=True)
print("Num items (pixels)", len(train_dset))




train_loader = thd.DataLoader(train_dset, batch_size=args.BATCH_SIZE, shuffle=True)
print("Num batches:", len(train_loader))





model = InkDetector().to(args.DEVICE)


if __name__ == '__main__':

    if args.TRAIN_RUN:
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.SGD(model.parameters(), lr=args.LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.LEARNING_RATE, total_steps=args.TRAINING_STEPS)
        model.train()
        running_loss = 0.0
        running_accuracy = 0.0
        running_fbeta = 0.0
        denom = 0
        pbar = tqdm(enumerate(train_loader), total=args.TRAINING_STEPS)
        for i, (subvolumes, inklabels) in pbar:
            if i >= args.TRAINING_STEPS:
                break
            optimizer.zero_grad()
            outputs = model(subvolumes.to(args.DEVICE))
            loss = criterion(outputs, inklabels.to(args.DEVICE))
            loss.backward()
            optimizer.step()
            scheduler.step()
            pred_ink = outputs.detach().sigmoid().gt(0.4).cpu().int()
            accuracy = (pred_ink == inklabels).sum().float().div(inklabels.size(0))
            running_fbeta += fbeta_score(inklabels.view(-1).numpy(), pred_ink.view(-1).numpy(), beta=0.5)
            running_accuracy += accuracy.item()
            running_loss += loss.item()
            denom += 1
            pbar.set_postfix({"Loss": running_loss / denom, "Accuracy": running_accuracy / denom, "Fbeta@0.5": running_fbeta / denom})
            if (i + 1) % 500 == 0:
                running_loss = 0.
                running_accuracy = 0.
                running_fbeta = 0.
                denom = 0

        torch.save(model.state_dict(), out_path / "model.pt")

    else:
        model_weights = torch.load(out_path / "model.pt")
        model.load_state_dict(model_weights)




    # # Evaluate

    # # Clear memory before loading test fragments
    # train_dset.labels = None
    # train_dset.image_stacks = []
    # del train_loader, train_dset
    # gc.collect()

    # test_fragments = [train_path / fragment_name for fragment_name in test_path.iterdir()]
    # print("All fragments:", test_fragments)

    # pred_images = []
    # model.eval()
    # for test_fragment in test_fragments:
    #     outputs = []
    #     eval_dset = SubvolumeDataset(fragments=[test_fragment], voxel_shape=(48, 64, 64), load_inklabels=False)
    #     eval_loader = thd.DataLoader(eval_dset, batch_size=args.BATCH_SIZE, shuffle=False)
    #     with torch.no_grad():
    #         for i, (subvolumes, _) in enumerate(tqdm(eval_loader)):
    #             output = model(subvolumes.to(args.DEVICE)).view(-1).sigmoid().cpu().numpy()
    #             outputs.append(output)
    #     # we only load 1 fragment at a time
    #     image_shape = eval_dset.image_stacks[0].shape[1:]
    #     eval_dset.labels = None
    #     eval_dset.image_stacks = None
    #     del eval_loader
    #     gc.collect()

    #     pred_image = np.zeros(image_shape, dtype=np.uint8)
    #     outputs = np.concatenate(outputs)
    #     for (y, x, _), prob in zip(eval_dset.pixels[:outputs.shape[0]], outputs):
    #         pred_image[y ,x] = prob > 0.4
    #     pred_images.append(pred_image)
        
    #     eval_dset.pixels = None
    #     del eval_dset
    #     gc.collect()
    #     print("Finished", test_fragment)




    # submission = defaultdict(list)
    # for fragment_id, fragment_name in enumerate(test_fragments):
    #     submission["Id"].append(fragment_name.name)
    #     submission["Predicted"].append(rle(pred_images[fragment_id]))

    # pd.DataFrame.from_dict(submission).to_csv(out_path / "submission.csv", index=False)