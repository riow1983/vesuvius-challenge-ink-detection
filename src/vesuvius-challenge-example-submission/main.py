#########################################
################ main.py ################
#########################################
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
from utils import rle, send_line_notification
warnings.simplefilter('ignore', UndefinedMetricWarning)

import yaml
import shutil

class CFG(object):
    def __init__(self, filepath):

        with open(filepath) as file:
            args = yaml.load(file, Loader=yaml.FullLoader) # Loader is recommended

        self.EXP = args["EXP"]
        self.TRAINING_STEPS = args["TRAINING_STEPS"]
        self.LEARNING_RATE = args["LEARNING_RATE"]
        self.TRAIN_RUN = args["TRAIN_RUN"] # To avoid re-running when saving the notebook
        self.BATCH_SIZE = args["BATCH_SIZE"]
        self.VOXEL_SHAPE = (args["VOXEL_SHAPE"][0], args["VOXEL_SHAPE"][1], args["VOXEL_SHAPE"][2])
        self.WANDB = args["WANDB"]
        self.THRES = args["THRES"]

        if type(self.EXP) == str:
            self.TRAINING_STEPS = 1500

    

comp_name = "vesuvius-challenge-ink-detection"
proj_name = "vesuvius-challenge-example-submission"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

KAGGLE_ENV = True if 'KAGGLE_URL_BASE' in set(os.environ.keys()) else False
if KAGGLE_ENV:
    base_path = Path(f"/kaggle/input/{comp_name}/")
else:
    base_path = Path(f"/content/drive/MyDrive/colab_notebooks/kaggle/{comp_name}/input/{comp_name}/")
    json_path = Path("/content/drive/MyDrive/colab_notebooks/kaggle/")
# base_path = Path("/kaggle/input/vesuvius-challenge/")
train_path = base_path / "train"
test_path = base_path / "test"
args = CFG(f"/content/drive/MyDrive/colab_notebooks/kaggle/{comp_name}/src/{proj_name}/config.yaml")
print("#################################################################################################################")
print(f"############ EXP: {args.EXP}, TRAIN_RUN: {args.TRAIN_RUN}, TRAINING_STEPS: {args.TRAINING_STEPS}, DEVICE: {DEVICE}, VOXEL_SHAPE: {args.VOXEL_SHAPE} ############")
print("#################################################################################################################")
if KAGGLE_ENV:
    out_path = Path("/kaggle/working/")
else:
    out_path = Path(f"/content/drive/MyDrive/colab_notebooks/kaggle/{comp_name}/output/{proj_name}/EXP-{args.EXP}-{proj_name}/")
    os.makedirs(out_path.as_posix(), exist_ok=True)

if args.WANDB:
    os.system("pip install wandb")
    import wandb
    from _wandb import build_wandb
    wandbrun = build_wandb(wandb_json_path=json_path / "wandb.json", kaggle_env=KAGGLE_ENV, 
                           dir=out_path, project=comp_name, name=f"EXP-{args.EXP}-{proj_name}", config=args, group=proj_name) # build_wandb(wandb_json_path, kaggle_env, dir, project, name, config, group)
    print(f"wandb run id: {wandbrun.id}")


all_fragments = sorted([f.name for f in train_path.iterdir()])
print("All fragments:", all_fragments)

if args.TRAIN_RUN:
    # Due to limited memory on Kaggle, we can only load 1 full fragment
    if type(args.EXP) == str:
        train_fragments = [train_path / fragment_name for fragment_name in ["1"]]
    else:
        train_fragments = [train_path / fragment_name for fragment_name in ["1", "2", "3"]]
    
    if type(args.EXP) == str:
        train_dset = SubvolumeDataset(fragments=train_fragments, voxel_shape=(6, 16, 16), filter_edge_pixels=True)
    else:
        train_dset = SubvolumeDataset(fragments=train_fragments, voxel_shape=args.VOXEL_SHAPE, filter_edge_pixels=True)
    print("Num items (pixels)", len(train_dset))
    
    train_loader = thd.DataLoader(train_dset, batch_size=args.BATCH_SIZE, shuffle=True)
    print("Num batches:", len(train_loader))


# Play ground
# inputs, classes = next(iter(train_loader))
# print("inputs.shape: ", inputs.shape)
# print("classes.shape: ", classes.shape)
# print("inputs:\n\n", inputs)
# print("classes:\n\n", classes)
# raise ValueError

model = InkDetector().to(DEVICE)


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
            outputs = model(subvolumes.to(DEVICE))
            loss = criterion(outputs, inklabels.to(DEVICE))
            loss.backward()
            optimizer.step()
            scheduler.step()
            pred_ink = outputs.detach().sigmoid().gt(0.4).cpu().int()
            accuracy = (pred_ink == inklabels).sum().float().div(inklabels.size(0))
            running_fbeta += fbeta_score(inklabels.view(-1).numpy(), pred_ink.view(-1).numpy(), beta=0.5)
            running_accuracy += accuracy.item()
            running_loss += loss.item()
            denom += 1
            metric_dict = {"Loss": running_loss / denom, "Accuracy": running_accuracy / denom, "Fbeta@0.5": running_fbeta / denom}
            pbar.set_postfix(metric_dict)
            if (i + 1) % 500 == 0:
                if args.WANDB:
                    wandb.log(metric_dict)
                
                # initialize metrics for next "epoch"
                running_loss = 0.
                running_accuracy = 0.
                running_fbeta = 0.
                denom = 0

        torch.save(model.state_dict(), out_path / "model.pt")
        shutil.copyfile(f"/content/drive/MyDrive/colab_notebooks/kaggle/{comp_name}/src/{proj_name}/config.yaml",
                        out_path / "config.yaml")

    else:
        model_weights = torch.load(out_path / "model.pt")
        model.load_state_dict(model_weights)

    
    # Clear memory before loading test fragments
    # train_dset.labels = None
    # train_dset.image_stacks = []
    # del train_loader, train_dset
    # gc.collect()


    if not args.TRAIN_RUN:
        # Evaluate

        test_fragments = [test_path / fragment_name for fragment_name in test_path.iterdir()]
        print("All fragments:", test_fragments)

        pred_images = []
        model.eval()
        for test_fragment in test_fragments:
            outputs = []

            if type(args.EXP) == str:
                eval_dset = SubvolumeDataset(fragments=[test_fragment], voxel_shape=(6, 16, 16), load_inklabels=False)
            else:
                eval_dset = SubvolumeDataset(fragments=[test_fragment], voxel_shape=args.VOXEL_SHAPE, load_inklabels=False)
            
            eval_loader = thd.DataLoader(eval_dset, batch_size=args.BATCH_SIZE, shuffle=False)

            with torch.no_grad():
                for i, (subvolumes, _) in enumerate(tqdm(eval_loader)):
                    output = model(subvolumes.to(DEVICE)).view(-1).sigmoid().cpu().numpy()
                    outputs.append(output)

                    if type(args.EXP) == str:
                        if i == 1000:
                            print(f"\nReached to {i}th iteration; breaking here for a debugging purpose.")
                            break
            # we only load 1 fragment at a time
            image_shape = eval_dset.image_stacks[0].shape[1:]
            eval_dset.labels = None
            eval_dset.image_stacks = None
            del eval_loader
            gc.collect()

            pred_image = np.zeros(image_shape, dtype=np.uint8)
            outputs = np.concatenate(outputs)
            for (y, x, _), prob in zip(eval_dset.pixels[:outputs.shape[0]], outputs):
                # pred_image[y ,x] = prob > 0.4
                pred_image[y ,x] = prob
            pred_images.append(pred_image)
            
            eval_dset.pixels = None
            del eval_dset
            gc.collect()
            print("Finished", test_fragment)




        submission = defaultdict(list)
        for fragment_id, fragment_name in enumerate(test_fragments):
            submission["Id"].append(fragment_name.name)
            submission["Predicted"].append(rle(pred_images[fragment_id], args.THRES))

        pd.DataFrame.from_dict(submission).to_csv(out_path / "submission.csv", index=False)

    

    if not KAGGLE_ENV:
        if args.WANDB:
            send_line_notification(f"Training of {proj_name} has been done. See {wandbrun.url}", json_path / "line.json")
        else:
            send_line_notification(f"Training of {proj_name} has been done.", json_path / "line.json")