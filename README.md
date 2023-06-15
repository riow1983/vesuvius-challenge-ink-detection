# Overview
https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection

# EXP
- 1: Standard run
- 2: Voxel shape

# ToDos
- Distribution comparison between ink parts and non-ink parts
- Will sub sampling tiffs improve score?
- Object function weight on precision using torch pos_weight https://discuss.pytorch.org/t/weights-in-bcewithlogitsloss/27452/4
- CV strategy


```python
    #### RIOW
    # criterion = nn.BCEWithLogitsLoss()
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([.35], device=DEVICE))
    #### RIOWRIOW
```

100%|██████████| 60000/60000 [50:15<00:00, 19.90it/s, Loss=0.164, Accuracy=0.854, Fbeta@0.5=0.481]   


# CV Folds
https://www.kaggle.com/code/yururoi/pytorch-unet-baseline-with-train-code?scriptVersionId=122620610&cellId=22
```python
data_set = []
data_set.append(
    {
        "train_img": [img1, img2],
        "train_label": [img1_label, img2_label],
        "valid_img": [img3],
        "valid_label": [img3_label],
        "valid_mask": [img3_mask],
    }
)

data_set.append(
    {
        "train_img": [img1, img3],
        "train_label": [img1_label, img3_label],
        "valid_img": [img2],
        "valid_label": [img2_label],
        "valid_mask": [img2_mask],
    }
)

data_set.append(
    {
        "train_img": [img2, img3],
        "train_label": [img2_label, img3_label],
        "valid_img": [img1],
        "valid_label": [img1_label],
        "valid_mask": [img1_mask],
    }
)
```

# 2023-06-15
`src/2-5d-segmentaion-baseline-training`のEXP1とEXP2の比較結:
```
[EXP 1]
-------- exp_info -----------------
best_th: 0.5, fbeta: 0.5089870965985367
Epoch 15 - avg_train_loss: 0.3655  avg_val_loss: 0.4501  time: 72s
Epoch 15 - avgScore: 0.5090
-------- exp_info -----------------
Epoch 15 - avg_train_loss: 0.3254  avg_val_loss: 0.6164  time: 62s
Epoch 15 - avgScore: 0.3941
best_th: 0.1, fbeta: 0.4037166002688188
-------- exp_info -----------------
Epoch 15 - avg_train_loss: 0.3638  avg_val_loss: 0.3804  time: 74s
Epoch 15 - avgScore: 0.4535
best_th: 0.5, fbeta: 0.5139692463216654

[EXP 2]
-------- exp_info -----------------
Epoch 45 - avg_train_loss: 0.1141  avg_val_loss: 0.5670  time: 70s
Epoch 45 - avgScore: 0.4945
best_th: 0.5, fbeta: 0.5518652693152954
-------- exp_info -----------------
Epoch 45 - avg_train_loss: 0.1026  avg_val_loss: 0.8426  time: 64s
Epoch 45 - avgScore: 0.3234
best_th: 0.15, fbeta: 0.40391082025522707
-------- exp_info -----------------
Epoch 45 - avg_train_loss: 0.1221  avg_val_loss: 0.4019  time: 70s
Epoch 45 - avgScore: 0.5515
best_th: 0.5, fbeta: 0.5769704817882036
```

# W&B
https://wandb.ai/riow1983/vesuvius-challenge-ink-detection/table?workspace=user-riow1983

# Discussions
- [For those guys who scored > 0.2](https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection/discussion/413949)
- [Unannotated Signal in Fragment ID 1???](https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection/discussion/417071)

# Notebooks
- [Vesuvius Challenge: Ink Detection tutorial](https://www.kaggle.com/code/jpposma/vesuvius-challenge-ink-detection-tutorial)
- [Pytorch UNet baseline (with train code)](https://www.kaggle.com/code/yururoi/pytorch-unet-baseline-with-train-code/notebook)
- [[0.11] Simplest Possible Solution: submit testmask](https://www.kaggle.com/code/lucasvw/0-11-simplest-possible-solution-submit-testmask)

# Documentations
- [Segmentation Models](https://smp.readthedocs.io/en/latest/index.html)

# GitHub
- [Loss functions for image segmentation](https://github.com/JunMa11/SegLoss)

# Snipets
[Early stopping]
```python
# Credit to: https://www.kaggle.com/code/yururoi/pytorch-unet-baseline-with-train-code?scriptVersionId=122620610&cellId=20
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, fold=""):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            logger.info(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            logger.info(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        torch.save(model.state_dict(), CP_DIR / f"{HOST}_{NB}_checkpoint_{fold}.pt")
        self.val_loss_min = val_loss
```