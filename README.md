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
https://www.kaggle.com/code/yururoi/pytorch-unet-baseline-with-train-code?cellIds=22&kernelSessionId=122620610
```python
# Credit to 
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

# Notebooks
- [Pytorch UNet baseline (with train code)](https://www.kaggle.com/code/yururoi/pytorch-unet-baseline-with-train-code/notebook)
- [[0.11] Simplest Possible Solution: submit testmask](https://www.kaggle.com/code/lucasvw/0-11-simplest-possible-solution-submit-testmask)

# Documentations
- [Segmentation Models](https://smp.readthedocs.io/en/latest/index.html)