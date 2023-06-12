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


# Discussions
- [For those guys who scored > 0.2](https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection/discussion/413949)

# Notebooks
- [Pytorch UNet baseline (with train code)](https://www.kaggle.com/code/yururoi/pytorch-unet-baseline-with-train-code/notebook)

# Documentations
- [Segmentation Models](https://smp.readthedocs.io/en/latest/index.html)