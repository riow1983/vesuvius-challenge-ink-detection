# EXP
- 1.0: Standard run

# ToDos
- Distribution omparison between ink parts and non-ink parts
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