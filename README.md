# Overview
https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection


# Ideas
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

# 反省点
- 2.5d segmentationというのが流行っているらしい
- "データ分析"ができなかった(やる気が起きなかった)
- [こういう発見](https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection/discussion/417071)をしてみたい
- 次回からはせめて何らかの"分布"は自分で見るようにしたい
- [こういう日記](https://ai-ml-dl.hatenablog.com/entry/2020/06/30/095516)書こう; せめて公開ノートブックのキュレーション記録くらいは.
- 序盤に見つけていた公開ノートブックを終盤に"再発見"するなどの愚は記録を付けて見返す習慣があれば防げた
- 3 folds以外考えられなかった
- 画像を分割してfold数を増やす方法くらい思いつけと１ヶ月前の自分に言いたい
- foldごとの訓練は意外といい. チーム組んでやる上でも分担効率が良い.
- foldごとに実験結果を比較できるようW&BのRUN名にfold数も入れるべきだった
- W&Bは最初期の実行時から適用すべきだった (EXP1の結果だけW&Bに無いなどということが無いように)
- 最終週だけ業務時間ほぼ全投入. それでようやく面白くなってきたが時間切れ. 
- 火を付けるためにはコンペ参加初期にもあえて全投入週を作るべきかも.
- セグメンテーションは`segmentation_models_pytorch`のお陰でモデル実装は簡便だが, データオーグメンテーションは`albumentations`に習熟する必要があり難しい
- 次期コンペもセグメンテーションコンペにするか?
- 特に何も思いつかなくても脳死状態でもアンサンブルはできる; 特にハイパラ変更すらせずモデルの種類を変更するだけのアンサンブルでも可
- [37th solution](https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection/discussion/417258)に触発されて[late submit](https://www.kaggle.com/code/riow1983/2-5d-segmentaion-model-with-rotate-tta?scriptVersionId=133643142)したら少しスコア上がった(0.36 -> 0.45)ので, ちょっとした変更をするだけで効果はある
- 1stを含めtop solutionsの多くが[2.5 starter code](https://www.kaggle.com/code/tanakar/2-5d-segmentaion-baseline-training)を使っており, 自分の公開ノートブックを見る目は正しかったと判明; なおこのノートブックには[ここ](https://github.com/riow1983/vesuvius-challenge-ink-detection/issues/1#issue-1699376178)に記載しているようにMay 8の時点で認知していたが、実際に触り始めたのはコンペ最終週からだったというのが悔やまれる

# Top solutions
[1] https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection/discussion/417496  
[2] https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection/discussion/417255  
[3] https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection/discussion/417536  
[4] https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection/discussion/417779  
[5] https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection/discussion/417642  
[6] https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection/discussion/417274  
[7] https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection/discussion/417430  
[8] https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection/discussion/417383  
[9] https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection/discussion/417361  
[10] https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection/discussion/417363  
[11] https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection/discussion/417281  

# Q&A
- Q)テストデータのfragment数は?: A)2つ
```
# https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection/data
You can expect two fragments in the hidden test set, which together are roughly the same size as a single training fragment. The sample slices available to download in the test folders are simply copied from training fragment one, but when you submit your notebook they will be substituted with the real test data.
```
- Q)`segmentation_models_pytorch`で使われるエンコーダはpre-trainedなのか?: A)引数`encoder_weights`に`imagenet`を渡すとpre-trainedになるが, `None`を渡すとrandom initialized (no pre-trained)となる https://segmentation-modelspytorch.readthedocs.io/en/latest/#segmentation_models_pytorch.Unet なお上位解法ではどちらも採用されていた模様

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
- [Segmentation Models / Encoders](https://segmentation-modelspytorch.readthedocs.io/en/latest/#encoders)

# GitHub
- [ink-ID](https://github.com/educelab/ink-id/tree/develop)
- [Loss functions for image segmentation](https://github.com/JunMa11/SegLoss)
- [SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers](https://github.com/NVlabs/SegFormer)

# Papers
- [SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers](https://arxiv.org/pdf/2105.15203.pdf)

# Snipets
[Ensemble models w/ TTA on parallel processing]
```python
# Credit to https://www.kaggle.com/code/riow1983/2-5d-segmentaion-model-with-rotate-tta/notebook?scriptVersionId=133522169
class CustomModel(nn.Module):
    def __init__(self, cfg, weight=None):
        super().__init__()
        self.cfg = cfg

        self.encoder = smp.Unet(
            encoder_name=cfg.backbone, 
            encoder_weights=weight,
            in_channels=cfg.in_chans,
            classes=cfg.target_size,
            activation=None,
        )

    def forward(self, image):
        output = self.encoder(image)
        output = output.squeeze(-1)
        return output

def build_model(cfg, weight="imagenet"):
    print('model_name', cfg.model_name)
    print('backbone', cfg.backbone)

    model = CustomModel(cfg, weight)
    return model


class EnsembleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.ModuleList()
        for fold in [1, 2, 3]:
            _model = build_model(CFG, weight=None)
            model_path = f'/kaggle/input/{CFG.exp_name}/vesuvius-challenge-ink-detection-models/Unet_fold{fold}_best.pth'
            state = torch.load(model_path)['model']
            state = torch.load(model_path)['model']
            _model.load_state_dict(state)
            _model.eval()

            self.model.append(_model)
    
    def forward(self,x):
        output=[]
        for m in self.model:
            output.append(m(x))
        output=torch.stack(output, dim=0).mean(0)
        return output
        
model = EnsembleModel()
model = nn.DataParallel(model, device_ids=[0, 1])
model = model.cuda()


def TTA(x:tc.Tensor, model:nn.Module):
    #x.shape=(batch,c,h,w)
    if CFG.TTA:
        shape=x.shape
        x=[x,*[tc.rot90(x,k=i,dims=(-2,-1)) for i in range(1,4)]]
        x=tc.cat(x,dim=0)
        x=model(x)
        x=torch.sigmoid(x)
        x=x.reshape(4,shape[0],*shape[2:])
        x=[tc.rot90(x[i],k=-i,dims=(-2,-1)) for i in range(4)]
        x=tc.stack(x,dim=0)
        return x.mean(0)
    else :
        x=model(x)
        x=torch.sigmoid(x)
        return x


# Under an inference iteration:
    for step, (images) in tqdm(enumerate(test_loader), total=len(test_loader)):
        images = images.cuda()
        batch_size = images.size(0)

        with torch.no_grad():
            y_preds = TTA(images,model).cpu().numpy()
```

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