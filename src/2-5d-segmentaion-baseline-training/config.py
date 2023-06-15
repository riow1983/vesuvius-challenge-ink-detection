#########################################
################ config.py ##############
#########################################

import albumentations as A
from albumentations.pytorch import ToTensorV2

class CFG:
    # ============== comp exp name =============
    comp_name = 'vesuvius-challenge-ink-detection'
    proj_name = "2-5d-segmentaion-baseline-training"

    # comp_dir_path = './'
    # comp_dir_path = '/kaggle/input/'
    comp_dir_path = f"/content/drive/MyDrive/colab_notebooks/kaggle/{comp_name}/input/"
    comp_folder_name = 'vesuvius-challenge-ink-detection'
    # comp_dataset_path = f'{comp_dir_path}datasets/{comp_folder_name}/'
    comp_dataset_path = f'{comp_dir_path}{comp_folder_name}/'
    
    exp_name = f"exp-3-{proj_name}"
    # exp_name = "debug"

    # ============== pred target =============
    target_size = 1

    # ============== model cfg =============
    model_name = 'Unet'
    # backbone = 'efficientnet-b0'
    # backbone = 'se_resnext50_32x4d'
    # backbone = 'resnet101'
    backbone = 'resnet34'

    in_chans = 6 # 65
    # ============== training cfg =============
    size = 320
    tile_size = 320
    stride = tile_size // 2

    train_batch_size = 16 # 32
    valid_batch_size = train_batch_size * 2
    use_amp = True

    scheduler = 'GradualWarmupSchedulerV2'
    # scheduler = 'CosineAnnealingLR'
    epochs = 15 # 30

    # adamW warmupあり
    warmup_factor = 10
    # lr = 1e-4 / warmup_factor
    lr = 1e-4 / warmup_factor

    # ============== fold =============
    valid_id = 3

    # objective_cv = 'binary'  # 'binary', 'multiclass', 'regression'
    metric_direction = 'maximize'  # maximize, 'minimize'
    # metrics = 'dice_coef'

    # ============== fixed =============
    pretrained = True
    inf_weight = 'best'  # 'best'

    min_lr = 1e-6
    weight_decay = 1e-6
    max_grad_norm = 1000

    print_freq = 50
    num_workers = 4

    seed = 42

    # ============== set dataset path =============
    print('set dataset path')

    # outputs_path = f'/kaggle/working/outputs/{comp_name}/{exp_name}/'
    outputs_path = f"/content/drive/MyDrive/colab_notebooks/kaggle/{comp_name}/output/{proj_name}/{exp_name}/"

    submission_dir = outputs_path + 'submissions/'
    submission_path = submission_dir + f'submission_{exp_name}.csv'

    model_dir = outputs_path + f'{comp_name}-models/'

    figures_dir = outputs_path + 'figures/'

    log_dir = outputs_path + 'logs/'
    log_path = log_dir + f'{exp_name}.txt'

    # ============== augmentation =============
    train_aug_list = [
        A.Resize(size, size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.75),
        A.GaussNoise(var_limit=[10, 50]),
        A.GaussianBlur(),
        A.RandomGamma(p=0.5, gamma_limit=(50, 200)),
        A.OneOf([A.OpticalDistortion(p=0.3), A.GridDistortion(p=0.3), A.ElasticTransform(p=0.1)], p=0.5),
        A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
        A.CoarseDropout(max_holes=1, max_width=int(size * 0.3), max_height=int(size * 0.3), mask_fill_value=0, p=0.5),
        A.Normalize(
            mean= [0] * in_chans,
            std= [1] * in_chans
        ),
        ToTensorV2(transpose_mask=True)
    ]

    valid_aug_list = [
        A.Resize(size, size),
        A.Normalize(
            mean= [0] * in_chans,
            std= [1] * in_chans
        ),
        ToTensorV2(transpose_mask=True),
    ]
    
    wandb_json_path = "/content/drive/MyDrive/colab_notebooks/kaggle/wandb.json"
    line_json_path = "/content/drive/MyDrive/colab_notebooks/kaggle/line.json"