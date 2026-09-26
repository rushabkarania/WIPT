"""Experiment constants used by the WIPT paper."""

BACKBONE = "vit_small_patch16_224.augreg_in21k_ft_in1k"
EMBED_DIM = 384

N_WAY = 5
N_SHOT = 5
N_QUERY = 15

TRAIN_EPISODES = 200
VAL_EPISODES = 50
EPOCHS = 50
LEARNING_RATE = 1e-4
ETA_MIN = 1e-6

WIPT_LAYERS = 2
WIPT_HEADS = 6
WIPT_DROPOUT = 0.1

TRAIN_SEEDS = (0, 1, 2, 3, 4)
EVAL_SEEDS = (42, 123, 2024, 7, 999)
INDOMAIN_EPISODES_PER_SEED = 2000
CROSS_DOMAIN_EPISODES_PER_SEED = 300
CORRUPTION_EPISODES_PER_SEED = 200
NOISE_EPISODES_PER_SEED = 200

# Local/cloud DataLoader default. Override per CLI with --workers if needed.
NUM_WORKERS = 6

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DEFAULT_DATA_PATHS = {
    "miniImageNet_train": "data/miniimagenet_split/train",
    "miniImageNet_val": "data/miniimagenet_split/val",
    "miniImageNet_test": "data/miniimagenet_split/test",
    "CUB": "data/CUB_200_2011/images",
    "EuroSAT": "data/EuroSAT",
    "ISIC": "data/ISIC",
    "CropDisease": "data/CropDisease",
    "ChestX": "data/ChestX",
}

CHECKPOINTS = {
    "WIPT-2": "checkpoints/wipt_l2.pth",
    "WIPT-4": "checkpoints/wipt_l4.pth",
    "WIPT-6": "checkpoints/wipt_l6.pth",
    "WIPT-Residual": "checkpoints/wipt_residual.pth",
    "WIPT-All": "checkpoints/wipt_all.pth",
    "WIPT-Margin": "checkpoints/wipt_margin.pth",
    "SupportTransformer+ViT": "checkpoints/support_transformer.pth",
    "RelationHead+ViT": "checkpoints/relation_head.pth",
}
