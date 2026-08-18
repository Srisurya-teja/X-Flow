from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import yaml
from easydict import EasyDict as edict

config = edict()

# ----------------------------- NETWORK -----------------------------
config.NETWORK = edict()
config.NETWORK.PRETRAINED = ''
config.NETWORK.IMAGE_SIZE = (112, 112)
config.NETWORK.FACTOR = 1               # used only to tag the output directory

# ----------------------------- LOSS --------------------------------
config.LOSS = edict()
config.LOSS.TYPE = 'CosFace'            # 'ArcFace' | 'CosFace' | 'Combined'
config.LOSS.WEIGHT_PRED = 1.0           # weight of the mask-prediction loss (Mask mode)
config.LOSS.SCALE = 64.0                # feature scale s for the margin softmax
config.LOSS.MARGIN = 0.4                # margin: CosFace m, ArcFace margin, Combined m2
config.LOSS.MARGIN_WARMUP_EPOCHS = 0    # epochs to linearly ramp margin 0 -> MARGIN (0 = off)

# ----------------------------- DATASET -----------------------------
config.DATASET = edict()
config.DATASET.REC_PATH = ''            # dir containing train.rec / train.idx
config.DATASET.NUM_CLASS = 10572        # number of identities
config.DATASET.TRAIN_DATASET = 'Custom'

# ----------------------------- PartialFC / DDP ---------------------
config.SAMPLE_RATE = 1.0                # fraction of negative classes sampled per step
config.FP16 = False                     # mixed-precision training
config.LR_SCHEDULER = 'multistep'       # 'multistep' | 'polynomial'
config.NUM_IMAGE = 0                    # total images (for polynomial LR schedule)
config.WARMUP_ITERS = 0                 # linear LR-warmup steps (multistep path); 0 = off
config.WARMUP_LR = 0.0                  # starting LR for the warmup ramp

# ----------------------------- Azure File Share upload (optional) ---
config.USE_AZURE = False                                        # upload checkpoints + logs each epoch
config.AZURE_REMOTE_DIR = 'training-checkpoint/from'           # remote base dir on the file share
config.AZURE_SCRIPTS_PATH = '/home/shared_scripts/azure_test/' # where the Azure helper modules live

# ----------------------------- TRAIN -------------------------------
config.TRAIN = edict()
config.TRAIN.OUTPUT_DIR = 'output'
config.TRAIN.MODEL = 'LResNet50E_IR_FPN'
config.TRAIN.MODE = 'Clean'             # 'Clean' | 'Occ' | 'Mask'
config.TRAIN.WORKERS = 8
config.TRAIN.PRINT_FREQ = 100

config.TRAIN.OPTIMIZER = 'sgd'          # 'sgd' | 'adam' | 'adamw'
config.TRAIN.LR = 0.1                    # LR for the PartialFC head (and backbone if BACKBONE_LR=0)
config.TRAIN.BACKBONE_LR = 0            # separate LR for the backbone; 0 = same as LR
config.TRAIN.LR_FACTOR = 0.1
config.TRAIN.LR_STEP = [15, 30]
config.TRAIN.MOMENTUM = 0.9
config.TRAIN.WD = 0.0005

config.TRAIN.START_EPOCH = 0
config.TRAIN.END_EPOCH = 40
config.TRAIN.BATCH_SIZE = 128

config.TRAIN.PATTERN = 5                # grid size for mask quantization
config.TRAIN.NUM_MASK = 226            # number of grid classes (derived from PATTERN)


def _update_dict(k, v):
    for vk, vv in v.items():
        if vk in config[k]:
            config[k][vk] = vv
        else:
            raise ValueError("{}.{} not exist in config.py".format(k, vk))


def update_config(config_file):
    with open(config_file) as f:
        exp_config = edict(yaml.load(f, Loader=yaml.FullLoader))
    for k, v in exp_config.items():
        if k in config:
            if isinstance(v, dict):
                _update_dict(k, v)
            else:
                config[k] = v
        else:
            raise ValueError("{} not exist in config.py".format(k))


def get_model_name(cfg):
    return '{}_{}'.format(cfg.TRAIN.MODEL, cfg.LOSS.TYPE)
