import os
import time
import random
import logging
from pathlib import Path

import torch
import numpy as np

from lib.core.config import get_model_name

logger = logging.getLogger(__name__)


# ############## logger related ######################
def create_temp_logger():
    output_dir = 'temp'
    root_output_dir = Path(output_dir)
    if not root_output_dir.exists():
        print('creating {}'.format(root_output_dir))
        root_output_dir.mkdir()

    final_output_dir = root_output_dir
    final_output_dir.mkdir(parents=True, exist_ok=True)

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}.log'.format(time_str)
    final_log_file = final_output_dir / log_file

    _setup_logging(final_log_file)

    tensorboard_log_dir = root_output_dir / 'tensorboard'
    print('creating {}'.format(tensorboard_log_dir))
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

    return logging.getLogger(), str(final_output_dir), str(tensorboard_log_dir)


def create_logger(cfg, cfg_name, phase='train'):
    root_output_dir = Path(cfg.TRAIN.OUTPUT_DIR)
    if not root_output_dir.exists():
        print('creating {}'.format(root_output_dir))
        root_output_dir.mkdir()

    dataset = cfg.DATASET.TRAIN_DATASET
    model = get_model_name(cfg) + '-' + cfg.TRAIN.MODE
    cfg_name = os.path.basename(cfg_name).split('.')[0]

    pretrained = 1 if cfg.NETWORK.PRETRAINED else 0
    flag = '-pattern_{}-weight_{}-lr_{}-optim_{}-pretrained_{}_factor_{}'.format(
        cfg.TRAIN.PATTERN, cfg.LOSS.WEIGHT_PRED, cfg.TRAIN.LR,
        cfg.TRAIN.OPTIMIZER, pretrained, cfg.NETWORK.FACTOR)
    cfg_name += flag

    final_output_dir = root_output_dir / dataset / model / cfg_name
    print('creating {}'.format(final_output_dir))
    final_output_dir.mkdir(parents=True, exist_ok=True)

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}.log'.format(time_str, phase)
    final_log_file = final_output_dir / log_file

    _setup_logging(final_log_file)

    tensorboard_file = 'tensorboard_{}'.format(time_str)
    tensorboard_log_dir = final_output_dir / tensorboard_file
    print('creating {}'.format(tensorboard_log_dir))
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

    return logging.getLogger(), str(final_output_dir), str(tensorboard_log_dir)


def _setup_logging(final_log_file):
    fmt = "[%(asctime)s] %(message)s"
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    fh = logging.FileHandler(final_log_file)
    fh.setFormatter(logging.Formatter(fmt))
    fh.setLevel(logging.INFO)
    root_logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter(fmt))
    sh.setLevel(logging.INFO)
    root_logger.addHandler(sh)
    root_logger.propagate = False


# ############## save and load ######################
def load_part_params(model, trained_params, ignore):
    """Copy matching params from trained_params into model, skipping any
    parameter whose name contains `ignore` (e.g. 'regress')."""
    dict_params = dict(model.state_dict())
    for n, p in trained_params.items():
        if n in dict_params and ignore not in n:
            dict_params[n].data.copy_(p.data)
    model.load_state_dict(dict_params, strict=False)
    return model


# ############## mask and grid related ######################
def occlist_reader(fileList):
    occList = []
    with open(fileList, 'r') as file:
        for line in file.readlines():
            occList.append(line.strip())
    return occList


def occluded_image_ratio(img, occ, factor=1.0):
    """Paste occluder `occ` (scaled by `factor`) at a random location on `img`.
    Returns the occluded image, a binary pixel mask, and the occluded area ratio."""
    W, H = img.size
    occ_w, occ_h = occ.size

    new_w, new_h = min(W - 1, int(factor * occ_w)), min(H - 1, int(factor * occ_h))
    occ = occ.resize((new_w, new_h))

    center_x = random.choice(range(0, W))
    center_y = random.choice(range(0, H))

    start_x = center_x - new_w // 2
    start_y = center_y - new_h // 2
    end_x = center_x + (new_w + 1) // 2
    end_y = center_y + (new_h + 1) // 2

    img.paste(occ, (start_x, start_y, end_x, end_y))

    start_x = max(start_x, 0)
    start_y = max(start_y, 0)
    end_x = min(W - 1, end_x)
    end_y = min(H - 1, end_y)
    mask = np.zeros((H, W))
    mask[start_y:end_y, start_x:end_x] = 1.0

    ratio = ((end_y - start_y) * (end_x - start_x)) / float(H * W)
    return img, mask, ratio


def get_grids(H, W, N):
    """Enumerate all axis-aligned rectangles on an N x N grid over an HxW image.
    Index 0 is the empty (no-occlusion) grid."""
    grid_ori = np.zeros((H, W))

    x_axis = np.linspace(0, W, N + 1, True, dtype=int)
    y_axis = np.linspace(0, H, N + 1, True, dtype=int)

    vertex_set = []
    for y in y_axis:
        for x in x_axis:
            vertex_set.append((y, x))

    grids = [grid_ori]
    for start in vertex_set:
        for end in vertex_set:
            if end[0] > start[0] and end[1] > start[1]:
                grid = grid_ori.copy()
                grid[start[0]:end[0], start[1]:end[1]] = 1.0
                grids.append(grid)
    return grids


def cal_IoU(mask1, mask2):
    inter = np.sum(mask1 * mask2)
    union = np.sum(np.clip(mask1 + mask2, 0, 1)) + 1e-10
    return inter / union


def cal_similarity_label(grids, mask):
    """Return the index of the grid rectangle with highest IoU against `mask`."""
    scores = [cal_IoU(grid, mask) for grid in grids]
    return int(np.argmax(scores))
