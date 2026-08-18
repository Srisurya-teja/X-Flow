"""
Distributed training script for FROM + PartialFC.
Replaces the original train.py with DDP + PartialFC_V2 support.

Launch with:
    torchrun --nproc_per_node=N train_distributed.py --cfg experiments/your_config.yaml [options]
"""

import os
import sys
import time
import json
import shutil
import re
import csv
import math
import logging
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.utils.data
import torch.optim
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler
from torch.cuda.amp import GradScaler

cudnn.benchmark = True

import lib.core.utils as utils
from lib.core.config import config, update_config

# Models — 112x112 versions
from lib.models.fpn_112 import LResNet50E_IR_Occ_112 as LResNet50E_IR_FPN
from lib.models.fpn_112 import LResNet50E_IR_Occ_2D_112 as LResNet50E_IR_Occ_2D
from lib.models.fpn_112 import LResNet50E_IR_Occ_FC_112 as LResNet50E_IR_Occ_FC
from lib.models.iresnet_fpn import iresnet50_occ

# PartialFC + margin losses
from lib.models.partial_fc_v2 import PartialFC_V2
from lib.models.margin_losses import CombinedMarginLoss, ArcFace, CosFace

# Dataset
from lib.datasets.dataset_rec import MXFaceDataset_Occ

# LR scheduler
from lib.models.lr_scheduler import PolynomialLRWarmup

# setup random seed
torch.manual_seed(0)
np.random.seed(0)

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='FROM + PartialFC Distributed Training')
    parser.add_argument('--cfg', help='experiment config file', required=True, type=str)
    args, rest = parser.parse_known_args()
    update_config(args.cfg)

    parser.add_argument('--frequent', help='logging frequency', default=config.TRAIN.PRINT_FREQ, type=int)
    parser.add_argument('--workers', help='num dataloader workers', type=int)
    parser.add_argument('--batch_size', help='per-GPU batch size', type=int)
    parser.add_argument('--weight_pred', help='weight for mask prediction loss', type=float)
    parser.add_argument('--pattern', help='grid pattern size', type=int)
    parser.add_argument('--lr', help='initial learning rate', type=float)
    parser.add_argument('--optim', help='optimizer type', type=str)
    parser.add_argument('--pretrained', help='pretrained model path or "No"', type=str)
    parser.add_argument('--debug', help='debug mode', default=0, type=int)
    parser.add_argument('--model', help='model name', type=str)
    parser.add_argument('--loss', help='margin loss type (ArcFace/CosFace/Combined)', type=str)
    parser.add_argument('--ratio', help='ratio of masked images', default=4, type=int)
    parser.add_argument('--resume', help='checkpoint to resume from', type=str, default='')
    parser.add_argument('--save_occ_samples', help='save N occluded sample images before training', type=int, default=0)
    parser.add_argument('--max_steps', help='stop each epoch after N steps (0=full); for smoke tests', type=int, default=0)
    args = parser.parse_args()
    return args


def reset_config(config, args):
    if args.workers:
        config.TRAIN.WORKERS = args.workers
    if args.model:
        config.TRAIN.MODEL = args.model
    if args.loss:
        config.LOSS.TYPE = args.loss
    if args.pattern:
        config.TRAIN.PATTERN = args.pattern
        config.TRAIN.NUM_MASK = len(utils.get_grids(*config.NETWORK.IMAGE_SIZE, args.pattern))
    if args.batch_size:
        config.TRAIN.BATCH_SIZE = args.batch_size
    if args.lr:
        config.TRAIN.LR = args.lr
    if args.pretrained == 'No':
        config.NETWORK.PRETRAINED = ''
    elif args.pretrained:
        config.NETWORK.PRETRAINED = args.pretrained
    if args.optim:
        config.TRAIN.OPTIMIZER = args.optim
    if args.weight_pred is not None:
        config.LOSS.WEIGHT_PRED = args.weight_pred


def setup_distributed():
    """Initialize distributed training."""
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def train_one_epoch(train_loader, model, module_pfc, criterion_mask_pred,
                    optimizer, epoch, config, rank, scaler=None,
                    lr_scheduler=None, scheduler_per_step=False,
                    warmup_iters=0, warmup_lr=0.0, base_lr=0.0, max_steps=0):
    """Train for one epoch."""
    model.train()
    time_curr = time.time()
    loss_display = 0.0
    loss_cls_dis = 0.0
    loss_pred_dis = 0.0

    # Whole-epoch accumulators (for the per-epoch summary line).
    epoch_loss = 0.0
    epoch_cls = 0.0
    epoch_pred = 0.0
    n_batches = 0

    pbar = tqdm(train_loader, total=len(train_loader), desc=f'Epoch {epoch}',
                disable=(rank != 0), mininterval=30, dynamic_ncols=True)
    for batch_idx, data in enumerate(pbar):
        iters = epoch * len(train_loader) + batch_idx

        # Linear LR warmup: ramp warmup_lr -> base_lr over the first warmup_iters steps.
        # (Set before the optimizer step so this iteration uses the warmed LR.)
        if warmup_iters and iters < warmup_iters:
            new_lr = warmup_lr + (base_lr - warmup_lr) * (iters + 1) / warmup_iters
            for pg in optimizer.param_groups:
                pg['lr'] = new_lr

        img, label, mask_label, _ = data
        img = img.cuda(non_blocking=True)
        label = label.cuda(non_blocking=True)
        mask_label = mask_label.cuda(non_blocking=True)

        # Forward through backbone
        if config.TRAIN.MODE == 'Clean' or config.TRAIN.MODE == 'Occ':
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    features = model(img)
                    embedding = features[-1]  # fc output
                    # Normalize before PartialFC
                    loss_cls = module_pfc(embedding, label)
                loss = loss_cls
                loss_pred = torch.tensor(0.0)
            else:
                features = model(img)
                embedding = features[-1]
                loss_cls = module_pfc(embedding, label)
                loss = loss_cls
                loss_pred = torch.tensor(0.0)

        elif config.TRAIN.MODE == 'Mask':
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    features = model(img)
                    fc_mask, mask, vec, fc = features
                    loss_cls = module_pfc(fc_mask, label)
                    loss_pred = criterion_mask_pred(vec, mask_label)
                loss = loss_cls + config.LOSS.WEIGHT_PRED * loss_pred
            else:
                features = model(img)
                fc_mask, mask, vec, fc = features
                loss_cls = module_pfc(fc_mask, label)
                loss_pred = criterion_mask_pred(vec, mask_label)
                loss = loss_cls + config.LOSS.WEIGHT_PRED * loss_pred
        else:
            raise ValueError(f'Unknown training mode: {config.TRAIN.MODE}')

        # Backward
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(module_pfc.parameters()), max_norm=5)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(module_pfc.parameters()), max_norm=5)
            optimizer.step()

        # Per-step LR schedules (e.g. polynomial warmup) advance every batch.
        if scheduler_per_step and lr_scheduler is not None:
            lr_scheduler.step()

        l, lc, lp = loss.item(), loss_cls.item(), loss_pred.item()
        loss_display += l
        loss_cls_dis += lc
        loss_pred_dis += lp
        epoch_loss += l
        epoch_cls += lc
        epoch_pred += lp
        n_batches += 1

        if rank == 0:
            pbar.set_postfix(loss=f'{l:.3f}', lr=f'{optimizer.param_groups[0]["lr"]:.2e}')

        if iters % config.TRAIN.PRINT_FREQ == 0 and iters != 0 and rank == 0:
            num_freq = min(batch_idx + 1, config.TRAIN.PRINT_FREQ)
            speed = num_freq / (time.time() - time_curr)
            loss_display /= num_freq
            loss_cls_dis /= num_freq
            loss_pred_dis /= num_freq

            logger.info(
                f'Epoch: {epoch} [{batch_idx}/{len(train_loader)} '
                f'({100. * batch_idx / len(train_loader):.0f}%)] '
                f'Loss: {loss_display:.6f}, '
                f'Cls: {loss_cls_dis:.4f}, Pred: {loss_pred_dis:.4f}*{config.LOSS.WEIGHT_PRED}, '
                f'Speed: {speed:.2f} batches/s')

            time_curr = time.time()
            loss_display = 0.0
            loss_cls_dis = 0.0
            loss_pred_dis = 0.0

        if max_steps and (batch_idx + 1) >= max_steps:
            break

    n = max(n_batches, 1)
    return epoch_loss / n, epoch_cls / n, epoch_pred / n


def save_occluded_samples(dataset, num_samples, output_dir):
    """Save occluded sample images to disk for visual inspection."""
    save_dir = os.path.join(output_dir, 'occluded_samples')
    os.makedirs(save_dir, exist_ok=True)

    # Temporarily bypass transform to get raw PIL images
    orig_transform = dataset.transform
    dataset.transform = None

    denorm = lambda t: ((t * 0.5 + 0.5) * 255).clamp(0, 255).byte()

    logger.info(f'Saving {num_samples} occluded sample images to {save_dir}/')
    for i in range(min(num_samples, len(dataset))):
        try:
            img_pil, label, mask_label, idx = dataset[i]
            if isinstance(img_pil, Image.Image):
                img_pil.save(os.path.join(save_dir, f'sample_{i:04d}_id{label}_grid{mask_label}.jpg'))
            elif isinstance(img_pil, torch.Tensor):
                from torchvision.utils import save_image
                save_image(denorm(img_pil).float() / 255.0,
                           os.path.join(save_dir, f'sample_{i:04d}_id{label}_grid{mask_label}.jpg'))
        except Exception as e:
            logger.info(f'Failed to save sample {i}: {e}')

    dataset.transform = orig_transform
    logger.info(f'Saved {num_samples} samples to {save_dir}/')


_METRICS_FIELDS = ['timestamp', 'epoch', 'avg_loss', 'cls', 'pred',
                   'lr', 'epoch_time_sec', 'img_per_sec']


def _append_metrics_row(csv_path, row):
    """Append one epoch's metrics to csv_path, writing the header if the file is new."""
    write_header = not os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=_METRICS_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _azure_upload(local_path, remote_dir):
    """Best-effort upload of one file to the Azure File Share.

    The SAS token is refreshed on every call (tokens expire on long runs), and
    the Azure helper modules are imported lazily from config.AZURE_SCRIPTS_PATH,
    so training never hard-depends on them. Never raises — logs on failure.
    """
    try:
        scripts_path = config.get('AZURE_SCRIPTS_PATH', '')
        if scripts_path and scripts_path not in sys.path:
            sys.path.append(scripts_path)
        from sas_token_generator import SASTokenGenerator
        from azure_fileshare_user import FileShareUser

        uploader = FileShareUser(SASTokenGenerator(api_resource='fileshare', delete=True))
        uploader.upload_local_file_to_azure(local_path, remote_dir)
        logger.info(f'Azure upload OK: {os.path.basename(local_path)} -> {remote_dir}')
    except Exception as e:
        logger.error(f'Azure upload failed for {local_path}: {e}')


def _set_margin(margin_loss, loss_type, value):
    """Update the margin on the loss object in place (PartialFC holds the same ref)."""
    if loss_type == 'CosFace':
        margin_loss.m = value
    elif loss_type == 'ArcFace':
        margin_loss.margin = value
    elif loss_type == 'Combined':
        margin_loss.m2 = value


def main():
    args = parse_args()
    reset_config(config, args)

    # Setup distributed
    rank, local_rank, world_size = setup_distributed()

    # Only rank 0 sets up logging and resolves the output dir; that dir is then
    # broadcast so EVERY rank writes its checkpoint shard to the same directory.
    if rank == 0:
        if args.debug:
            log, final_output_dir, tb_log_dir = utils.create_temp_logger()
        else:
            log, final_output_dir, tb_log_dir = utils.create_logger(config, args.cfg, 'train')
        logger.info('Config: \n' + json.dumps(config, indent=4, sort_keys=True))
    else:
        final_output_dir, tb_log_dir = None, None

    # Broadcast the resolved output dir from rank 0 to all ranks.
    _paths = [final_output_dir, tb_log_dir]
    dist.broadcast_object_list(_paths, src=0)
    final_output_dir, tb_log_dir = _paths
    os.makedirs(final_output_dir, exist_ok=True)

    # Azure File Share upload setup (rank 0 only)
    use_azure = (rank == 0) and config.get('USE_AZURE', False)
    azure_remote_dir = None
    if use_azure:
        azure_remote_dir = '{}/{}'.format(
            config.AZURE_REMOTE_DIR.rstrip('/'),
            os.path.basename(final_output_dir.rstrip('/')))
        logger.info(f'Azure upload enabled -> {azure_remote_dir}')

    # ================================ MODEL ================================
    model = {
        'LResNet50E_IR_FPN': LResNet50E_IR_FPN(num_mask=config.TRAIN.NUM_MASK),
        'LResNet50E_IR_Occ_2D': LResNet50E_IR_Occ_2D(num_mask=config.TRAIN.NUM_MASK),
        'LResNet50E_IR_Occ_FC': LResNet50E_IR_Occ_FC(num_mask=config.TRAIN.NUM_MASK),
        'iResNet50_FPN': iresnet50_occ(num_mask=config.TRAIN.NUM_MASK),
    }[config.TRAIN.MODEL]

    # Load pretrained backbone (optional)
    # Load pretrained backbone (optional)
    if config.NETWORK.PRETRAINED and config.NETWORK.PRETRAINED != '':
        pretrained_path = config.NETWORK.PRETRAINED
        if os.path.isfile(pretrained_path):
            if rank == 0:
                logger.info(f'Loading pretrained backbone from {pretrained_path}')
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            
            # Handle different checkpoint formats
            if 'state_dict' in checkpoint:
                pretrained_state = checkpoint['state_dict']
            else:
                pretrained_state = checkpoint
            
            model_state = model.state_dict()
            
            # FROM branch prefixes — keep these random
            skip_prefixes = ('fpn.', 'mask.', 'regress.', 'fc.')
            
            loaded_count = 0
            skipped_count = 0
            for key, value in pretrained_state.items():
                if key in model_state and model_state[key].shape == value.shape:
                    if not key.startswith(skip_prefixes):
                        model_state[key] = value
                        loaded_count += 1
                    else:
                        skipped_count += 1
                else:
                    skipped_count += 1
            
            model.load_state_dict(model_state)
            if rank == 0:
                logger.info(f'Loaded {loaded_count} pretrained backbone params')
                logger.info(f'Skipped {skipped_count} params (FROM branch or mismatched)')
    else:
        if rank == 0:
            logger.info(f'No pretrained model found at {pretrained_path}')
        else:
            if rank == 0:
                logger.info(f'No pretrained model found at {pretrained_path}')

    model = model.cuda()
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    # ================================ PARTIAL FC ================================
    use_fp16 = config.get('FP16', False)

    s = config.LOSS.SCALE
    m = config.LOSS.MARGIN
    margin_loss = {
        'ArcFace': ArcFace(s=s, margin=m),
        'CosFace': CosFace(s=s, m=m),
        'Combined': CombinedMarginLoss(s=s, m1=1.0, m2=m, m3=0.0),
    }.get(config.LOSS.TYPE, CosFace(s=s, m=m))

    module_pfc = PartialFC_V2(
        margin_loss=margin_loss,
        embedding_size=512,
        num_classes=config.DATASET.NUM_CLASS,
        sample_rate=config.get('SAMPLE_RATE', 1.0),
        fp16=use_fp16,
    ).cuda()

    # Mask prediction loss (for Mask mode)
    criterion_mask_pred = nn.CrossEntropyLoss().cuda()

    # ================================ OPTIMIZER ================================
    opt_params = [
        {'params': model.parameters()},
        {'params': module_pfc.parameters()},
    ]

    if config.TRAIN.OPTIMIZER == 'sgd':
        optimizer = torch.optim.SGD(
            opt_params,
            lr=config.TRAIN.LR,
            momentum=config.TRAIN.MOMENTUM,
            weight_decay=config.TRAIN.WD)
    elif config.TRAIN.OPTIMIZER == 'adam':
        optimizer = torch.optim.Adam(opt_params, lr=config.TRAIN.LR)
    elif config.TRAIN.OPTIMIZER == 'adamw':
        optimizer = torch.optim.AdamW(opt_params, lr=config.TRAIN.LR, weight_decay=config.TRAIN.WD)
    else:
        raise ValueError(f'Unknown optimizer: {config.TRAIN.OPTIMIZER}')

    # ================================ LR SCHEDULER ================================
    lr_scheduler_type = config.get('LR_SCHEDULER', 'multistep')

    if lr_scheduler_type in ('polynomial', 'cosine'):
        # Per-step schedules need the total step count (from NUM_IMAGE).
        num_images = config.get('NUM_IMAGE', 0)
        batch_total = config.TRAIN.BATCH_SIZE * world_size
        steps_per_epoch = num_images // batch_total if num_images > 0 else 1000
        total_steps = steps_per_epoch * config.TRAIN.END_EPOCH
        if num_images <= 0 and rank == 0:
            logger.info('WARNING: NUM_IMAGE not set; per-step LR schedule length is a guess.')

        if lr_scheduler_type == 'polynomial':
            lr_scheduler = PolynomialLRWarmup(
                optimizer, warmup_iters=total_steps // 10, total_iters=total_steps, power=2.0)
        else:  # cosine: linear warmup (WARMUP_ITERS) then cosine decay to 0
            warm = config.get('WARMUP_ITERS', 0)
            base = config.TRAIN.LR
            warm_start = (config.get('WARMUP_LR', 0.0) / base) if base > 0 else 0.0

            def _cosine_lambda(step, warm=warm, warm_start=warm_start,
                               total_steps=total_steps):
                if warm and step < warm:
                    return warm_start + (1.0 - warm_start) * step / warm
                progress = min(1.0, (step - warm) / max(1, total_steps - warm))
                return 0.5 * (1.0 + math.cos(math.pi * progress))

            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _cosine_lambda)
        scheduler_per_step = True
    else:
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, config.TRAIN.LR_STEP, config.TRAIN.LR_FACTOR)
        scheduler_per_step = False

    # FP16 scaler
    scaler = GradScaler() if use_fp16 else None

    # ================================ RESUME ================================
    # Each rank resumes from ITS OWN checkpoint shard (PartialFC is sharded per
    # rank). --resume accepts a directory, a per-rank file, or a plain file:
    #   dir           -> <dir>/checkpoint_gpu_<rank>.pt
    #   .../gpu_N.pt  -> remapped to gpu_<rank>.pt
    #   other file    -> loaded as-is on every rank (single-GPU / legacy)
    start_epoch = config.TRAIN.START_EPOCH
    if args.resume:
        if os.path.isdir(args.resume):
            resume_path = os.path.join(args.resume, f'checkpoint_gpu_{rank}.pt')
        elif re.search(r'gpu_\d+', os.path.basename(args.resume)):
            resume_path = re.sub(r'gpu_\d+', f'gpu_{rank}', args.resume)
        else:
            resume_path = args.resume

        if not os.path.isfile(resume_path):
            raise FileNotFoundError(f'No resume checkpoint for rank {rank}: {resume_path}')

        logger.info(f'[rank {rank}] resuming from {resume_path}')
        checkpoint = torch.load(resume_path, map_location='cpu')

        # Backbone (identical across ranks)
        model.module.load_state_dict(checkpoint['state_dict'])

        # PartialFC classifier shard for THIS rank — without it the head restarts
        # from random init and the loss spikes for several epochs after resume.
        if 'state_dict_softmax_fc' in checkpoint:
            module_pfc.load_state_dict(checkpoint['state_dict_softmax_fc'])
        elif rank == 0:
            logger.info('WARNING: checkpoint has no PartialFC head; classifier starts fresh.')

        # Optimizer (momentum buffers for backbone + this rank's classifier shard)
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])

        # LR scheduler position — restores the warmup/decay step so the LR continues
        # instead of restarting from step 0 (critical for the polynomial schedule).
        if 'state_lr_scheduler' in checkpoint:
            lr_scheduler.load_state_dict(checkpoint['state_lr_scheduler'])
        elif rank == 0:
            logger.info('WARNING: checkpoint has no LR scheduler state; schedule restarts.')

        # FP16 GradScaler state (only relevant when training with fp16).
        if scaler is not None and checkpoint.get('scaler') is not None:
            scaler.load_state_dict(checkpoint['scaler'])

        start_epoch = checkpoint.get('epoch', 0)
        if rank == 0:
            logger.info(f'Resumed at epoch {start_epoch}, '
                        f'LR {optimizer.param_groups[0]["lr"]:.6f}')

    # ================================ DATA ================================
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    train_dataset = MXFaceDataset_Occ(
        root_dir=config.DATASET.REC_PATH,
        mode=config.TRAIN.MODE,
        img_size=config.NETWORK.IMAGE_SIZE,
        pattern=config.TRAIN.PATTERN,
        ratio=args.ratio,
        transform=train_transform,
    )

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE,
        sampler=train_sampler,
        num_workers=config.TRAIN.WORKERS,
        pin_memory=False,   # per-device pin_memory can throw "CUDA invalid argument" on non-0 ranks
        drop_last=True,
    )

    if rank == 0:
        logger.info(f'Train dataset: {len(train_dataset)} images')
        logger.info(f'Num classes: {config.DATASET.NUM_CLASS}')
        logger.info(f'World size: {world_size}, Per-GPU batch: {config.TRAIN.BATCH_SIZE}')

    # ================================ SAVE OCCLUDED SAMPLES ================================
    if args.save_occ_samples > 0 and rank == 0:
        save_occluded_samples(train_dataset, args.save_occ_samples, final_output_dir)

    # ================================ TRAIN ================================
    start = time.time()

    # Manual linear LR warmup for the multistep path (the polynomial scheduler
    # already warms up internally, so disable it there to avoid double warmup).
    warmup_iters = 0 if scheduler_per_step else config.get('WARMUP_ITERS', 0)
    warmup_lr = config.get('WARMUP_LR', 0.0)
    base_lr = config.TRAIN.LR
    if rank == 0 and warmup_iters:
        logger.info(f'LR warmup: {warmup_lr} -> {base_lr} over {warmup_iters} steps')

    # Margin warmup: smooth linear ramp 0.0 -> LOSS.MARGIN over MARGIN_WARMUP_EPOCHS.
    target_margin = config.LOSS.MARGIN
    margin_warmup_epochs = config.LOSS.get('MARGIN_WARMUP_EPOCHS', 0)
    if rank == 0 and margin_warmup_epochs:
        logger.info(f'Margin warmup: 0.0 -> {target_margin} over {margin_warmup_epochs} epochs')

    for epoch in range(start_epoch, config.TRAIN.END_EPOCH):
        train_sampler.set_epoch(epoch)
        epoch_start = time.time()

        # Set this epoch's margin (linear ramp, then held at target_margin).
        if margin_warmup_epochs > 0:
            cur_margin = min(1.0, epoch / margin_warmup_epochs) * target_margin
        else:
            cur_margin = target_margin
        _set_margin(margin_loss, config.LOSS.TYPE, cur_margin)
        if rank == 0:
            logger.info(f'Epoch {epoch}: {config.LOSS.TYPE} margin = {cur_margin:.4f}')

        avg_loss, avg_cls, avg_pred = train_one_epoch(
            train_loader, model, module_pfc, criterion_mask_pred,
            optimizer, epoch, config, rank, scaler,
            lr_scheduler=lr_scheduler, scheduler_per_step=scheduler_per_step,
            warmup_iters=warmup_iters, warmup_lr=warmup_lr, base_lr=base_lr,
            max_steps=args.max_steps)

        # Per-epoch LR schedules (e.g. multistep) advance once per epoch.
        if not scheduler_per_step:
            lr_scheduler.step()

        # --- Resumable checkpoint: EVERY rank saves its own shard ---
        # PartialFC shards the classifier across GPUs, so each rank holds a
        # different slice of the head + its optimizer state. The backbone is
        # DDP-synced (identical everywhere) but included in each file so a rank
        # can resume from its own checkpoint alone.
        torch.save({
            'epoch': epoch + 1,
            'state_dict': model.module.state_dict(),
            'state_dict_softmax_fc': module_pfc.state_dict(),
            'optimizer': optimizer.state_dict(),
            'state_lr_scheduler': lr_scheduler.state_dict(),
            'scaler': scaler.state_dict() if scaler is not None else None,
        }, os.path.join(final_output_dir, f'checkpoint_gpu_{rank}.pt'))

        # Ensure all ranks finished writing before rank 0 reads/uploads them.
        dist.barrier()

        # --- Rank-0-only: metrics, eval backbone, logging, Azure uploads ---
        if rank == 0:
            cur_lr = optimizer.param_groups[0]['lr']
            epoch_time = time.time() - epoch_start

            # Estimated time remaining, from average epoch time so far.
            epochs_done = epoch - start_epoch + 1
            epochs_left = config.TRAIN.END_EPOCH - epoch - 1
            eta_h = (time.time() - start) / epochs_done * epochs_left / 3600.0

            logger.info(
                f'Epoch {epoch} done | avg_loss={avg_loss:.6f} '
                f'cls={avg_cls:.4f} pred={avg_pred:.4f} | LR={cur_lr:.6f} | '
                f'time={epoch_time:.1f}s | ETA={eta_h:.1f}h ({epochs_left} epochs left)')

            # Per-epoch metrics CSV (one row per epoch; appended, survives resume).
            metrics_csv = os.path.join(final_output_dir, 'training_metrics.csv')
            _append_metrics_row(metrics_csv, {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'epoch': epoch,
                'avg_loss': round(avg_loss, 6),
                'cls': round(avg_cls, 6),
                'pred': round(avg_pred, 6),
                'lr': round(cur_lr, 8),
                'epoch_time_sec': round(epoch_time, 2),
                'img_per_sec': round(len(train_dataset) / epoch_time, 1) if epoch_time > 0 else 0,
            })

            # Per-epoch backbone weights — load these in your own evaluation pipeline.
            epoch_path = os.path.join(final_output_dir, f'backbone_epoch_{epoch:03d}.pth.tar')
            torch.save({'epoch': epoch + 1, 'state_dict': model.module.state_dict()}, epoch_path)

            # Upload this epoch's outputs to Azure (best-effort; never blocks training).
            if use_azure:
                _azure_upload(epoch_path, azure_remote_dir)
                for r in range(world_size):
                    _azure_upload(
                        os.path.join(final_output_dir, f'checkpoint_gpu_{r}.pt'), azure_remote_dir)
                _azure_upload(metrics_csv, azure_remote_dir)

        dist.barrier()

    if rank == 0:
        time_used = (time.time() - start) / 3600.0
        logger.info(f'Done Training. Consumed {time_used:.2f} hours')

        # Upload training logs at the end (best-effort).
        if use_azure:
            import glob
            for log_file in glob.glob(os.path.join(final_output_dir, '*.log')):
                _azure_upload(log_file, azure_remote_dir)

    dist.destroy_process_group()


if __name__ == '__main__':
    main()
