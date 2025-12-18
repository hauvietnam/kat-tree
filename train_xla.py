#!/usr/bin/env python3
""" ImageNet Training Script for TPU (XLA)
Adapted from original user script.
Hardware: TPU v5e-8 / Google Cloud TPU
"""
import argparse
import importlib
import json
import logging
import os
import time
from collections import OrderedDict
from contextlib import suppress
from datetime import datetime
from functools import partial

import torch
import torch.nn as nn
import torchvision.utils
import yaml

# --- XLA IMPORTS ---
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.utils.utils as xu

from timm import utils
from timm.data import create_dataset, create_loader, resolve_data_config, Mixup, FastCollateMixup, AugMixDataset
from timm.layers import convert_splitbn_model, convert_sync_batchnorm, set_fast_norm
from timm.loss import JsdCrossEntropy, SoftTargetCrossEntropy, BinaryCrossEntropy, LabelSmoothingCrossEntropy
from timm.models import create_model, safe_model_name, resume_checkpoint, load_checkpoint, model_parameters
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler_v2, scheduler_kwargs
from timm.utils import NativeScaler 

# Setup WandB
try:
    import wandb
    has_wandb = True
except ImportError:
    has_wandb = False

# Import Custom Model
try:
    import katransformer
except ImportError:
    print("Warning: 'katransformer' module not found. Ensure it is in the path.")

import torch.nn.functional as F

# --- FIXED FOCAL LOSS (Soft-Target Compatible) ---
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha 
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        inputs: [B, C] logits
        targets: [B] indices OR [B, C] soft-labels (one-hot)
        """
        # Cross Entropy without reduction
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=None)
        pt = torch.exp(-ce_loss)
        
        # Focal term
        focal_term = (1 - pt) ** self.gamma
        
        # Alpha Weighting handling
        if self.alpha is not None:
            if targets.dim() == 1: # Hard targets (indices)
                alpha_t = self.alpha.gather(0, targets)
            else: # Soft targets (Mixup/Cutmix)
                # Tính trọng số trung bình dựa trên xác suất label
                alpha_t = (targets * self.alpha.unsqueeze(0)).sum(dim=1)
            loss = alpha_t * focal_term * ce_loss
        else:
            loss = focal_term * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

# --- AUTO WEIGHT COMPUTATION (Giữ nguyên logic cũ) ---
def compute_auto_weights(dataset, num_classes, method='smooth_power', beta=0.9999, power=0.3):
    import math
    class_counts = [0] * num_classes
    found = False
    
    # Logic tìm class counts (Rút gọn cho ngắn nhưng giữ nguyên logic chính)
    if hasattr(dataset, 'targets'): 
        targets = dataset.targets
        found = True
    elif hasattr(dataset, 'samples'): 
        targets = [s[1] for s in dataset.samples]
        found = True
    elif hasattr(dataset, 'imgs'):
        targets = [s[1] for s in dataset.imgs]
        found = True
    
    if found:
        for t in targets:
            if isinstance(t, torch.Tensor): t = t.item()
            if 0 <= t < num_classes: class_counts[t] += 1
    else:
        # Fallback: Quét thư mục (Nhanh nhất cho ImageFolder)
        if hasattr(dataset, 'data_dir') and os.path.exists(os.path.join(dataset.data_dir, 'train')):
            train_dir = os.path.join(dataset.data_dir, 'train')
            classes = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
            for i, c in enumerate(classes):
                if i < num_classes:
                    c_path = os.path.join(train_dir, c)
                    class_counts[i] = len([f for f in os.listdir(c_path) if os.path.isfile(os.path.join(c_path, f))])

    weights = []
    for count in class_counts:
        if count <= 0: weights.append(1.0)
        else:
            if method == 'inverse': weights.append(1.0 / count)
            elif method == 'effective': weights.append((1.0 - beta) / (1.0 - beta ** count))
            elif method == 'sqrt': weights.append(1.0 / math.sqrt(count))
            elif method == 'log': weights.append(1.0 / math.log(1.1 + count))
            elif method == 'smooth_power': weights.append(1.0 / (count ** power))
    
    weights = torch.FloatTensor(weights)
    weights = weights * (num_classes / weights.sum())
    return weights, class_counts

_logger = logging.getLogger('train')

# --- CONFIG & ARGS ---
config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE', help='YAML config file')

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training on TPU')

# Dataset parameters
group = parser.add_argument_group('Dataset parameters')
parser.add_argument('--data-dir', metavar='DIR', help='path to dataset (root dir)')
parser.add_argument('--dataset', metavar='NAME', default='', help='dataset type + name')
group.add_argument('--train-split', metavar='NAME', default='train', help='dataset train split')
group.add_argument('--val-split', metavar='NAME', default='validation', help='dataset validation split')
parser.add_argument('--train-num-samples', default=None, type=int, metavar='N', help='Manually specify num samples')
parser.add_argument('--val-num-samples', default=None, type=int, metavar='N', help='Manually specify num samples')
group.add_argument('--dataset-download', action='store_true', default=False, help='Allow download of dataset')
group.add_argument('--class-map', default='', type=str, metavar='FILENAME', help='path to class to idx mapping file')
group.add_argument('--input-img-mode', default=None, type=str, help='Dataset image conversion mode')
group.add_argument('--input-key', default=None, type=str, help='Dataset key for input images')
group.add_argument('--target-key', default=None, type=str, help='Dataset key for target labels')

# Model parameters
group = parser.add_argument_group('Model parameters')
group.add_argument('--model', default='resnet50', type=str, metavar='MODEL', help='Name of model to train')
group.add_argument('--pretrained', action='store_true', default=False, help='Start with pretrained version')
group.add_argument('--pretrained-path', default=None, type=str, help='Load this checkpoint as pretrained weights')
group.add_argument('--initial-checkpoint', default='', type=str, metavar='PATH', help='Load this checkpoint after init')
group.add_argument('--resume', default='', type=str, metavar='PATH', help='Resume full model and optimizer state')
group.add_argument('--no-resume-opt', action='store_true', default=False, help='prevent resume of optimizer state')
group.add_argument('--num-classes', type=int, default=None, metavar='N', help='number of label classes')
group.add_argument('--gp', default=None, type=str, metavar='POOL', help='Global pool type')
group.add_argument('--img-size', type=int, default=None, metavar='N', help='Image size')
group.add_argument('--in-chans', type=int, default=None, metavar='N', help='Image input channels')
group.add_argument('--input-size', default=None, nargs=3, type=int, metavar='N N N', help='Input all image dimensions')
group.add_argument('--crop-pct', default=None, type=float, metavar='N', help='Input image center crop percent')
group.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN', help='Override mean pixel value')
group.add_argument('--std', type=float, nargs='+', default=None, metavar='STD', help='Override std deviation')
group.add_argument('--interpolation', default='', type=str, metavar='NAME', help='Image resize interpolation type')
group.add_argument('-b', '--batch-size', type=int, default=128, metavar='N', help='Input batch size for training PER CORE')
group.add_argument('-vb', '--validation-batch-size', type=int, default=None, metavar='N', help='Validation batch size override')
group.add_argument('--channels-last', action='store_true', default=False, help='Use channels_last memory layout')
group.add_argument('--fuser', default='', type=str, help="Select jit fuser")
group.add_argument('--grad-accum-steps', type=int, default=1, metavar='N', help='The number of steps to accumulate gradients')
group.add_argument('--grad-checkpointing', action='store_true', default=False, help='Enable gradient checkpointing')
group.add_argument('--fast-norm', default=False, action='store_true', help='enable experimental fast-norm')
group.add_argument('--model-kwargs', nargs='*', default={}, action=utils.ParseKwargs)
group.add_argument('--head-init-scale', default=None, type=float, help='Head initialization scale')
group.add_argument('--head-init-bias', default=None, type=float, help='Head initialization bias value')

# Scripting
scripting_group = group.add_mutually_exclusive_group()
scripting_group.add_argument('--torchscript', dest='torchscript', action='store_true', help='torch.jit.script the full model')

# Device & distributed
group = parser.add_argument_group('Device parameters')
group.add_argument('--synchronize-step', action='store_true', default=False, help='synchronize end of each step')
parser.add_argument('--device-modules', default=None, type=str, nargs='+', help="Python imports for device backend modules.")

# Optimizer parameters
group = parser.add_argument_group('Optimizer parameters')
group.add_argument('--opt', default='sgd', type=str, metavar='OPTIMIZER', help='Optimizer')
group.add_argument('--opt-eps', default=None, type=float, metavar='EPSILON', help='Optimizer Epsilon')
group.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA', help='Optimizer Betas')
group.add_argument('--momentum', type=float, default=0.9, metavar='M', help='Optimizer momentum')
group.add_argument('--weight-decay', type=float, default=2e-5, help='weight decay')
group.add_argument('--clip-grad', type=float, default=None, metavar='NORM', help='Clip gradient norm')
group.add_argument('--clip-mode', type=str, default='norm', help='Gradient clipping mode')
group.add_argument('--layer-decay', type=float, default=None, help='layer-wise learning rate decay')
group.add_argument('--opt-kwargs', nargs='*', default={}, action=utils.ParseKwargs)

# Learning rate schedule parameters
group = parser.add_argument_group('Learning rate schedule parameters')
group.add_argument('--sched', type=str, default='cosine', metavar='SCHEDULER', help='LR scheduler')
group.add_argument('--sched-on-updates', action='store_true', default=False, help='Apply LR scheduler step on update')
group.add_argument('--lr', type=float, default=None, metavar='LR', help='learning rate')
group.add_argument('--lr-base', type=float, default=0.1, metavar='LR', help='base learning rate')
group.add_argument('--lr-base-size', type=int, default=256, metavar='DIV', help='base learning rate batch size')
group.add_argument('--lr-base-scale', type=str, default='', metavar='SCALE', help='base learning rate scaling')
group.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct', help='learning rate noise')
group.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT', help='learning rate noise limit')
group.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV', help='learning rate noise std-dev')
group.add_argument('--lr-cycle-mul', type=float, default=1.0, metavar='MULT', help='learning rate cycle len multiplier')
group.add_argument('--lr-cycle-decay', type=float, default=0.5, metavar='MULT', help='decay each learning rate cycle')
group.add_argument('--lr-cycle-limit', type=int, default=1, metavar='N', help='learning rate cycle limit')
group.add_argument('--lr-k-decay', type=float, default=1.0, help='learning rate k-decay')
group.add_argument('--warmup-lr', type=float, default=1e-5, metavar='LR', help='warmup learning rate')
group.add_argument('--min-lr', type=float, default=0, metavar='LR', help='lower lr bound')
group.add_argument('--epochs', type=int, default=300, metavar='N', help='number of epochs')
group.add_argument('--epoch-repeats', type=float, default=0., metavar='N', help='epoch repeat multiplier')
group.add_argument('--start-epoch', default=None, type=int, metavar='N', help='manual epoch number')
group.add_argument('--decay-milestones', default=[90, 180, 270], type=int, nargs='+', metavar="MILESTONES", help='decay epoch indices')
group.add_argument('--decay-epochs', type=float, default=90, metavar='N', help='epoch interval to decay LR')
group.add_argument('--warmup-epochs', type=int, default=5, metavar='N', help='epochs to warmup LR')
group.add_argument('--warmup-prefix', action='store_true', default=False, help='Exclude warmup period from decay schedule')
group.add_argument('--cooldown-epochs', type=int, default=0, metavar='N', help='epochs to cooldown LR')
group.add_argument('--patience-epochs', type=int, default=30, metavar='N', help='patience epochs')
group.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE', help='LR decay rate')

# Augmentation & regularization parameters
group = parser.add_argument_group('Augmentation and regularization')
group.add_argument('--no-aug', action='store_true', default=False, help='Disable all training augmentation')
group.add_argument('--train-crop-mode', type=str, default=None, help='Crop-mode in train')
group.add_argument('--scale', type=float, nargs='+', default=[0.08, 1.0], metavar='PCT', help='Random resize scale')
group.add_argument('--ratio', type=float, nargs='+', default=[3. / 4., 4. / 3.], metavar='RATIO', help='Random resize aspect ratio')
group.add_argument('--hflip', type=float, default=0.5, help='Horizontal flip probability')
group.add_argument('--vflip', type=float, default=0., help='Vertical flip probability')
group.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT', help='Color jitter factor')
group.add_argument('--color-jitter-prob', type=float, default=None, metavar='PCT', help='Probability of color jitter')
group.add_argument('--grayscale-prob', type=float, default=None, metavar='PCT', help='Probability of grayscale')
group.add_argument('--gaussian-blur-prob', type=float, default=None, metavar='PCT', help='Probability of gaussian blur')
group.add_argument('--aa', type=str, default=None, metavar='NAME', help='Use AutoAugment policy')
group.add_argument('--aug-repeats', type=float, default=0, help='Number of augmentation repetitions')
group.add_argument('--aug-splits', type=int, default=0, help='Number of augmentation splits')
group.add_argument('--jsd-loss', action='store_true', default=False, help='Enable JSD + CE loss')
group.add_argument('--bce-loss', action='store_true', default=False, help='Enable BCE loss')
group.add_argument('--bce-sum', action='store_true', default=False, help='Sum over classes when using BCE loss')
group.add_argument('--bce-target-thresh', type=float, default=None, help='Threshold for binarizing softened BCE targets')
group.add_argument('--bce-pos-weight', type=float, default=None, help='Positive weighting for BCE loss')
group.add_argument('--reprob', type=float, default=0., metavar='PCT', help='Random erase prob')
group.add_argument('--remode', type=str, default='pixel', help='Random erase mode')
group.add_argument('--recount', type=int, default=1, help='Random erase count')
group.add_argument('--resplit', action='store_true', default=False, help='Do not random erase first split')
group.add_argument('--mixup', type=float, default=0.0, help='mixup alpha')
group.add_argument('--cutmix', type=float, default=0.0, help='cutmix alpha')
group.add_argument('--cutmix-minmax', type=float, nargs='+', default=None, help='cutmix min/max ratio')
group.add_argument('--mixup-prob', type=float, default=1.0, help='Probability of mixup/cutmix')
group.add_argument('--mixup-switch-prob', type=float, default=0.5, help='Probability of switching to cutmix')
group.add_argument('--mixup-mode', type=str, default='batch', help='How to apply mixup/cutmix')
group.add_argument('--mixup-off-epoch', default=0, type=int, metavar='N', help='Turn off mixup after this epoch')
group.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing')
group.add_argument('--train-interpolation', type=str, default='random', help='Training interpolation')
group.add_argument('--drop', type=float, default=0.0, metavar='PCT', help='Dropout rate')
group.add_argument('--drop-connect', type=float, default=None, metavar='PCT', help='Drop connect rate')
group.add_argument('--drop-path', type=float, default=None, metavar='PCT', help='Drop path rate')
group.add_argument('--drop-block', type=float, default=None, metavar='PCT', help='Drop block rate')

# Batch norm parameters
group = parser.add_argument_group('Batch norm parameters')
group.add_argument('--bn-momentum', type=float, default=None, help='BatchNorm momentum override')
group.add_argument('--bn-eps', type=float, default=None, help='BatchNorm epsilon override')
group.add_argument('--sync-bn', action='store_true', help='Enable Synchronized BatchNorm')
group.add_argument('--dist-bn', type=str, default='reduce', help='Distribute BatchNorm stats')
group.add_argument('--split-bn', action='store_true', help='Enable separate BN layers per split')

# EMA
group = parser.add_argument_group('Model EMA parameters')
group.add_argument('--model-ema', action='store_true', default=False, help='Enable tracking moving average of model weights')
group.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='Force ema to be tracked on CPU')
group.add_argument('--model-ema-decay', type=float, default=0.9998, help='Decay factor for model weights moving average')
group.add_argument('--model-ema-warmup', action='store_true', help='Enable warmup for model EMA decay')

# Misc
group = parser.add_argument_group('Miscellaneous parameters')
group.add_argument('--seed', type=int, default=42, metavar='S', help='random seed')
group.add_argument('--worker-seeding', type=str, default='all', help='worker seed mode')
group.add_argument('--log-interval', type=int, default=50, metavar='N', help='how many batches to wait before logging')
group.add_argument('--recovery-interval', type=int, default=0, metavar='N', help='batches to wait before writing recovery')
group.add_argument('--checkpoint-hist', type=int, default=10, metavar='N', help='number of checkpoints to keep')
group.add_argument('-j', '--workers', type=int, default=4, metavar='N', help='how many training processes to use')
group.add_argument('--save-images', action='store_true', default=False, help='save images for debugging')
group.add_argument('--pin-mem', action='store_true', default=False, help='Pin CPU memory')
group.add_argument('--no-prefetcher', action='store_true', default=False, help='disable fast prefetcher')
group.add_argument('--output', default='', type=str, metavar='PATH', help='path to output folder')
group.add_argument('--experiment', default='', type=str, metavar='NAME', help='name of train experiment')
group.add_argument('--eval-metric', default='top1', type=str, metavar='EVAL_METRIC', help='Best metric')
group.add_argument('--tta', type=int, default=0, metavar='N', help='Test/inference time augmentation factor')
group.add_argument('--use-multi-epochs-loader', action='store_true', default=False, help='use multi-epochs-loader')
group.add_argument('--log-wandb', action='store_true', default=False, help='log to wandb')

# Custom Loss Args
group.add_argument('--focal-loss', action='store_true', default=False, help='Use Focal Loss')
group.add_argument('--focal-gamma', type=float, default=2.0, help='Gamma for Focal Loss')
group.add_argument('--auto-weight', action='store_true', default=False, help='Auto compute class weights')
group.add_argument('--auto-weight-method', type=str, default='inverse', choices=['inverse', 'effective', 'sqrt', 'log', 'smooth_power'], help='Weight method')
group.add_argument('--power', type=float, default=0.3, help='Power for smooth_power')
group.add_argument('--effective-beta', type=float, default=0.9999, help='Beta for effective number')
group.add_argument('--class-weights', type=str, default=None, help='Manual class weights')

def _parse_args():
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)
    args = parser.parse_args(remaining)
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text

def _mp_entry(index, args):
    """
    Main Training Loop for XLA Multiprocessing
    index: Rank of the process (0-7 for TPU v3-8/v5e-8)
    """
    utils.setup_default_logging()
    args.rank = index
    args.world_size = xm.xrt_world_size()
    args.device = xm.xla_device()
    args.distributed = args.world_size > 1

    # Only master process logs
    if not xm.is_master_ordinal():
        _logger.setLevel(logging.ERROR)

    if args.device_modules:
        for module in args.device_modules:
            importlib.import_module(module)

    args.prefetcher = not args.no_prefetcher
    args.grad_accum_steps = max(1, args.grad_accum_steps)

    if xm.is_master_ordinal():
        _logger.info(f'Training on TPU. Process {args.rank}, total {args.world_size}.')

    utils.random_seed(args.seed, args.rank)

    if args.fuser:
        utils.set_jit_fuser(args.fuser)
    if args.fast_norm:
        set_fast_norm()

    in_chans = 3
    if args.in_chans is not None:
        in_chans = args.in_chans
    elif args.input_size is not None:
        in_chans = args.input_size[0]

    factory_kwargs = {}
    if args.pretrained_path:
        factory_kwargs['pretrained_cfg_overlay'] = dict(
            file=args.pretrained_path,
            num_classes=-1, 
        )

    model = create_model(
        args.model,
        pretrained=args.pretrained,
        in_chans=in_chans,
        num_classes=args.num_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=args.drop_block,
        global_pool=args.gp,
        bn_momentum=args.bn_momentum,
        bn_eps=args.bn_eps,
        scriptable=args.torchscript,
        checkpoint_path=args.initial_checkpoint,
        **factory_kwargs,
        **args.model_kwargs,
    )

    if args.head_init_scale is not None:
        with torch.no_grad():
            model.get_classifier().weight.mul_(args.head_init_scale)
            model.get_classifier().bias.mul_(args.head_init_scale)
    if args.head_init_bias is not None:
        nn.init.constant_(model.get_classifier().bias, args.head_init_bias)

    if args.num_classes is None:
        assert hasattr(model, 'num_classes'), 'Model must have `num_classes`'
        args.num_classes = model.num_classes

    if args.grad_checkpointing:
        model.set_grad_checkpointing(enable=True)

    if xm.is_master_ordinal():
        _logger.info(f'Model {safe_model_name(args.model)} created, param count:{sum([m.numel() for m in model.parameters()])}')

    data_config = resolve_data_config(vars(args), model=model, verbose=xm.is_master_ordinal())

    num_aug_splits = 0
    if args.aug_splits > 0:
        num_aug_splits = args.aug_splits

    if args.split_bn:
        model = convert_splitbn_model(model, max(num_aug_splits, 2))

    # Move model to TPU
    model.to(args.device)

    # Sync BN for TPU
    if args.sync_bn:
        args.dist_bn = ''
        model = convert_sync_batchnorm(model)
        if xm.is_master_ordinal():
            _logger.info('Converted model to use Synchronized BatchNorm.')

    if args.torchscript:
        model = torch.jit.script(model)

    # Calculate LR
    if not args.lr:
        global_batch_size = args.batch_size * args.world_size * args.grad_accum_steps
        batch_ratio = global_batch_size / args.lr_base_size
        if not args.lr_base_scale:
            on = args.opt.lower()
            args.lr_base_scale = 'sqrt' if any([o in on for o in ('ada', 'lamb')]) else 'linear'
        if args.lr_base_scale == 'sqrt':
            batch_ratio = batch_ratio ** 0.5
        args.lr = args.lr_base * batch_ratio
        if xm.is_master_ordinal():
            _logger.info(f'Learning rate ({args.lr}) calculated from base ({args.lr_base}) and global batch size ({global_batch_size}).')

    optimizer = create_optimizer_v2(model, **optimizer_kwargs(cfg=args), **args.opt_kwargs)

    # Resume
    resume_epoch = None
    if args.resume:
        resume_epoch = resume_checkpoint(
            model,
            args.resume,
            optimizer=None if args.no_resume_opt else optimizer,
            loss_scaler=None, # No scaler needed for XLA/BF16
            log_info=xm.is_master_ordinal(),
        )

    # Model EMA
    model_ema = None
    if args.model_ema:
        model_ema = utils.ModelEmaV3(
            model,
            decay=args.model_ema_decay,
            use_warmup=args.model_ema_warmup,
            device='cpu' if args.model_ema_force_cpu else None,
        )
        if args.resume:
            load_checkpoint(model_ema.module, args.resume, use_ema=True)

    # Datasets
    if args.data and not args.data_dir:
        args.data_dir = args.data
    input_img_mode = 'RGB' if data_config['input_size'][0] == 3 else 'L'
    if args.input_img_mode is not None:
        input_img_mode = args.input_img_mode

    dataset_train = create_dataset(
        args.dataset, root=args.data_dir, split=args.train_split, is_training=True,
        class_map=args.class_map, download=args.dataset_download,
        batch_size=args.batch_size, seed=args.seed, repeats=args.epoch_repeats,
        input_img_mode=input_img_mode, input_key=args.input_key, target_key=args.target_key,
        num_samples=args.train_num_samples,
    )

    if args.val_split:
        dataset_eval = create_dataset(
            args.dataset, root=args.data_dir, split=args.val_split, is_training=False,
            class_map=args.class_map, download=args.dataset_download,
            batch_size=args.batch_size, input_img_mode=input_img_mode,
            input_key=args.input_key, target_key=args.target_key,
            num_samples=args.val_num_samples,
        )

    # Mixup/Cutmix
    collate_fn = None
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_args = dict(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.num_classes
        )
        if args.prefetcher:
            collate_fn = FastCollateMixup(**mixup_args)
        else:
            mixup_fn = Mixup(**mixup_args)

    if num_aug_splits > 1:
        dataset_train = AugMixDataset(dataset_train, num_splits=num_aug_splits)

    train_interpolation = args.train_interpolation
    if args.no_aug or not train_interpolation:
        train_interpolation = data_config['interpolation']

    # Create Loaders (TIMM creates DistributedSampler automatically if args.distributed is True)
    loader_train = create_loader(
        dataset_train,
        input_size=data_config['input_size'],
        batch_size=args.batch_size,
        is_training=True,
        no_aug=args.no_aug,
        re_prob=args.reprob,
        re_mode=args.remode,
        re_count=args.recount,
        re_split=args.resplit,
        train_crop_mode=args.train_crop_mode,
        scale=args.scale,
        ratio=args.ratio,
        hflip=args.hflip,
        vflip=args.vflip,
        color_jitter=args.color_jitter,
        color_jitter_prob=args.color_jitter_prob,
        grayscale_prob=args.grayscale_prob,
        gaussian_blur_prob=args.gaussian_blur_prob,
        auto_augment=args.aa,
        num_aug_repeats=args.aug_repeats,
        num_aug_splits=num_aug_splits,
        interpolation=train_interpolation,
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        distributed=args.distributed, # This needs to be True for TPU
        collate_fn=collate_fn,
        pin_memory=args.pin_mem,
        device=args.device,
        use_prefetcher=args.prefetcher,
        use_multi_epochs_loader=args.use_multi_epochs_loader,
        worker_seeding=args.worker_seeding,
    )

    loader_eval = None
    if args.val_split:
        loader_eval = create_loader(
            dataset_eval,
            input_size=data_config['input_size'],
            batch_size=args.validation_batch_size or args.batch_size,
            is_training=False,
            interpolation=data_config['interpolation'],
            mean=data_config['mean'],
            std=data_config['std'],
            num_workers=args.workers,
            distributed=args.distributed,
            crop_pct=data_config['crop_pct'],
            pin_memory=args.pin_mem,
            device=args.device,
            use_prefetcher=args.prefetcher,
        )

    # Class Weights
    class_weights = None
    if hasattr(args, 'auto_weight') and args.auto_weight:
        computed_weights, class_counts = compute_auto_weights(
            dataset_train, args.num_classes, method=args.auto_weight_method,
            beta=args.effective_beta, power=args.power
        )
        class_weights = computed_weights.to(args.device)
        if xm.is_master_ordinal():
            _logger.info(f"Auto-computed weights: {class_weights}")
    elif hasattr(args, 'class_weights') and args.class_weights:
        weights = [float(w) for w in args.class_weights.split(',')]
        class_weights = torch.FloatTensor(weights).to(args.device)

    # Setup Loss Function
    if hasattr(args, 'focal_loss') and args.focal_loss:
        train_loss_fn = FocalLoss(alpha=class_weights, gamma=args.focal_gamma).to(args.device)
        validate_loss_fn = FocalLoss(alpha=class_weights, gamma=args.focal_gamma).to(args.device)
        if xm.is_master_ordinal():
            _logger.info(f"Using Focal Loss (gamma={args.focal_gamma})")
    elif args.jsd_loss:
        train_loss_fn = JsdCrossEntropy(num_splits=num_aug_splits, smoothing=args.smoothing).to(args.device)
        validate_loss_fn = nn.CrossEntropyLoss().to(args.device)
    elif mixup_active:
        if args.bce_loss:
            train_loss_fn = BinaryCrossEntropy(target_threshold=args.bce_target_thresh, sum_classes=args.bce_sum, pos_weight=args.bce_pos_weight).to(args.device)
        else:
            train_loss_fn = SoftTargetCrossEntropy().to(args.device)
        validate_loss_fn = nn.CrossEntropyLoss().to(args.device)
    elif args.smoothing:
        train_loss_fn = LabelSmoothingCrossEntropy(smoothing=args.smoothing).to(args.device)
        validate_loss_fn = nn.CrossEntropyLoss().to(args.device)
    else:
        train_loss_fn = nn.CrossEntropyLoss(weight=class_weights).to(args.device)
        validate_loss_fn = nn.CrossEntropyLoss().to(args.device)

    # Saver
    eval_metric = args.eval_metric if loader_eval is not None else 'loss'
    decreasing_metric = eval_metric == 'loss'
    best_metric = None
    best_epoch = None
    saver = None
    output_dir = None
    if xm.is_master_ordinal():
        if args.experiment:
            exp_name = args.experiment
        else:
            exp_name = '-'.join([datetime.now().strftime("%Y%m%d-%H%M%S"), safe_model_name(args.model)])
        output_dir = utils.get_outdir(args.output if args.output else './output/train', exp_name)
        saver = utils.CheckpointSaver(
            model=model, optimizer=optimizer, args=args, model_ema=model_ema,
            checkpoint_dir=output_dir, recovery_dir=output_dir,
            decreasing=decreasing_metric, max_history=args.checkpoint_hist
        )

    # WandB (Rank 0 only)
    if xm.is_master_ordinal() and args.log_wandb and has_wandb:
        wandb.init(project="scale-kan-tpu", name=exp_name, config=args)

    # LR Scheduler
    updates_per_epoch = (len(loader_train) + args.grad_accum_steps - 1) // args.grad_accum_steps
    lr_scheduler, num_epochs = create_scheduler_v2(
        optimizer, **scheduler_kwargs(args, decreasing_metric=decreasing_metric),
        updates_per_epoch=updates_per_epoch,
    )
    
    start_epoch = 0
    if args.start_epoch is not None:
        start_epoch = args.start_epoch
    elif resume_epoch is not None:
        start_epoch = resume_epoch
    
    if lr_scheduler is not None and start_epoch > 0:
        if args.sched_on_updates:
            lr_scheduler.step_update(start_epoch * updates_per_epoch)
        else:
            lr_scheduler.step(start_epoch)

    # TRAINING LOOP
    try:
        for epoch in range(start_epoch, num_epochs):
            if hasattr(dataset_train, 'set_epoch'):
                dataset_train.set_epoch(epoch)
            elif args.distributed and hasattr(loader_train.sampler, 'set_epoch'):
                loader_train.sampler.set_epoch(epoch)

            train_metrics = train_one_epoch(
                epoch, model, loader_train, optimizer, train_loss_fn, args,
                lr_scheduler=lr_scheduler, saver=saver, output_dir=output_dir,
                model_ema=model_ema, mixup_fn=mixup_fn,
            )

            if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
                utils.distribute_bn(model, args.world_size, args.dist_bn == 'reduce')

            eval_metrics = None
            if loader_eval is not None:
                eval_metrics = validate(
                    model, loader_eval, validate_loss_fn, args,
                )
                if model_ema is not None and not args.model_ema_force_cpu:
                    if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
                        utils.distribute_bn(model_ema, args.world_size, args.dist_bn == 'reduce')
                    ema_eval_metrics = validate(
                        model_ema, loader_eval, validate_loss_fn, args, log_suffix=' (EMA)',
                    )
                    eval_metrics = ema_eval_metrics

            if output_dir is not None and xm.is_master_ordinal():
                lrs = [param_group['lr'] for param_group in optimizer.param_groups]
                utils.update_summary(
                    epoch, train_metrics, eval_metrics,
                    filename=os.path.join(output_dir, 'summary.csv'),
                    lr=sum(lrs) / len(lrs), write_header=best_metric is None,
                    log_wandb=args.log_wandb and has_wandb,
                )

            if eval_metrics is not None:
                latest_metric = eval_metrics[eval_metric]
            else:
                latest_metric = train_metrics[eval_metric]

            if saver is not None and xm.is_master_ordinal():
                best_metric, best_epoch = saver.save_checkpoint(epoch, metric=latest_metric)

            if lr_scheduler is not None:
                lr_scheduler.step(epoch + 1, latest_metric)

    except KeyboardInterrupt:
        pass

def train_one_epoch(epoch, model, loader, optimizer, loss_fn, args,
                    lr_scheduler=None, saver=None, output_dir=None,
                    model_ema=None, mixup_fn=None):
    
    update_time_m = utils.AverageMeter()
    data_time_m = utils.AverageMeter()
    losses_m = utils.AverageMeter()

    model.train()

    accum_steps = args.grad_accum_steps
    last_accum_steps = len(loader) % accum_steps
    updates_per_epoch = (len(loader) + accum_steps - 1) // accum_steps
    num_updates = epoch * updates_per_epoch
    last_batch_idx = len(loader) - 1
    last_batch_idx_to_accum = len(loader) - last_accum_steps

    data_start_time = update_start_time = time.time()
    optimizer.zero_grad()
    update_sample_count = 0
    
    # --- TPU PARALLEL LOADER ---
    # This wraps the PyTorch loader to feed data to TPU efficiently
    para_loader = pl.ParallelLoader(loader, [args.device])
    
    for batch_idx, (input, target) in enumerate(para_loader.per_device_loader(args.device)):
        last_batch = batch_idx == last_batch_idx
        need_update = last_batch or (batch_idx + 1) % accum_steps == 0
        update_idx = batch_idx // accum_steps
        if batch_idx >= last_batch_idx_to_accum:
            accum_steps = last_accum_steps

        # On TPU, input/target are already on device via ParallelLoader
        if not args.prefetcher:
            # Only if not using prefetcher, ensure mixup
            # Note: ParallelLoader yields device tensors.
            if mixup_fn is not None:
                input, target = mixup_fn(input, target)
        
        if args.channels_last:
            input = input.contiguous(memory_format=torch.channels_last)

        data_time_m.update(accum_steps * (time.time() - data_start_time))

        # Forward
        output = model(input)
        loss = loss_fn(output, target)
        if accum_steps > 1:
            loss /= accum_steps
        
        # Backward
        loss.backward()

        if need_update:
            if args.clip_grad is not None:
                # XLA handles clipping in optimizer step usually, but explicit clip works too
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            
            # --- TPU OPTIMIZER STEP ---
            xm.optimizer_step(optimizer)
            optimizer.zero_grad()
            
            num_updates += 1
            if model_ema is not None:
                model_ema.update(model, step=num_updates)
            
            # Step timing
            time_now = time.time()
            update_time_m.update(time.time() - update_start_time)
            update_start_time = time_now

            if update_idx % args.log_interval == 0:
                lrl = [param_group['lr'] for param_group in optimizer.param_groups]
                lr = sum(lrl) / len(lrl)

                if args.distributed:
                    reduced_loss = xm.mesh_reduce('loss_reduce', loss.item(), np.mean)
                    losses_m.update(reduced_loss * accum_steps, input.size(0))
                else:
                    losses_m.update(loss.item() * accum_steps, input.size(0))

                if xm.is_master_ordinal():
                    _logger.info(
                        f'Train: {epoch} [{update_idx:>4d}/{updates_per_epoch} '
                        f'({100. * update_idx / (updates_per_epoch - 1):>3.0f}%)]  '
                        f'Loss: {losses_m.val:#.3g} ({losses_m.avg:#.3g})  '
                        f'Time: {update_time_m.val:.3f}s, {input.size(0) / update_time_m.val:>7.2f}/s  '
                        f'LR: {lr:.3e}  '
                        f'Data: {data_time_m.val:.3f} ({data_time_m.avg:.3f})'
                    )

            if lr_scheduler is not None:
                lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)

        update_sample_count = 0
        data_start_time = time.time()

    return OrderedDict([('loss', losses_m.avg)])

def validate(model, loader, loss_fn, args, log_suffix=''):
    batch_time_m = utils.AverageMeter()
    losses_m = utils.AverageMeter()
    top1_m = utils.AverageMeter()
    top5_m = utils.AverageMeter()

    model.eval()

    end = time.time()
    last_idx = len(loader) - 1
    
    # --- TPU PARALLEL LOADER ---
    para_loader = pl.ParallelLoader(loader, [args.device])

    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(para_loader.per_device_loader(args.device)):
            last_batch = batch_idx == last_idx
            
            if args.channels_last:
                input = input.contiguous(memory_format=torch.channels_last)

            output = model(input)
            if isinstance(output, (tuple, list)):
                output = output[0]

            loss = loss_fn(output, target)
            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))

            # Reduce across TPU cores
            reduced_loss = xm.mesh_reduce('val_loss', loss.item(), np.mean)
            acc1 = xm.mesh_reduce('val_acc1', acc1.item(), np.mean)
            acc5 = xm.mesh_reduce('val_acc5', acc5.item(), np.mean)

            losses_m.update(reduced_loss, input.size(0))
            top1_m.update(acc1, output.size(0))
            top5_m.update(acc5, output.size(0))

            batch_time_m.update(time.time() - end)
            end = time.time()
            if xm.is_master_ordinal() and (last_batch or batch_idx % args.log_interval == 0):
                log_name = 'Test' + log_suffix
                _logger.info(
                    f'{log_name}: [{batch_idx:>4d}/{last_idx}]  '
                    f'Time: {batch_time_m.val:.3f} ({batch_time_m.avg:.3f})  '
                    f'Loss: {losses_m.val:>7.3f} ({losses_m.avg:>6.3f})  '
                    f'Acc@1: {top1_m.val:>7.3f} ({top1_m.avg:>7.3f})  '
                    f'Acc@5: {top5_m.val:>7.3f} ({top5_m.avg:>7.3f})'
                )

    metrics = OrderedDict([('loss', losses_m.avg), ('top1', top1_m.avg), ('top5', top5_m.avg)])
    return metrics

import numpy as np # Helper for reduction

def main():
    args, args_text = _parse_args()
    # Launch Multiprocessing on TPU (8 cores)
    xmp.spawn(_mp_entry, args=(args,), nprocs=8, start_method='fork')

if __name__ == '__main__':
    main()