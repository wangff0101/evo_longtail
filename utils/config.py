import argparse
import torch
import os
import random
import numpy as np
from datetime import datetime


def parse_args(run_type='terminal'):
    parser = argparse.ArgumentParser(description='Python Training')

    # Dataset options
    parser.add_argument('--data_dir', default='data/')
    parser.add_argument('--dataset', default='cifar100',
                        help='Dataset: cifar100')
    parser.add_argument('--num_class', type=int,
                        default=100, help='class number')
    parser.add_argument('--imb_ratio', type=int, default=100,
                        help='Imbalance ratio for data')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                        help='Validation set ratio (default: 0.15=15%%, increased for better Few-shot class accuracy estimation)')
    parser.add_argument('--val_from_train_ratio', type=float, default=0.5,
                        help='Ratio of validation samples from training set in mixed sampling (default: 0.5=50%%, i.e., 5%% from train, 5%% from unused)')
    parser.add_argument('--balanced_val', action='store_true',default=True,
                        help='Create balanced validation set (same number of samples per class, matching test set distribution)')
    parser.add_argument('--val_samples_per_class', type=int, default=100,
                        help='Number of validation samples per class for balanced validation set (default: 100, matching test set). If None, calculated from val_ratio')

    # Optimization options
    parser.add_argument('--network', default='resnet34',
                        help='Network: resnet34')
    parser.add_argument('--epochs', default=200, type=int,
                        metavar='N', help='number of total epochs to run')
    parser.add_argument('--batch_size', default=64, type=int,
                        metavar='N', help='train batchsize')

    parser.add_argument('--cur_stage', default='stage1',
                        help='stage1 feature learning, stage2 classifier learning')
    # feature extractor learning parameters
    parser.add_argument('--lr', '--learning-rate', default=0.01,
                        type=float, metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9,
                        type=float, help='SGD momentum')
    parser.add_argument('--wd', default=5e-3, type=float,
                        help='weight decay factor for optimizer')
    parser.add_argument('--nesterov', action='store_true',
                        help="Utilizing Nesterov")

    # classifier learning parameters
    parser.add_argument('--finetune_epoch', default=20, type=int,
                        metavar='N', help='number of total epochs to run')
    parser.add_argument('--finetune_lr', default=0.0005,
                        type=float, help='learnign rate decay')
    parser.add_argument('--finetune_wd', default=0, type=float,
                        help='weight decay factor for optimizer')
    parser.add_argument('--label_smooth', default=0.98,
                        type=float, help='label smoothing')

    # --- MOOSF & Multi-Loss Configuration ---
    parser.add_argument('--losses', nargs='+', default=['los','bcl','aes'],
                        help='List of losses/strategies to use for stage 2 (e.g., los ce bs ldam cedrw kps bcl shike)')
    # LDAM Loss parameters
    parser.add_argument('--ldam_max_m', type=float,
                        default=0.5, help='LDAM loss max margin')
    parser.add_argument('--ldam_s', type=float, default=30.0,
                        help='LDAM loss scaling factor')
    # LDAM-DRW specific parameter
    parser.add_argument('--ldamdrw_reweight_epoch', type=int,
                        default=160, help='LDAM-DRW re-weighting epoch threshold')
    # KPS Loss parameters
    parser.add_argument('--kps_tau', type=float,
                        default=1.0, help='KPS loss temperature')
    # BCL Loss parameters
    parser.add_argument('--bcl_temperature', type=float,
                        default=0.1, help='BCL loss temperature')
    parser.add_argument('--bcl_base_temperature', type=float,
                        default=0.07, help='BCL loss base temperature')
    # SHIKE Loss parameters
    parser.add_argument('--shike_queue_size', type=int,
                        default=8192, help='SHIKE loss queue size')
    parser.add_argument('--shike_feat_dim', type=int, default=512,
                        help='SHIKE loss feature dimension (MUST match backbone output)')
    parser.add_argument('--shike_momentum', type=float,
                        default=0.999, help='SHIKE loss momentum for queue')
    parser.add_argument('--shike_temperature', type=float,
                        default=0.07, help='SHIKE loss temperature')
    # MBACK/MOOSF specific parameters
    parser.add_argument('--tasks', nargs='+', default=None,
                        help='Tasks for MBACK (defaults to --losses if None)')
    parser.add_argument('--pla', action='store_true',
                        help='Enable PLA strategy in MBACK')
    parser.add_argument('--pcg', action='store_true',
                        help='Enable PCGrad strategy in MBACK')
    parser.add_argument('--mgda', action='store_true',
                        help='Enable MGDA strategy in MBACK')
    parser.add_argument('--mgda_mode', type=str, default='none',
                        help='MGDA gradient normalization mode (none, l2, loss, loss+)')
    parser.add_argument('--chs', action='store_true',
                        help='Enable Chebyshev strategy in MBACK')
    # --- Add MOOSF shortcut argument ---
    parser.add_argument('--MOOSF', action='store_true',
                        help='Shortcut to enable common MOOSF strategies (PLA + PCG)')
    # --- End MOOSF & Multi-Loss Configuration ---

    # Pretrained model path
    parser.add_argument('--pretrained_pth', default="best_model_stage1.pth",
                        type=str, help='The pretrained model path')

    # Checkpoints save dir
    parser.add_argument('--out', default='output',
                        help='Directory to output the result')

    # Miscs
    parser.add_argument('--workers', type=int, default=16, help='# workers')
    parser.add_argument('--seed', type=str, default='None', help='manual seed')
    parser.add_argument('--gpu', default="1", type=str,
                        help='id(s) for CUDA_VISIBLE_DEVICES')

    args = parser.parse_args()

    # --- Apply MOOSF shortcut logic ---
    if args.MOOSF:
        args.pcg = True
        args.pla = True
        # args.out_cut = True # --out_cut is not defined, omitting this line.
        print("Info: --MOOSF flag enabled, automatically setting --pcg and --pla to True.")
    # --- End MOOSF shortcut logic ---

    now = datetime.now()
    # Renamed variable
    time_str = f'{now.month}-{now.day}_{now.hour}-{now.minute}-{now.second}'

    # Updated output directory logic
    base_out_dir = args.out if args.out != 'output' else f'output/{args.dataset}_IR={args.imb_ratio}_{args.cur_stage}'
    # Use os.path.join for robustness
    args.out = os.path.join(base_out_dir, time_str)

    # Create the output directory if it doesn't exist
    if not os.path.exists(args.out):
        os.makedirs(args.out, exist_ok=True)  # Use exist_ok=True

    if args.gpu:
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # --- Set default args.tasks based on args.losses if not provided ---
    if args.tasks is None:
        args.tasks = args.losses
    # ------------------------------------------------------------------

    return args


def reproducibility(seed):
    if seed == 'None':
        return
    else:
        seed = int(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        np.random.seed(seed)
        random.seed(seed)
