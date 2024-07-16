# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Train diffusion-based generative model using the techniques described in the
paper "Elucidating the Design Space of Diffusion-Based Generative Models"."""

import os
import re
import json
import click
import torch
import dnnlib
from torch_utils import distributed as dist
from training import training_loop

import warnings
warnings.filterwarnings('ignore', 'Grad strides do not match bucket view strides') # False warning printed by PyTorch 1.12.

#----------------------------------------------------------------------------
# Parse a comma separated list of numbers or ranges and return a list of ints.
# Example: '1,2,5-10' returns [1, 2, 5, 6, 7, 8, 9, 10]

def parse_int_list(s):
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

#----------------------------------------------------------------------------

@click.command()

# Main options.
@click.option('--outdir',        help='Where to save the results', metavar='DIR',                   type=str, required=True)
@click.option('--data',          help='Path to the dataset', metavar='ZIP|DIR',                     type=str, required=True)
@click.option('--arch',          help='Network architecture', metavar='ddpmpp|ncsnpp|adm',          type=click.Choice(['ddpmpp', 'ncsnpp', 'adm']), default='ncsnpp', show_default=True)
@click.option('--source-pkl',    help='Network weights to distill', metavar='PKL',                  type=str)

# Denoiser Options
@click.option('--denoiser',      help='Method for denoising xt', metavar='net|knn|stf|dsm',         type=click.Choice(['net', 'knn', 'stf', 'dsm']), default='net', show_default=True)
@click.option('--ref-size',      help='Number of reference images to sample', metavar='INT',        type=click.FloatRange(min=1), default=256, show_default=True)
@click.option('--k',             help='KNN neighbourhood size', metavar='INT',                      type=click.IntRange(min=0), default=2048, show_default=True)
@click.option('--quantize',      help='Apply quantization to the index', metavar='BOOL',            type=bool, default=False, show_default=True)
@click.option('--knn-device',    help='Device for knn index.', metavar='gpu|cpu',                   type=click.Choice(['gpu', 'cpu']), default='gpu', show_default=True)

# Loss Hyperparameters.
@click.option('--solver',        help='Solver for PF-ODE Integration', metavar='euler|heun',        type=click.Choice(['euler', 'heun']), default='heun', show_default=True)
@click.option('--sigma-min',     help='Minimum noise level', metavar='FLOAT',                       type=float, default=2E-3, show_default=True)
@click.option('--sigma-max',     help='Maximum noise level', metavar='FLOAT',                       type=float, default=80., show_default=True)
@click.option('--metric',        help='Distance metric', metavar='l2|l1|lpips|huber',               type=click.Choice(['l2', 'l1', 'lpips', 'huber']), default='huber', show_default=True)
@click.option('--hub-cache',     help='Path to cached LPIPS weights', metavar='PATH',               type=str)
@click.option('--schedule',      help='Discretization schedule', metavar='constant|exp|linear|base',type=click.Choice(['constant', 'exp', 'linear', 'base']), default='exp', show_default=True)
@click.option('--s0',            help='Minimum number of discretization steps', metavar='INT',      type=click.IntRange(min=1), default=10, show_default=True)
@click.option('--s1',            help='Maximum number of discretization steps', metavar='INT',      type=click.IntRange(min=1), default=1280, show_default=True)
@click.option('--weighting',     help='Loss weighting method.', metavar='uniform|inverse',          type=click.Choice(['uniform', 'inverse']), default='inverse', show_default=True)
@click.option('--sigma-dist',    help='Distribution for noise levels', metavar='uniform|lognormal|disc', type=click.Choice(['uniform', 'lognormal', 'disc']), default='disc', show_default=True)
@click.option('--p-mean',        help='Sigma dist lognormal mean.', metavar='FLOAT',                type=float, default=-1.1, show_default=True)
@click.option('--p-std',         help='Sigma dist lognormal mean.', metavar='FLOAT',                type=click.FloatRange(min=1E-9), default=2., show_default=True)
@click.option('--huber-c',       help='Constant for Huber Loss', metavar='FLOAT',                   type=click.FloatRange(min=0), default=0.03, show_default=True)

# Hyperparameters.
@click.option('--duration',      help='Training duration', metavar='MIMG',                          type=click.FloatRange(min=0, min_open=True), default=204.8, show_default=True)
@click.option('--batch',         help='Total batch size', metavar='INT',                            type=click.IntRange(min=1), default=512, show_default=True)
@click.option('--batch-gpu',     help='Limit batch size per GPU', metavar='INT',                    type=click.IntRange(min=1))
@click.option('--cbase',         help='Channel multiplier  [default: varies]', metavar='INT',       type=int)
@click.option('--cres',          help='Channels per resolution  [default: varies]', metavar='LIST', type=parse_int_list)
@click.option('--lr',            help='Learning rate', metavar='FLOAT',                             type=click.FloatRange(min=0, min_open=True), default=1e-4, show_default=True)
@click.option('--ema',           help='EMA per-step beta', metavar='FLOAT',                         type=click.FloatRange(min=0), default=0.9999, show_default=True)
@click.option('--dropout',       help='Dropout probability', metavar='FLOAT',                       type=click.FloatRange(min=0, max=1), default=0.3, show_default=True)
@click.option('--xflip',         help='Enable dataset x-flips', metavar='BOOL',                     type=bool, default=True, show_default=True)

# Performance-related.
@click.option('--fp16',          help='Enable mixed-precision training', metavar='BOOL',            type=bool, default=False, show_default=True)
@click.option('--ls',            help='Loss scaling', metavar='FLOAT',                              type=click.FloatRange(min=0, min_open=True), default=1, show_default=True)
@click.option('--bench',         help='Enable cuDNN benchmarking', metavar='BOOL',                  type=bool, default=True, show_default=True)
@click.option('--cache',         help='Cache dataset in CPU memory', metavar='BOOL',                type=bool, default=True, show_default=True)
@click.option('--workers',       help='DataLoader worker processes', metavar='INT',                 type=click.IntRange(min=1), default=1, show_default=True)

# I/O-related.
@click.option('--desc',          help='String to include in result dir name', metavar='STR',        type=str)
@click.option('--nosubdir',      help='Do not create a subdirectory for results',                   is_flag=True)
@click.option('--tick',          help='How often to print progress', metavar='KIMG',                type=click.IntRange(min=1), default=50, show_default=True)
@click.option('--snap',          help='How often to save snapshots', metavar='TICKS',               type=click.IntRange(min=1), default=50, show_default=True)
@click.option('--dump',          help='How often to dump state', metavar='TICKS',                   type=click.IntRange(min=1), default=100, show_default=True)
@click.option('--seed',          help='Random seed  [default: random]', metavar='INT',              type=int)
@click.option('--transfer',      help='Transfer learning from network pickle', metavar='PKL|URL',   type=str)
@click.option('--resume',        help='Resume from previous training state', metavar='PT',          type=str)
@click.option('-n', '--dry-run', help='Print training options and exit',                            is_flag=True)

def main(**kwargs):
    """Train diffusion-based generative model using the techniques described in the
    paper "Elucidating the Design Space of Diffusion-Based Generative Models".

    Examples:

    \b
    # Train DDPM++ model for class-conditional CIFAR-10 using 8 GPUs
    torchrun --standalone --nproc_per_node=8 train.py --outdir=training-runs \\
        --data=datasets/cifar10-32x32.zip --cond=1 --arch=ddpmpp
    """
    opts = dnnlib.EasyDict(kwargs)
    torch.multiprocessing.set_start_method('spawn')
    dist.init()

    if opts.hub_cache is not None:
        torch.hub.set_dir(opts.hub_cache)

    # Initialize config dict.
    c = dnnlib.EasyDict()
    c.dataset_kwargs = dnnlib.EasyDict(
        class_name='training.dataset.ImageFolderDataset', path=opts.data, use_labels=False, xflip=opts.xflip,
        cache=opts.cache)
    c.data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, num_workers=opts.workers, prefetch_factor=2)
    c.network_kwargs = dnnlib.EasyDict()
    c.loss_kwargs = dnnlib.EasyDict()
    c.optimizer_kwargs = dnnlib.EasyDict(class_name='torch.optim.RAdam', lr=opts.lr, betas=[0.9,0.999], eps=1e-8)

    # Validate dataset options.
    try:
        dataset_obj = dnnlib.util.construct_class_by_name(**c.dataset_kwargs)
        dataset_name = dataset_obj.name
        c.dataset_kwargs.resolution = dataset_obj.resolution # be explicit about dataset resolution
        c.dataset_kwargs.max_size = len(dataset_obj) # be explicit about dataset size
        del dataset_obj # conserve memory
    except IOError as err:
        raise click.ClickException(f'--data: {err}')

    # Network architecture.
    if opts.arch == 'ddpmpp':
        c.network_kwargs.update(model_type='SongUNet', embedding_type='positional', encoder_type='standard', decoder_type='standard')
        c.network_kwargs.update(channel_mult_noise=1, resample_filter=[1,1], model_channels=128, channel_mult=[2,2,2])
    elif opts.arch == 'ncsnpp':
        c.network_kwargs.update(model_type='SongUNet', embedding_type='fourier', encoder_type='residual', decoder_type='standard')
        c.network_kwargs.update(channel_mult_noise=2, resample_filter=[1,3,3,1], model_channels=128, channel_mult=[2,2,2])
    else:
        assert opts.arch == 'adm'
        c.network_kwargs.update(model_type='DhariwalUNet', model_channels=192, channel_mult=[1,2,3,4])

    # Preconditioning & loss function.
    c.network_kwargs.class_name = 'training.networks.EDMPrecond'
    c.network_kwargs.sigma_min = opts.sigma_min
    c.loss_kwargs.class_name = 'training.loss.ConsistencyLoss'

    # Network options.
    if opts.cbase is not None:
        c.network_kwargs.model_channels = opts.cbase
    if opts.cres is not None:
        c.network_kwargs.channel_mult = opts.cres
    c.network_kwargs.update(dropout=opts.dropout, use_fp16=opts.fp16)

    # Training options.
    c.total_kimg = max(int(opts.duration * 1000), 1)
    c.ema_beta = opts.ema
    c.update(batch_size=opts.batch, batch_gpu=opts.batch_gpu)
    c.update(loss_scaling=opts.ls, cudnn_benchmark=opts.bench)
    c.update(kimg_per_tick=opts.tick, snapshot_ticks=opts.snap, state_dump_ticks=opts.dump)

    # Random seed.
    if opts.seed is not None:
        c.seed = opts.seed
    else:
        seed = torch.randint(1 << 31, size=[], device=torch.device('cuda'))
        torch.distributed.broadcast(seed, src=0)
        c.seed = int(seed)

    # Transfer learning and resume.
    if opts.transfer is not None:
        if opts.resume is not None:
            raise click.ClickException('--transfer and --resume cannot be specified at the same time')
        c.resume_pkl = opts.transfer
    elif opts.resume is not None:
        match = re.fullmatch(r'training-state-(\d+).pt', os.path.basename(opts.resume))
        if not match or not os.path.isfile(opts.resume):
            raise click.ClickException('--resume must point to training-state-*.pt from a previous training run')
        c.resume_pkl = os.path.join(os.path.dirname(opts.resume), f'network-snapshot-{match.group(1)}.pkl')
        c.resume_kimg = int(match.group(1))
        c.resume_state_dump = opts.resume

    # Load source network.
    if opts.denoiser == 'net' and opts.source_pkl is None:
        raise click.ClickException('--source_net must be specified for network score estimation')
    if opts.source_pkl:
        c.source_pkl = opts.source_pkl

    # Denoiser hyperparameters
    c.denoiser_kwargs = dnnlib.EasyDict()
    if opts.denoiser == 'dsm':
        assert opts.solver == 'euler', 'Only Euler integration is permitted for DSM denoising.'
    c.denoiser_kwargs.update(type=opts.denoiser, ref_size=opts.ref_size, k=opts.k, quantize=opts.quantize,
                             gpu=(opts.knn_device == 'gpu'))

    # Integration hyperparameters.
    c.loss_kwargs = dnnlib.EasyDict()
    training_iterations = opts.duration * 1E6 // opts.batch
    c.loss_kwargs.update(
        solver=opts.solver, sigma_min=opts.sigma_min, sigma_max=opts.sigma_max, metric=opts.metric,
        schedule=opts.schedule, s0=opts.s0, s1=opts.s1, K=training_iterations,
        weight=opts.weighting, sigma_dist=opts.sigma_dist, P_mean=opts.p_mean, P_std=opts.p_std, huber_const=opts.huber_c,
    )

    # Description string.
    dtype_str = 'fp16' if c.network_kwargs.use_fp16 else 'fp32'
    desc = f'{dataset_name:s}-{opts.arch:s}-gpus{dist.get_world_size():d}-batch{c.batch_size:d}-{dtype_str:s}'
    if opts.desc is not None:
        desc += f'-{opts.desc}'

    # Pick output directory.
    if dist.get_rank() != 0:
        c.run_dir = None
    elif opts.nosubdir:
        c.run_dir = opts.outdir
    else:
        prev_run_dirs = []
        if os.path.isdir(opts.outdir):
            prev_run_dirs = [x for x in os.listdir(opts.outdir) if os.path.isdir(os.path.join(opts.outdir, x))]
        prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
        prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
        cur_run_id = max(prev_run_ids, default=-1) + 1
        c.run_dir = os.path.join(opts.outdir, f'{cur_run_id:05d}-{desc}')
        assert not os.path.exists(c.run_dir)

    # Dry run?
    if opts.dry_run:
        print_summary(c, opts)
        dist.print0('Dry run; exiting.')
        return

    print_summary(c, opts)

    # Create output directory.
    dist.print0('Creating output directory...')
    if dist.get_rank() == 0:
        os.makedirs(c.run_dir, exist_ok=True)
        with open(os.path.join(c.run_dir, 'training_options.json'), 'wt') as f:
            json.dump(c, f, indent=2)
        dnnlib.util.Logger(file_name=os.path.join(c.run_dir, 'log.txt'), file_mode='a', should_flush=True)

    # Train.
    training_loop.training_loop(**c)

def print_summary(c, opts):
    "Print out a summary of the training options."
    dist.print0()
    dist.print0('Training options:')
    dist.print0(json.dumps(c, indent=2))
    dist.print0()
    dist.print0(f'Output directory:        {c.run_dir}')
    dist.print0(f'Dataset path:            {c.dataset_kwargs.path}')
    dist.print0(f'Network architecture:    {opts.arch}')
    dist.print0(f'Number of GPUs:          {dist.get_world_size()}')
    dist.print0(f'Batch size:              {c.batch_size}')
    dist.print0(f'Mixed-precision:         {c.network_kwargs.use_fp16}')
    dist.print0()

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
