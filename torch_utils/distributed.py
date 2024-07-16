# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

import os
import re
import shutil
import torch
from . import training_stats

#----------------------------------------------------------------------------

def init():
    if is_slurm():
        distributed_vars = get_slurm_vars()
    else:
        distributed_vars = {
            'MASTER_ADDR': 'localhost',
            'MASTER_PORT': '29500',
            'RANK': '0',
            'LOCAL_RANK': '0',
            'WORLD_SIZE': '1',
        }

    for k in distributed_vars:
        if k not in os.environ:
            os.environ[k] = distributed_vars[k]

    backend = 'gloo' if os.name == 'nt' else 'nccl'
    torch.distributed.init_process_group(backend=backend, init_method='env://')
    n_devices = torch.cuda.device_count()
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    if n_devices == 1 and local_rank > 0:
        local_rank = 0
    torch.cuda.set_device(local_rank)

    sync_device = torch.device('cuda') if get_world_size() > 1 else None
    training_stats.init_multiprocessing(rank=get_rank(), sync_device=sync_device)

def is_slurm():
    srun_exists = shutil.which("srun") is not None
    job_id_exists = os.environ.get("SLURM_JOB_ID") is not None
    return srun_exists or job_id_exists

def get_slurm_vars():
    # Master address
    nodelist = os.environ.get("SLURM_NODELIST", "127.0.0.1")
    master_addr = resolve_root_node_address(nodelist)

    # Master port:
    job_id = os.environ.get("SLURM_JOB_ID")
    if job_id is not None:
        if job_id is not None:
            # use the last 4 numbers in the job id as the id
            default_port = job_id[-4:]
            # all ports should be in the 10k+ range
            default_port = int(default_port) + 15000
        else:
            default_port = 29500

    global_rank = os.environ["SLURM_PROCID"]
    local_rank = os.environ["SLURM_LOCALID"]
    world_size = os.environ["SLURM_NTASKS"]

    return {
        'MASTER_ADDR': master_addr,
        'MASTER_PORT': str(default_port),
        'RANK': global_rank,
        'LOCAL_RANK': local_rank,
        'WORLD_SIZE': world_size,
    }


def resolve_root_node_address(nodes):
    nodes = re.sub(r"\[(.*?)[,-].*\]", "\\1",
                   nodes)  # Take the first node of every node range
    nodes = re.sub(r"\[(.*?)\]", "\\1",
                   nodes)  # handle special case where node range is single number
    return nodes.split(" ")[0].split(",")[0]

#----------------------------------------------------------------------------

def get_rank():
    return torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

#----------------------------------------------------------------------------

def get_world_size():
    return torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1

#----------------------------------------------------------------------------

def should_stop():
    return False

#----------------------------------------------------------------------------

def update_progress(cur, total):
    _ = cur, total

#----------------------------------------------------------------------------

def print0(*args, **kwargs):
    if get_rank() == 0:
        print(*args, **kwargs)

#----------------------------------------------------------------------------
