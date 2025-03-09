# Modified from: https://github.com/fangwei123456/Spike-Element-Wise-ResNet/blob/main/imagenet/utils.py

import os
import torch
import torch.distributed as dist

def is_master():
    if not dist.is_available():  # if not distributed mode, return True
        return True
    elif not dist.is_initialized():  # if distributed mode but not initialized, return True
        return True
    else:  # if distributed mode, return True only when rank is 0
        return dist.get_rank() == 0

def save_on_master(*args, **kwargs):
    if is_master():
        torch.save(*args, **kwargs)
        
def init_dist(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:  # distributed mode
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.local_rank = int(os.environ['LOCAL_RANK'])
        args.distributed = True
    else:  # not distributed mode
        args.distributed = False
        return

    torch.cuda.set_device(args.local_rank)
    print('| distributed init (rank {}, local rank {}): {}'.format(args.rank, args.local_rank, args.dist_url), flush=True)
    dist.init_process_group(backend=args.backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)
    enable_print(is_master())
    
    
def enable_print(is_master):
    '''
    This function disables printing when not in master process
    '''
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def global_meters_all_avg(args, *meters):
    '''
    meters: scalar values of loss/accuracy calculated in each rank
    '''
    tensors = [torch.tensor(meter, device=args.local_rank, dtype=torch.float32) for meter in meters]
    for tensor in tensors:
        # each item of `tensors` is all-reduced starting from index 0 (in-place)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    return [(tensor / args.world_size).item() for tensor in tensors]


def global_meters_all_sum(args, *meters):
    '''
    meters: scalar values calculated in each rank
    '''
    tensors = [torch.tensor(meter, device=args.local_rank, dtype=torch.float32) for meter in meters]
    for tensor in tensors:
        # each item of `tensors` is all-reduced starting from index 0 (in-place)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    
    return [tensor.item() for tensor in tensors]


def global_meters_sum(*meters):
    '''
    meters: scalar values calculated in each rank
    '''
    local_rank = int(os.environ['LOCAL_RANK'])
    tensors = [torch.tensor(meter, device=local_rank, dtype=torch.float32) for meter in meters]
    for tensor in tensors:
        # each item of `tensors` is all-reduced starting from index 0 (in-place)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    if len(tensors) == 1:
        return tensors[0].item()
    else:
        return tuple(tensor.item() for tensor in tensors)