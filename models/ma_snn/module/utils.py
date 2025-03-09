import os
import math
import random
import numpy as np
import torch
from torch import nn

def set_seed(seed):
    # for reproducibility. 
    # note that pytorch is not completely reproducible 
    # https://pytorch.org/docs/stable/notes/randomness.html  
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.initial_seed() # dataloader multi processing 
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return None


def str2bool(v):
    if v == 'True':
        return True
    else:
        return False


def global_avgpool2d(x):
    batch_size = x.shape[0]
    channel_size = x.shape[1]
    return x.reshape(batch_size, channel_size, -1).mean(dim=2)


def winner_take_all(x, sparsity_ratio):
    k = math.ceil(sparsity_ratio * x.shape[1])
    winner_idx = x.topk(k, 1)[1]
    winner_mask = torch.zeros_like(x)
    winner_mask.scatter_(1, winner_idx, 1)
    x = x * winner_mask

    return x, winner_mask


def lr_scheduler(optimizer, epoch, init_lr=0.1, lr_decay_epoch=100):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs.
    :param lr_decay_epoch:
    :param init_lr:
    :param epoch:
    :type optimizer: object
    """

    if epoch % lr_decay_epoch == 0 and epoch > 1:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1

    return optimizer


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


# init method
def paramInit(model, method='xavier'):
    scale = 0.05
    for name, w in model.named_parameters():
        if 'weight' in name:
            if method == 'xavier':
                nn.init.xavier_normal_(w)
            elif method == 'kaiming':
                nn.init.kaiming_normal_(w)
            else:
                nn.init.normal_(w)
                w *= scale
        elif 'bias' in name:
            nn.init.constant_(w, 0)
        else:
            pass