# coding=utf8
import random
import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR, CosineAnnealingWarmRestarts


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_optimizer(lr, model):
    """get optimizer, default Adam"""
    return Adam(model.parameters(), lr=lr)


def get_lr_scheduler(config, optimizer):
    """get different schedulers
    Args:
        config: config argparser
        optimizer: torch optimizer
    Returns:
        scheduler: torch lr scheduler
    """
    if config.scheduler == 'step':
        return StepLR(
            optimizer=optimizer,
            step_size=config.step_size,
            gamma=config.gamma
        )
    elif config.scheduler == 'cosine_restart':
        return CosineAnnealingWarmRestarts(
            optimizer=optimizer,
            T_0=config.anneal_period,
            T_mult=1,
            eta_min=config.lr_min
        )
    else:
        print('%s is not supported, step will be used!'%(scheduler))