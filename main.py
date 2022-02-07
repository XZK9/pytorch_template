# coding=utf8
import os
import sys
import time
import logging

import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from config import get_parser
from utils import set_seed
from dataset.dataset import CustomDataset, DataCollator
from dataset.vocab import Vocab
from model.model import Model
from trainer import Trainer
from trainer_ddp import DDP_Trainer
from model.sampler import Sampler


def get_logger(config):
    """get logger"""
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.DEBUG)

    # stream to console
    stream_handler = logging.StreamHandler(sys.stdout)
    #stream_handler.setLevel(level=logging.DEBUG)
    logger.addHandler(stream_handler)

    # stream to file
    log_file = os.path.join(config.save_path, config.log_file)
    file_handler = logging.FileHandler(log_file, 'w')
    file_handler.setLevel(level=logging.INFO)
    logger.addHandler(file_handler)

    return logger


def train_model(config):
    """train model"""
    logger = get_logger(config)
    os.makedirs(config.save_path, exist_ok=True)
    set_seed(config.seed)

    collate_fn = DataCollator()
    trainset = CustomDataset(config.trainset)
    validset = CustomDataset(config.validset)
    trainloader = DataLoader(dataset=trainset, 
                             batch_size=config.batch_size,
                             shuffle=True,
                             num_workers=config.num_workers,
                             collate_fn=collate_fn)
    validloader = DataLoader(dataset=validset,
                             batch_size=config.batch_size,
                             shuffle=False,
                             num_workers=config.num_workers,
                             collate_fn=collate_fn)

    model = Model(config)
    #comment = time.strftime('%m-%d-%H_%M_%S',time.localtime(time.time()))
    comment = 'Layer:%d-DropOut:%.2f'%(config.num_layers, config.dropout)
    log_dir = 'summary/%s'%(comment)
    summarywriter = SummaryWriter(log_dir)
    trainer = Trainer(config, model, logger, summarywriter)
    trainer.fit(trainloader, validloader, print_epoch_log=True, print_step_log=True)
    

def train_ddp(local_rank, config):
    """traind model with multi-GPU"""
    set_seed(config.seed + local_rank)
    dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
        master_ip=config.master_ip, master_port=config.master_port)
    dist.init_process_group(backend='nccl', init_method=dist_init_method, world_size=config.world_size, rank=local_rank)
    device = torch.device('cuda', local_rank)
    collate_fn = DataCollator()
    trainset = CustomDataset(config.trainset)
    validset = CustomDataset(config.validset)
    train_sampler = DistributedSampler(trainset)
    trainloader = DataLoader(dataset=trainset,
                             batch_size=config.batch_size,
                             num_workers=config.num_workers,
                             collate_fn=collate_fn,
                             sampler=train_sampler)
    
    valid_sampler = DistributedSampler(validset)
    validloader = DataLoader(dataset=validset,
                             batch_size=config.batch_size,
                             num_workers=config.num_workers,
                             collate_fn=collate_fn,
                             sampler=valid_sampler)
    print('dataset loaded !')
    model = Model(config).to(device)
    ddp_model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    os.makedirs(config.save_path, exist_ok=True)
    logger = get_logger(config) if local_rank==0 else None
    comment = 'Layer:%d-DropOut:%.2f'%(config.num_layers, config.dropout)
    log_dir = 'summary/%s'%(comment)
    summarywriter = SummaryWriter(log_dir) if local_rank==0 else None

    ddp_trainer = DDP_Trainer(config, ddp_model, local_rank, logger, summarywriter)
    ddp_trainer.fit(trainloader, validloader, print_epoch_log=True, print_step_log=True)
    dist.destroy_process_group()


def sample(config):
    """sample smiles"""
    state_dict = torch.load(config.model_load_path)
    model_config = state_dict['config']
    model_state = state_dict['model']
    model = Model(model_config)
    model.load_state_dict(model_state)
    print('model loaded !')


def run():
    config = get_parser()
    if config.mode == 'train':
        train_model(config)
    elif config.mode == 'sample':
        sample(config)
    elif config.mode == 'train_ddp':
        mp.spawn(train_ddp, nprocs=config.world_size, args=(config,))


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
    run()
