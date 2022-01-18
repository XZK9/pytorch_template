# coding=utf8
import os
import time

import torch
from torch import optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import torch.distributed as dist

from utils import get_optimizer, get_lr_scheduler


def criterion(pred, target, padding_idx=0):
    """compute loss"""
    loss = F.cross_entropy(
            pred[:, :-1, :].reshape(-1, pred.size(-1)),
            target[:, 1:].reshape(-1),
            ignore_index=padding_idx
        )
    return loss


class DDP_Trainer:
    def __init__(self, config, model, local_rank, logger, summarywriter):
        self.model = model
        self.config = config
        self.local_rank = local_rank
        self.logger = logger
        self.writer = summarywriter
        self.device = torch.device('cuda', local_rank)
        self.optimizer = get_optimizer(config.lr, model)
        self.scheduler = get_lr_scheduler(config, self.optimizer)

    def _train_epoch(self, epoch, trainloader, print_log=True):
        """
        Args:
            epoch: epoch index
            train_loader: train dataloader
            print_log: print log or not
        Returns:
            train_epoch_info: train info
        """
        self.model.train()
        epoch_loss = 0
        epoch_lr = self.scheduler.get_last_lr()[0]
        for i_step, (x, y) in enumerate(trainloader):
            x, y = x.to(self.device), y.to(self.device)
            pred_y = self.model(x)
            step_loss = criterion(pred_y, y)
            self.optimizer.zero_grad()
            step_loss.backward()
            clip_grad_norm_(self.model.parameters(), self.config.clip_grad)
            self.optimizer.step()
            step_info = '%d,%d,%.4f,%.5f,train'%(epoch,i_step,step_loss.item(),epoch_lr)
            if self.local_rank == 0 and print_log:
                self.logger.debug(step_info)
            epoch_loss += step_loss.item()

        epoch_loss /= len(trainloader)
        epoch_loss = torch.tensor(epoch_loss).to(self.device)
        dist.all_reduce(epoch_loss, op=dist.ReduceOp.SUM)
        epoch_loss = epoch_loss.detach() / self.config.world_size
        # epoch,eloss,lr,mode
        train_epoch_info = {'epoch':epoch,
                            'epoch_loss':epoch_loss.item(),
                            'epoch_lr':epoch_lr}

        return train_epoch_info

    @torch.no_grad()
    def _valid_epoch(self, epoch, validloader, print_log=True):
        """valid one epoch
        Args:
            epoch: epoch index
            validloader: valid dataloader
            print_log: print log info or not, default True
        Return:
            valid epoch info
        """
        if not validloader:
            return {'epoch':epoch, 'epoch_loss':0, 'epoch_lr':0}
        
        self.model.eval()
        epoch_loss = 0
        epoch_lr = self.scheduler.get_last_lr()[0]
        for i_step, (x, y) in enumerate(validloader):
            x, y = x.to(self.device), y.to(self.device)
            pred_y = self.model(x)
            step_loss = criterion(pred_y, y)
            step_info = '%d,%d,%.4f,%.5f,valid'%(
                epoch,i_step,step_loss.item(),epoch_lr)
            if self.local_rank == 0 and print_log:
                self.logger.debug(step_info)
            epoch_loss += step_loss.item()

        epoch_loss /= len(validloader)
        epoch_loss = torch.tensor(epoch_loss).to(self.device)
        dist.all_reduce(epoch_loss, op=dist.ReduceOp.SUM)
        epoch_loss = epoch_loss.detach() / self.config.world_size
        valid_epoch_info = {'epoch':epoch,
                            'epoch_loss':epoch_loss.item(),
                            'epoch_lr':epoch_lr}
        return valid_epoch_info

    def save_model(self, epoch, filepath):
        """save model
        Args:
            epoch: epoch index
            filepath: model save path
        Return:
            no return
        """
        model_state = { 'model':self.model.state_dict(),
                        'optimizer':self.optimizer.state_dict(),
                        'epoch':epoch,
                        'vocab':self.model.vocab,
                        'config':self.config }

        torch.save(model_state, filepath)

    def get_epoch_log(self, epoch_info, mode):
        """get epoch log
        Args:
            epoch_info: epoch info dict
            mode: train or valid or test
        Returns:
            log line
        """
        info = '%d,%.4f,%.5f,%s'%(
            epoch_info['epoch'],
            epoch_info['epoch_loss'],
            epoch_info['epoch_lr'],
            mode)
        return info

    def fit(self, trainloader, validloader=None, print_epoch_log=True, print_step_log=True):
        """train model in ddp
        Args:
            trainloader: train dataloader
            validloader: valid dataloader
            print_epoch_log: print train and valid epoch info or not
            print_step_log: print train and valid step info or not
        Return:
            no return
        """
        if self.local_rank == 0:
            self.logger.info('epoch,epoch_loss,lr,mode')

        for epoch in range(self.config.num_epochs):
            st = time.time()
            trainloader.sampler.set_epoch(epoch) # shuffle data across gpus, no need for validation
            train_info = self._train_epoch(epoch, trainloader, print_log=print_step_log)
            valid_info = self._valid_epoch(epoch, validloader, print_log=print_step_log)
            train_epoch_info = self.get_epoch_log(train_info, 'train')
            valid_epoch_info = self.get_epoch_log(valid_info, 'valid')
            writer_info = {'train_loss':train_info['epoch_loss'],'valid_loss':valid_info['epoch_loss']}
            dist.barrier()
            if self.local_rank == 0:
                self.writer.add_scalars('epoch_loss', writer_info, epoch)
                self.writer.add_scalars('epoch_lr', {'lr':train_info['epoch_lr']}, epoch)

            if self.local_rank == 0 and print_epoch_log:
                self.logger.info(train_epoch_info)
                self.logger.info(valid_epoch_info)

            self.scheduler.step()  # update the lr
            if (epoch % self.config.save_freq == 0) and (self.local_rank == 0):
                filepath = os.path.join(self.config.save_path, 'model_%03d.pt'%(epoch))
                self.save_model(epoch, filepath)
            dist.barrier()
            et = time.time()
            if self.local_rank == 0:
                print('epoch %d time con: %.2fs'%(epoch, et-st))
        if self.writer:
            self.writer.close()
