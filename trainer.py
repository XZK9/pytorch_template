# coding=utf8
import os
import time

import torch
from torch import optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from .utils import get_optimizer, get_lr_scheduler


def criterion(pred, target, padding_idx=0):
    """compute loss"""
    loss = F.cross_entropy(
            pred[:, :-1, :].reshape(-1, pred.size(-1)),
            target[:, 1:].reshape(-1),
            ignore_index=padding_idx
        )
    return loss


class Trainer:
    def __init__(self, config, model, logger, summarywriter):
        self.config = config
        self.device = torch.device(config.device)
        self.model = model.to(self.device)
        self.logger = logger
        self.writer = summarywriter
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
            if print_log:
                self.logger.debug(step_info)
            epoch_loss += step_loss.item()

        epoch_loss /= len(trainloader)
        # epoch,eloss,lr,mode
        train_epoch_info = {'epoch':epoch,
                            'epoch_loss':epoch_loss,
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
            if print_log:
                self.logger.debug(step_info)
            epoch_loss += step_loss.item()

        epoch_loss /= len(validloader)
        valid_epoch_info = {'epoch':epoch,
                            'epoch_loss':epoch_loss,
                            'epoch_lr':epoch_lr}
        return valid_epoch_info

    def run_epoch(self, epoch, dataloader, mode='train', print_log=True):
        """run one epoch
        Args:
            epoch: epoch index
            dataloader: dataloader
            mode: train or valid
            print_log: print log info or not, default True
        Return:
            epoch info
        """
        if not dataloader:
            return {'epoch': epoch, 'epoch_loss': 0, 'epoch_lr': 0}
        
        if mode == 'train':
            self.model.train()
        elif mode == 'valid':
            self.model.eval()

        epoch_loss = 0
        epoch_lr = self.scheduler.get_last_lr()[0]

        for i_step, (x, y) in enumerate(dataloader):
            x, y = x.to(self.device), y.to(self.device)
            pred_y = self.model(x)
            step_loss = criterion(pred_y, y)
            if mode == 'train':
                self.optimizer.zero_grad()
                step_loss.backward()
                clip_grad_norm_(self.model.parameters(), self.config.clip_grad)
                self.optimizer.step()
            step_info = '%d,%d,%.4f,%.5f,%s' % (
                epoch, i_step, step_loss.item(), epoch_lr, mode)
            if print_log:
                self.logger.debug(step_info)
            epoch_loss += step_loss.item()
        
        epoch_loss /= len(dataloader)
        epoch_info = {'epoch': epoch,
                      'epoch_loss': epoch_loss,
                      'epoch_lr': epoch_lr}
        
        return epoch_info

    
    def _train_epoch(self, epoch, dataloader, print_log=True):
        return self.run_epoch(epoch, dataloader, mode='train', print_log=print_log)

    
    @torch.no_grad()
    def _valid_epoch(self, epoch, dataloader, print_log=True):
        return self.run_epoch(epoch, dataloader, mode='valid', print_log=print_log)

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
                        'config':self.model.config }

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
        """train model, save model and log
        Args:
            trainloader: train dataloader
            validloader: valid dataloader
            print_epoch_log: print train and valid epoch info or not
            print_step_log: print train and valid step info or not
        Return:
            no return
        """
        # save config
        config_path = os.path.join(
            self.config.save_path, 'config.pt')
        torch.save(self.config, config_path)

        self.logger.info('epoch,epoch_loss,lr,mode')
        for epoch in range(self.config.num_epochs):
            train_info = self._train_epoch(epoch, trainloader, print_log=print_step_log)
            valid_info = self._valid_epoch(epoch, validloader, print_log=print_step_log)
            self.scheduler.step()  # update the lr

            train_epoch_info = self.get_epoch_log(train_info, 'train')
            valid_epoch_info = self.get_epoch_log(valid_info, 'valid')

            writer_info = {'train_loss':train_info['epoch_loss'],'valid_loss':valid_info['epoch_loss']}
            self.writer.add_scalars('epoch_loss', writer_info, epoch)
            self.writer.add_scalars('epoch_lr', {'lr':train_info['epoch_lr']}, epoch)

            if print_epoch_log:
                self.logger.info(train_epoch_info)
                self.logger.info(valid_epoch_info)

            if epoch % self.config.save_freq == 0:
                filepath = os.path.join(self.config.save_path, 'model_%03d.pt'%(epoch))
                self.save_model(epoch, filepath)
        self.writer.close()
    
