# coding=utf8
import torch
from torch import nn
import torch.nn.functional as F


class Sampler:
    """model sampler"""
    def __init__(self, model, config):
        self.config = config
        self.device = torch.device(config.device)
        self.model = model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def sample(self):
        """sample smiles"""
        samples = self.model.sample()
        return samples
        
    
