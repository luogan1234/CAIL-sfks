import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
    
    def forward(self, inputs):
        raise NotImplementedError