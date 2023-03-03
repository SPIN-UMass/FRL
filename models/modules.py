"""
Replications of modules from Wortsman et al. SupSup
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

import models.module_util as module_util

from args import args as pargs

StandardConv = nn.Conv2d
StandardBN = nn.BatchNorm2d

class NonAffineBN(nn.BatchNorm2d):
    def __init__(self, dim):
        super(NonAffineBN, self).__init__(dim, affine=False)

class NonAffineNoStatsBN(nn.BatchNorm2d):
    def __init__(self, dim):
        super(NonAffineNoStatsBN, self).__init__(
            dim, affine=False, track_running_stats=False
        )
class MaskConv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # initialize the scores
        self.scores = nn.Parameter(module_util.mask_init(self))

        self.weight.requires_grad = False
            
        # default sparsity
        self.sparsity = pargs.sparsity
        
    def forward(self, x):
        subnet = module_util.GetSubnet.apply(self.scores.abs(), self.sparsity)
        w = self.weight * subnet
        x = F.conv2d(
            x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x



