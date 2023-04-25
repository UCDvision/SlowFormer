import pdb

import torch.nn as nn
import torch.nn.functional as F
import torch
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision import datasets, transforms
import random

class Patch(nn.Module):
    def __init__(self, w, h, is_patch=False, eps=0.1):
        super().__init__()
        if not is_patch:
            w = 224; h = 224;

        self.adv_patch = nn.Parameter(torch.rand(w, h, 3) - 0.5)

        self.w = w
        self.h = h
        t = transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)

        self.is_patch = is_patch
        self.min_value = t(torch.zeros(1, 3, 1, 1))[0].permute(1,2,0)

        if is_patch:
            self.max_value = t(torch.ones(1, 3, 1, 1))[0].permute(1,2,0)
        else:
            self.max_value = t(eps*torch.ones(1, 3, 1, 1))[0].permute(1, 2, 0)
        self.patch_shift = 0
        
        self.project()

    def project(self,):

        weights = self.adv_patch.data
        min_value = self.min_value.expand_as(self.adv_patch).to(self.adv_patch.device)
        max_value = self.max_value.expand_as(self.adv_patch).to(self.adv_patch.device)
        weights = torch.min(weights, max_value)
        weights = torch.max(weights, min_value)
        weights = (((weights - min_value)/(max_value - min_value))*255).floor()/255.0
        weights = (weights * (max_value - min_value)) + min_value

        self.adv_patch.data = weights


    def forward(self, x):
        self.project()
        B, C, W, H = x.shape
        patch = self.adv_patch.unsqueeze(0).repeat(B, 1, 1, 1).permute(0, 3, 1, 2)
        if self.is_patch:
            x_clone = torch.clone(x)
            x_clone[:, :, self.patch_shift:self.w+self.patch_shift, self.patch_shift:self.h+self.patch_shift] = patch
        else:
            x = x + patch

        return x_clone




