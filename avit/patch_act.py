import pdb
from os.path import join

import torch.nn as nn
import torch.nn.functional as F
import torch
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision import datasets, transforms
from torchvision.utils import save_image


class Patch(nn.Module):
    def __init__(self, w, h, is_patch=False, eps=0.1):
        super().__init__()
        if not is_patch:
            w = 224
            h = 224

        self.w = w
        self.h = h
        t = transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)

        self.is_patch = is_patch
        self.min_value = t(torch.zeros(1, 3, 1, 1))[0].permute(1, 2, 0)

        self.adv_patch = nn.Parameter(torch.rand(w, h, 3) - 0.5)

        if is_patch:
            self.max_value = t(torch.ones(1, 3, 1, 1))[0].permute(1, 2, 0)
        else:
            self.max_value = t(eps*torch.ones(1, 3, 1, 1))[0].permute(1, 2, 0)
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

    def save_patch(self, out_dir, size, epoch):
        name = ''
        if epoch < 0:
            name = 'init_' + name
        torch.save(self.adv_patch, join(out_dir, name + 'patch_%d_ep_%03d.pth' % (size, epoch)))
        patch = self.adv_patch.unsqueeze(0).permute(0, 3, 1, 2)
        save_image(patch, join(out_dir, name + 'patch_img_%d_ep_%03d.png' % (size, epoch)))

    def load_patch(self, patch_path):
        patch = torch.load(patch_path)
        self.adv_patch.data = patch
        self.w, self.h, _ = patch.shape

    def forward(self, x):
        self.project()
        B, C, W, H = x.shape
        patch = self.adv_patch.unsqueeze(0).repeat(B, 1, 1, 1).permute(0, 3, 1, 2)
        h_ind = 0
        w_ind = 0
        x[:, :, w_ind:self.w+w_ind, h_ind:self.h+h_ind] = patch

        return x
