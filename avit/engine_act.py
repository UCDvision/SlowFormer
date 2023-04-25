# --------------------------------------------------------
# Copyright (C) 2022 NVIDIA Corporation. All rights reserved.
# Nvidia Source Code License-NC
# Official PyTorch implementation of CVPR2022 paper
# A-ViT: Adaptive Tokens for Efficient Vision Transformer
# Hongxu Yin, Arash Vahdat, Jose M. Alvarez, Arun Mallya, Jan Kautz,
# and Pavlo Molchanov
# --------------------------------------------------------

# The following snippet is initially based on:
# https://github.com/facebookresearch/deit
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.

# The code is modified to accomodate A-ViT training

"""
Train and eval functions used in main_act.py for A-ViT training and eval
"""

import time
import math
import pdb
import sys
from typing import Iterable, Optional
import torch
from timm.data import Mixup
from timm.utils import accuracy, ModelEma
from losses import DistillationLoss
import utils
from timm.utils.utils import *
import numpy as np
import os
from utils import RegularizationLoss
import pickle
import heapq, random
from PIL import Image
import cv2

from torchvision.utils import save_image
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from ats_lib.vit_model import DynamicVisionTransformer
from ats_lib.lib_utils import calculate_flops


def save_images(imgs, i):
    imgs = (imgs.permute(0,2,3,1).cpu() * torch.Tensor(IMAGENET_DEFAULT_STD)) \
           + torch.Tensor(IMAGENET_DEFAULT_MEAN)
    imgs = imgs.permute(0,3,1,2)
    save_image(imgs, 'trained_adv_imgs_%d.png' % i)


def train_one_epoch(model: torch.nn.Module, adv_patch, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True, args = None, tf_writer=None):

    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100

    idx = 0
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # temporarily disabled for act
        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            # outputs, rho, cnt = model(samples)
            samples = adv_patch(samples)
            outputs = model(samples)

            # now get the token rhos
            try:
                rho_token = torch.mean(model.module.rho_token)
                # for analysis and keeping track purpose
                cnt_token = model.module.counter_token.data.cpu().numpy()
                # for analysis and keeping track purpose
                cnt_token_diff = (torch.max(model.module.counter_token, dim=-1)[0]-torch.min(model.module.counter_token, dim=-1)[0]).data.cpu().numpy()
            except AttributeError:
                rho_token = torch.mean(model.rho_token)
                # for analysis and keeping track purpose
                cnt_token = model.counter_token.data.cpu().numpy()
                # for analysis and keeping track purpose
                cnt_token_diff = (torch.max(model.counter_token, dim=-1)[0]-torch.min(model.counter_token, dim=-1)[0]).data.cpu().numpy()

        try:
            model.module.batch_cnt += 1
        except AttributeError:
            model.batch_cnt += 1

        # Ponder loss
        ponder_loss_token = torch.mean(rho_token) * args.ponder_token_scale
        loss = -ponder_loss_token

        # Distributional prior
        if args.distr_prior_alpha > 0.:

            # KL loss
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                halting_score_distr = torch.stack(model.module.halting_score_layer)
                halting_score_distr = halting_score_distr / torch.sum(halting_score_distr)
                halting_score_distr = torch.clamp(halting_score_distr, 0.01, 0.99)
                distr_prior_loss = args.distr_prior_alpha * model.module.kl_loss(halting_score_distr.log(), model.module.distr_target)
            else:
                halting_score_distr = torch.stack(model.halting_score_layer)
                halting_score_distr = halting_score_distr / torch.sum(halting_score_distr)
                halting_score_distr = torch.clamp(halting_score_distr, 0.01, 0.99)
                distr_prior_loss = args.distr_prior_alpha * model.kl_loss(halting_score_distr.log(), model.distr_target)

            if distr_prior_loss.item() > 0.:
                loss += distr_prior_loss

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        # loss_scaler(loss, optimizer, clip_grad=max_norm,
        #             parameters=model.parameters(), create_graph=is_second_order)
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=adv_patch.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])


        if 1:
            # update logger
            metric_logger.update(cnt_token_mean=float(np.mean(cnt_token)))
            metric_logger.update(cnt_token_max=float(np.max(cnt_token)))
            metric_logger.update(cnt_token_min=float(np.min(cnt_token)))
            metric_logger.update(cnt_token_diff=float(np.mean(cnt_token_diff)))
            metric_logger.update(ponder_loss_token=ponder_loss_token.item())
            metric_logger.update(remaining_compute=float(np.mean(cnt_token/12.)))

            if args.distr_prior_alpha > 0.:
                metric_logger.update(distri_prior_loss=distr_prior_loss.item())


        if tf_writer is not None and torch.cuda.current_device()==0:
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                if model.module.batch_cnt % print_freq == 0:
                    tf_writer.add_scalar('train/lr', optimizer.param_groups[0]["lr"], model.module.batch_cnt)
                    tf_writer.add_scalar('train/loss', loss_value, model.module.batch_cnt)
                    tf_writer.add_scalar('train/cnt_token_mean', float(np.mean(cnt_token)), model.module.batch_cnt)
                    tf_writer.add_scalar('train/cnt_token_max', float(np.max(cnt_token)), model.module.batch_cnt)
                    tf_writer.add_scalar('train/cnt_token_min', float(np.min(cnt_token)), model.module.batch_cnt)
                    tf_writer.add_scalar('train/avg_cnt_token_diff', float(np.mean(cnt_token_diff)), model.module.batch_cnt)
                    tf_writer.add_scalar('train/ponder_loss_token', ponder_loss_token.item(), model.module.batch_cnt)
                    tf_writer.add_scalar('train/expected_depth_ratio', float(np.mean(cnt_token/12.)), model.module.batch_cnt)
                    if args.distr_prior_alpha > 0.:
                        tf_writer.add_scalar('train/distr_prior_loss', distr_prior_loss.item(), model.module.batch_cnt)
            else:
                if model.batch_cnt % print_freq == 0:
                    tf_writer.add_scalar('train/lr', optimizer.param_groups[0]["lr"], model.batch_cnt)
                    tf_writer.add_scalar('train/loss', loss_value, model.batch_cnt)
                    tf_writer.add_scalar('train/cnt_token_mean', float(np.mean(cnt_token)), model.batch_cnt)
                    tf_writer.add_scalar('train/cnt_token_max', float(np.max(cnt_token)), model.batch_cnt)
                    tf_writer.add_scalar('train/cnt_token_min', float(np.min(cnt_token)), model.batch_cnt)
                    tf_writer.add_scalar('train/avg_cnt_token_diff', float(np.mean(cnt_token_diff)), model.batch_cnt)
                    tf_writer.add_scalar('train/ponder_loss_token', ponder_loss_token.item(), model.batch_cnt)
                    tf_writer.add_scalar('train/expected_depth_ratio', float(np.mean(cnt_token/12.)), model.batch_cnt)
                    if args.distr_prior_alpha > 0.:
                        tf_writer.add_scalar('train/distr_prior_loss', distr_prior_loss.item(), model.batch_cnt)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, adv_patch, model, device, epoch, tf_writer=None, args=None, attack=False):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    cnt_token, cnt_token_diff = None, None
    cnt_token_layer = []

    idx = 0
    i = 0
    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        i += 1
        # compute output
        with torch.cuda.amp.autocast():
            if attack:
                images = adv_patch(images)
                if i < 3:
                    save_images(images, i)
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

        if cnt_token is None:
            try:
                n_token = model.module.counter_token.data.cpu().numpy()
            except AttributeError:
                n_token = model.counter_token.data.cpu().numpy()
            cnt_token = n_token
        else:
            try:
                n_token = model.module.counter_token.data.cpu().numpy()
            except AttributeError:
                n_token = model.counter_token.data.cpu().numpy()
            cnt_token = np.concatenate((cnt_token, n_token))

        # Loop over batch
        for j, item in enumerate(n_token):
            n_token_layer = np.ones((197, 12))
            # Loop over tokens
            for jj, val in enumerate(item):
                n_token_layer[jj, int(val):] = 0
            n_token_layer = n_token_layer.sum(0)
            cnt_token_layer.append(n_token_layer)

        if cnt_token_diff is None:
            try:
                cnt_token_diff = (torch.max(model.module.counter_token, dim=-1)[0] -
                                  torch.min(model.module.counter_token, dim=-1)[0]).data.cpu().numpy()
            except AttributeError:
                cnt_token_diff = (torch.max(model.counter_token, dim=-1)[0] -
                                  torch.min(model.counter_token, dim=-1)[0]).data.cpu().numpy()
        else:
            try:
                cnt_token_diff = np.concatenate((cnt_token_diff, \
                (torch.max(model.module.counter_token, dim=-1)[0]-torch.min(model.module.counter_token, dim=-1)[0]).data.cpu().numpy()))
            except AttributeError:
                cnt_token_diff = np.concatenate((cnt_token_diff, \
                                                 (torch.max(model.counter_token, dim=-1)[0]-torch.min(model.counter_token, dim=-1)[0]).data.cpu().numpy()))

        metric_logger.meters['cnt_token_mean'].update(np.mean(cnt_token), n=batch_size)

    cnt_token_layer = np.array(cnt_token_layer)
    mean_cnt_token_layer = np.mean(cnt_token_layer, 0)
    flops = avgtokens2flops(mean_cnt_token_layer, args.model, device)
    metric_logger.meters['flops'].update(flops, n=1)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
    print('Token mean: {}, max: {}, min: {}, avg_diff: {}'
          .format(np.mean(cnt_token), np.max(cnt_token), np.min(cnt_token), np.mean(cnt_token_diff)))
    print('Avg token per layer: ', mean_cnt_token_layer)
    print('Flops: ', flops)
    if 'tiny' in args.model:
        max_flops = 1.258
        min_flops = 0.873
    elif 'small' in args.model:
        max_flops = 4.608
        min_flops = 3.676
    attack_success = (flops - min_flops) / (max_flops - min_flops) * 100.
    metric_logger.meters['attack_succ'].update(attack_success, n=1)

    if tf_writer is not None and torch.cuda.current_device()==0:
        # writing all values
        tf_writer.add_scalar('test/acc_top1', metric_logger.acc1.global_avg, epoch)
        tf_writer.add_scalar('test/acc_top5', metric_logger.acc5.global_avg, epoch)
        tf_writer.add_scalar('test/loss', metric_logger.loss.global_avg, epoch)
        try:
            tf_writer.add_scalar('test/cnt_token_mean', float(np.mean(cnt_token)), model.module.batch_cnt)
            tf_writer.add_scalar('test/cnt_token_max', float(np.max(cnt_token)), model.module.batch_cnt)
            tf_writer.add_scalar('test/cnt_token_min', float(np.min(cnt_token)), model.module.batch_cnt)
            tf_writer.add_scalar('test/avg_cnt_token_diff', float(np.mean(cnt_token_diff)), model.module.batch_cnt)
            tf_writer.add_scalar('test/expected_depth_ratio', float(np.mean(cnt_token/12)), model.module.batch_cnt)
        except AttributeError:
            tf_writer.add_scalar('test/cnt_token_mean', float(np.mean(cnt_token)), model.batch_cnt)
            tf_writer.add_scalar('test/cnt_token_max', float(np.max(cnt_token)), model.batch_cnt)
            tf_writer.add_scalar('test/cnt_token_min', float(np.min(cnt_token)), model.batch_cnt)
            tf_writer.add_scalar('test/avg_cnt_token_diff', float(np.mean(cnt_token_diff)), model.batch_cnt)
            tf_writer.add_scalar('test/expected_depth_ratio', float(np.mean(cnt_token / 12)), model.batch_cnt)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def hconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    # snippet for merging and visualization
    h_min = max(im.shape[0] for im in im_list)
    im_list_resize = [cv2.resize(im, (int(im.shape[1] * h_min / im.shape[0]), h_min), interpolation=interpolation)
                      for im in im_list]
    return cv2.hconcat(im_list_resize)


def merge_image(im1, im2):
    # snippet for merging and visualization
    h_margin = 54
    v_margin = 80
    im2 = im2[h_margin+5:480-h_margin, v_margin:640-v_margin]
    return hconcat_resize_min([im1, im2])





def avgtokens2flops(n_tokens, arch, device):
    # if True and comm.is_main_process():
        # n_tokens = [int(t / len(num_tokens)) for t in avg_tokens]

    n_tokens = [round(item) for item in n_tokens]
    stages = list(np.arange(12))
    if 'tiny' in arch:
        dummy_model = DynamicVisionTransformer(
            patch_size=16,
            embed_dim=192,
            depth=12,
            num_heads=3,
            mlp_ratio=4,
            qkv_bias=True,
            integrate_attn=True,
            stages=stages,
            num_tokens=n_tokens,
        ).to(device)
    elif 'small' in arch:
        dummy_model = DynamicVisionTransformer(
            patch_size=16,
            embed_dim=384,
            depth=12,
            num_heads=6,
            mlp_ratio=4,
            qkv_bias=True,
            integrate_attn=True,
            stages=stages,
            num_tokens=n_tokens,
        ).to(device)

    dummy_model.eval()
    # Calculate model's flops.
    input_data = torch.rand([1, 3, 224, 224])
    input_data = input_data.to(device)
    flops = calculate_flops(dummy_model, input_data)
    print("GFlops: {}".format(flops))
    return flops
