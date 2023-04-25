"""
Train and eval functions used in main.py
"""
import math
import pdb
import sys
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma
from torch.nn.parallel.distributed import DistributedDataParallel
import torch.nn.functional as F

from lib.models.vit_model import DynamicVisionTransformer
from lib.utils.utils import calculate_flops
from losses import DistillationLoss
import utils

import random
from lib.utils.comm import comm
import pdb

class AdvLoss(torch.nn.Module):
    def __init__(self):
        super(AdvLoss, self).__init__()
        self.crossent = torch.nn.CrossEntropyLoss()
        self.target_class = 7

    def forward(self, x, target=None):
        # pdb.set_trace()
        if target is None:
            B = x.shape[0]
            target = torch.ones(B, dtype=torch.long).to(x.device) * self.target_class
            return self.crossent(x, target)
        else:
            return -self.crossent(x, target)

class clsLoss(torch.nn.Module):
    def __init__(self):
        super(clsLoss, self).__init__()
        self.mse = torch.nn.MSELoss()
    def forward(self, qk):
        # pdb.set_trace()
        q,k = qk
        cls = q[:, :, :1, :]
        k = k.detach().clone()[:,:,1:,:]
        # k = k.mean(dim=-2)

        # pdb.set_trace()
        sim = (cls @ k)

        loss = ((k-cls)**2).mean()
        # loss = (cls**2).mean()
        # loss = loss.sum(dim=(-1, -2)).mean()
        return loss


class HLoss(torch.nn.Module):
    def __init__(self):
        super(HLoss, self).__init__()
        self.mse = torch.nn.MSELoss()
    def forward(self, x):
        # pdb.set_trace()
        x = x[:,:,1:]
        # pdb.set_trace()
        _,H,N = x.shape
        target = torch.ones_like(x).to(x.device) / N
        # target = x.detach().clone().mean(dim=-1, keepdim=True).expand_as(x)
        # loss = torch.abs(x - target)
        loss = (x - target)**2
        loss = loss.sum(dim=(-1,-2)).mean()
        return loss





        # x = x / 0.5
        # b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        # b = 1.0 * b.sum(dim=-1)

        return b.mean()




def train_one_epoch(
    model: torch.nn.Module,
    adv_patch,
    criterion: DistillationLoss,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    writer_dict,
    max_norm: float = 0,
    model_ema: Optional[ModelEma] = None,
    mixup_fn: Optional[Mixup] = None,
    set_training_mode=True,
):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)
    print_freq = 100
    adv_criterion = HLoss()
    cls_criterion = clsLoss()
    reg_adv_criterion = AdvLoss()

    losses = []
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            # pdb.set_trace()

            x = adv_patch(samples)
            outputs = model(x)
            cls_attn = outputs[-2]
            qk_list = outputs[-1]
            alpha = 0.0
            loss = alpha * criterion(samples, outputs[:-2], targets)

            # loss = reg_adv_criterion(outputs[:-2][0], target=targets)

            # pdb.set_trace()
            # loss += adv_criterion(cls_attn[0])
            loss_coef = 1 / len(cls_attn)
            loss_coef = [1.0, 0.2, 0.05, 0.01, 0.005, 0.005, 0.005]
            # loss_coef = [1.0, 0.1, 0.01, 0.01, 0.005, 0.005, 0.005]
            # loss_coef = [1.0, 0.2, 0.05, 0.05, 0.01, 0.01, 0.01]
            # loss_coef = [1.0, 0.2, 0.01, 0.01, 0.01, 0.01, 0.01]
            loss_list = []
            for i in range(len(cls_attn)):
                # loss += (5**(-i))*adv_criterion(cls_attn[i])
                # loss += cls_criterion(qk_list[i])
                loss += loss_coef[i]*adv_criterion(cls_attn[i])
                # loss_list.append(adv_criterion(cls_attn[i]).item())

        # print(loss_list)

        loss_value = loss.item()
        losses += [loss_value]



        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = (
            hasattr(optimizer, "is_second_order") and optimizer.is_second_order
        )
        loss_scaler(
            loss,
            optimizer,
            clip_grad=max_norm,
            parameters=adv_patch.parameters(),
            create_graph=is_second_order,
        )

        torch.cuda.synchronize()
        # adv_patch.module.project()
        if model_ema is not None:
            model_ema.update(model)

        if writer_dict and comm.is_main_process():
            writer = writer_dict["writer"]
            global_steps = writer_dict["train_global_steps"]
            writer.add_scalar("train_step_loss", loss_value, global_steps)
            writer_dict["train_global_steps"] = global_steps + 1

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        # break
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    epoch_loss = sum(losses) / len(losses)
    if writer_dict and comm.is_main_process():
        writer = writer_dict["writer"]
        writer.add_scalar("train_epoch_loss", epoch_loss, epoch)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, adv_patch, model, device, writer_dict, attack=False):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    # switch to evaluation mode
    model.eval()
    num_tokens = []
    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            x = images
            if attack:

                x = adv_patch(x)

            output, _, _, _, policies, cls_attn, qk = model(x)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        num_tokens.append([p.sum().cpu().numpy() / p.shape[0] for p in policies])

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
        # break
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    avg_tokens = num_tokens[0]
    for i in range(1, len(num_tokens)):
        for j in range(len(avg_tokens)):
            avg_tokens[j] += num_tokens[i][j]

    str_avg_tokens = [str(t / len(num_tokens)) for t in avg_tokens]
    str_avg_tokens = ",".join(str_avg_tokens)
    print(
        "* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f} Avg Tokens {avg_tokens}".format(
            top1=metric_logger.acc1,
            top5=metric_logger.acc5,
            losses=metric_logger.loss,
            avg_tokens=str_avg_tokens,
        )
    )

    if True and comm.is_main_process():
        n_tokens = [int(t / len(num_tokens)) for t in avg_tokens]
        if isinstance(model, DistributedDataParallel):
            stages = model.module.stages
            model_wo_dd = model.module
        else:
            stages = model.stages
            model_wo_dd = model

        if isinstance(model_wo_dd, DynamicVisionTransformer):
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
        elif isinstance(model_wo_dd, DynamicLVViT):
            dummy_model = DynamicLVViT(
                patch_size=16,
                embed_dim=384,
                depth=16,
                num_heads=6,
                mlp_ratio=3.0,
                p_emb="4_2",
                skip_lam=2.0,
                return_dense=True,
                mix_token=True,
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

    if writer_dict and comm.is_main_process():
        writer = writer_dict["writer"]
        global_steps = writer_dict["valid_global_steps"]
        writer.add_scalar("valid_loss", metric_logger.loss.global_avg, global_steps)
        writer.add_scalar("valid_acc1", metric_logger.acc1.global_avg, global_steps)
        writer.add_scalar("valid_acc5", metric_logger.acc5.global_avg, global_steps)
        writer_dict["valid_global_steps"] = global_steps + 1

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
