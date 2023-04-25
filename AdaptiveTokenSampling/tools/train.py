import pdb

import _init_paths
import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
from pathlib import Path
from timm.data import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import NativeScaler, get_state_dict, ModelEma
from datasets import build_dataset
from engine import train_one_epoch, evaluate
from lib.utils.comm import comm
from samplers import RASampler
from losses import DistillATSLoss, ATSLoss
import utils
from vit import VisionTransformerTeacher, _cfg, checkpoint_filter_fn

from lib.models.vit_model import DynamicVisionTransformer
from lib.models.patch import Patch
from lib.utils.utils import calculate_flops
import math
from tensorboardX import SummaryWriter


def get_args_parser():
    parser = argparse.ArgumentParser(
        "DynamicViT training and evaluation script", add_help=False
    )
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--epochs", default=300, type=int)

    # Model parameters
    parser.add_argument(
        "--arch", default="deit_small", type=str, help="Name of model to train"
    )
    parser.add_argument("--input-size", default=224, type=int, help="images input size")
    parser.add_argument(
        "--distillw", type=float, default=0.5, help="distill rate (default: 0.5)"
    )
    parser.add_argument(
        "--ratiow",
        type=float,
        default=2.0,
        metavar="PCT",
        help="ratio rate (default: 2.0)",
    )
    parser.add_argument(
        "--drop",
        type=float,
        default=0.0,
        metavar="PCT",
        help="Dropout rate (default: 0.)",
    )
    parser.add_argument(
        "--drop-path",
        type=float,
        default=0.1,
        metavar="PCT",
        help="Drop path rate (default: 0.1)",
    )

    parser.add_argument("--model-ema", action="store_true")
    parser.add_argument("--no-model-ema", action="store_false", dest="model_ema")
    parser.set_defaults(model_ema=True)
    parser.add_argument("--model-ema-decay", type=float, default=0.99996, help="")
    parser.add_argument(
        "--model-ema-force-cpu", action="store_true", default=False, help=""
    )
    parser.add_argument("--model-path", default="./deit_small_patch16_224-cd65a155.pth")

    # Optimizer parameters
    parser.add_argument(
        "--opt",
        default="adamw",
        type=str,
        metavar="OPTIMIZER",
        help='Optimizer (default: "adamw"',
    )
    parser.add_argument(
        "--opt-eps",
        default=1e-8,
        type=float,
        metavar="EPSILON",
        help="Optimizer Epsilon (default: 1e-8)",
    )
    parser.add_argument(
        "--opt-betas",
        default=None,
        type=float,
        nargs="+",
        metavar="BETA",
        help="Optimizer Betas (default: None, use opt default)",
    )
    parser.add_argument(
        "--clip-grad",
        type=float,
        default=None,
        metavar="NORM",
        help="Clip gradient norm (default: None, no clipping)",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        metavar="M",
        help="SGD momentum (default: 0.9)",
    )
    parser.add_argument(
        "--weight-decay", type=float, default=0.0, help="weight decay (default: 0.05)"
    )
    # Learning rate schedule parameters
    parser.add_argument(
        "--sched",
        default="cosine",
        type=str,
        metavar="SCHEDULER",
        help='LR scheduler (default: "cosine"',
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-4,
        metavar="LR",
        help="learning rate (default: 5e-4)",
    )
    parser.add_argument(
        "--lr-noise",
        type=float,
        nargs="+",
        default=None,
        metavar="pct, pct",
        help="learning rate noise on/off epoch percentages",
    )
    parser.add_argument(
        "--lr-noise-pct",
        type=float,
        default=0.67,
        metavar="PERCENT",
        help="learning rate noise limit percent (default: 0.67)",
    )
    parser.add_argument(
        "--lr-noise-std",
        type=float,
        default=1.0,
        metavar="STDDEV",
        help="learning rate noise std-dev (default: 1.0)",
    )
    parser.add_argument(
        "--warmup-lr",
        type=float,
        default=1e-6,
        metavar="LR",
        help="warmup learning rate (default: 1e-6)",
    )
    parser.add_argument(
        "--min-lr",
        type=float,
        default=1e-5,
        metavar="LR",
        help="lower lr bound for cyclic schedulers that hit 0 (1e-5)",
    )

    parser.add_argument(
        "--decay-epochs",
        type=float,
        default=30,
        metavar="N",
        help="epoch interval to decay LR",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=0,
        metavar="N",
        help="epochs to warmup LR, if scheduler supports",
    )
    parser.add_argument(
        "--cooldown-epochs",
        type=int,
        default=10,
        metavar="N",
        help="epochs to cooldown LR at min_lr, after cyclic schedule ends",
    )
    parser.add_argument(
        "--patience-epochs",
        type=int,
        default=10,
        metavar="N",
        help="patience epochs for Plateau LR scheduler (default: 10",
    )
    parser.add_argument(
        "--decay-rate",
        "--dr",
        type=float,
        default=0.1,
        metavar="RATE",
        help="LR decay rate (default: 0.1)",
    )

    # Augmentation parameters
    parser.add_argument(
        "--color-jitter",
        type=float,
        default=0.4,
        metavar="PCT",
        help="Color jitter factor (default: 0.4)",
    )
    parser.add_argument(
        "--aa",
        type=str,
        default="rand-m9-mstd0.5-inc1",
        metavar="NAME",
        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)',
    ),
    parser.add_argument(
        "--smoothing", type=float, default=0.1, help="Label smoothing (default: 0.1)"
    )
    parser.add_argument(
        "--train-interpolation",
        type=str,
        default="bicubic",
        help='Training interpolation (random, bilinear, bicubic default: "bicubic")',
    )

    parser.add_argument("--repeated-aug", action="store_true")
    parser.add_argument("--no-repeated-aug", action="store_false", dest="repeated_aug")
    parser.set_defaults(repeated_aug=True)

    # * Random Erase params
    parser.add_argument(
        "--reprob",
        type=float,
        default=0.25,
        metavar="PCT",
        help="Random erase prob (default: 0.25)",
    )
    parser.add_argument(
        "--remode",
        type=str,
        default="pixel",
        help='Random erase mode (default: "pixel")',
    )
    parser.add_argument(
        "--recount", type=int, default=1, help="Random erase count (default: 1)"
    )
    parser.add_argument(
        "--resplit",
        action="store_true",
        default=False,
        help="Do not random erase first (clean) augmentation split",
    )

    # * Mixup params
    parser.add_argument(
        "--mixup",
        type=float,
        default=0.8,
        help="mixup alpha, mixup enabled if > 0. (default: 0.8)",
    )
    parser.add_argument(
        "--cutmix",
        type=float,
        default=1.0,
        help="cutmix alpha, cutmix enabled if > 0. (default: 1.0)",
    )
    parser.add_argument(
        "--cutmix-minmax",
        type=float,
        nargs="+",
        default=None,
        help="cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)",
    )
    parser.add_argument(
        "--mixup-prob",
        type=float,
        default=1.0,
        help="Probability of performing mixup or cutmix when either/both is enabled",
    )
    parser.add_argument(
        "--mixup-switch-prob",
        type=float,
        default=0.5,
        help="Probability of switching to cutmix when both mixup and cutmix enabled",
    )
    parser.add_argument(
        "--mixup-mode",
        type=str,
        default="batch",
        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"',
    )

    # Distillation parameters
    parser.add_argument(
        "--teacher-model",
        default="regnety_160",
        type=str,
        metavar="MODEL",
        help='Name of teacher model to train (default: "regnety_160"',
    )
    parser.add_argument(
        "--teacher-path", type=str, default="./deit_small_patch16_224-cd65a155.pth"
    )
    parser.add_argument(
        "--distillation-type",
        default="none",
        choices=["none", "soft", "hard"],
        type=str,
        help="",
    )
    parser.add_argument("--distillation-alpha", default=0.5, type=float, help="")
    parser.add_argument("--distillation-tau", default=1.0, type=float, help="")

    # * Finetuning params
    parser.add_argument("--finetune", default="", help="finetune from checkpoint")

    # Dataset parameters
    parser.add_argument(
        "--data-path", default="/media/datasets/ILSVRC/", type=str, help="dataset path"
    )
    parser.add_argument(
        "--data-set",
        default="IMNET",
        choices=["CIFAR", "IMNET", "INAT", "INAT19"],
        type=str,
        help="Image Net dataset path",
    )
    parser.add_argument(
        "--inat-category",
        default="name",
        choices=[
            "kingdom",
            "phylum",
            "class",
            "order",
            "supercategory",
            "family",
            "genus",
            "name",
        ],
        type=str,
        help="semantic granularity",
    )

    parser.add_argument(
        "--output_dir", default="", help="path where to save, empty for no saving"
    )
    parser.add_argument(
        "--device", default="cuda", help="device to use for training / testing"
    )
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument(
        "--start_epoch", default=0, type=int, metavar="N", help="start epoch"
    )
    parser.add_argument("--eval", action="store_true", help="Perform evaluation only")
    parser.add_argument(
        "--dist-eval",
        action="store_true",
        default=False,
        help="Enabling distributed evaluation",
    )
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument(
        "--pin-mem",
        action="store_true",
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )
    parser.add_argument("--no-pin-mem", action="store_false", dest="pin_mem", help="")
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument(
        "--world_size", default=1, type=int, help="number of distributed processes"
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
    )
    parser.add_argument(
        "--distill",
        action="store_true",
        default=True,
        help="Enabling distributed evaluation",
    )
    parser.add_argument("--base_rate", type=float, default=0.7)
    parser.add_argument(
        "--integrate_attn",
        action="store_true",
        default=True,
        help="Integrate attention into block 7.",
    )
    parser.add_argument(
        "--use_token_mse",
        action="store_true",
        default=False,
        help="Use token MSE loss.",
    )

    parser.add_argument(
        "--is_patch",
        action="store_true",
        default=False,
    )
    parser.add_argument("--stages", nargs="*", default=[3])
    parser.add_argument("--num_tokens", nargs="*", default=[108])

    return parser


def get_param_groups(model, weight_decay, stages, integrate_attn):
    dst_modules = []
    pre_sampling_groups_decay = []
    post_sampling_groups_decay = []
    pre_sampling_groups_no_decay = []
    post_sampling_groups_no_decay = []
    for name, param in model.named_parameters():
        if "blocks" in name:
            block_num = int(name.split(".")[1])
            if not integrate_attn:
                block_num -= 1
            # ATS modules
            if block_num in stages:
                dst_modules.append(param)

            # Blocks before ATS blocks
            elif block_num < stages[0] and param.requires_grad:
                if len(param.shape) == 1 or name.endswith(".bias"):
                    pre_sampling_groups_no_decay.append(param)
                else:
                    pre_sampling_groups_decay.append(param)

            # Blocks after ATS blocks
            elif block_num > stages[0] and param.requires_grad:
                if len(param.shape) == 1 or name.endswith(".bias"):
                    post_sampling_groups_no_decay.append(param)
                else:
                    post_sampling_groups_decay.append(param)
        elif "cls_token" in name or "pos_embed" in name:
            continue
        elif "patch_embed" in name:
            if len(param.shape) == 1 or name.endswith(".bias"):
                pre_sampling_groups_no_decay.append(param)
            else:
                pre_sampling_groups_decay.append(param)
        else:
            if len(param.shape) == 1 or name.endswith(".bias"):
                post_sampling_groups_no_decay.append(param)
            else:
                post_sampling_groups_decay.append(param)

    return [
        {"params": dst_modules, "weight_decay": weight_decay, "name": "dst_modules"},
        {
            "params": pre_sampling_groups_decay,
            "weight_decay": weight_decay,
            "name": "pre_sampling_groups_decay",
        },
        {
            "params": post_sampling_groups_decay,
            "weight_decay": weight_decay,
            "name": "post_sampling_groups_decay",
        },
        {
            "params": pre_sampling_groups_no_decay,
            "weight_decay": 0.0,
            "name": "pre_sampling_groups_no_decay",
        },
        {
            "params": post_sampling_groups_no_decay,
            "weight_decay": 0.0,
            "name": "post_sampling_groups_no_decay",
        },
    ]


def adjust_learning_rate(
    param_groups,
    init_lr,
    min_lr,
    step,
    max_step,
    warming_up_step=2,
    warmup_predictor=False,
    base_multi=0.1,
    summary_writer=None,
):
    cos_lr = (math.cos(step / max_step * math.pi) + 1) * 0.5
    # cos_lr = min_lr + cos_lr * (init_lr - min_lr)
    # cos_lr = 0.01 * min_lr + cos_lr * (0.01 * init_lr - 0.01 * min_lr)
    backbone_lr = cos_lr = 0.2

    print(
        "## Using lr  %.7f for BACKBONE, cosine lr = %.7f for PREDICTOR"
        % (backbone_lr, cos_lr)
    )

    for param_group in param_groups:
        param_group["lr"] = backbone_lr



def main(args):
    utils.init_distributed_mode(args)

    args.stages = list(map(int, args.stages))
    args.num_tokens = list(map(float, args.num_tokens))
    print(args)

    if args.distillation_type != "none" and args.finetune and not args.eval:
        raise NotImplementedError("Finetuning with distillation not yet supported")

    writer_dict = {
        "writer": SummaryWriter(logdir=args.output_dir),
        "train_global_steps": 0,
        "valid_global_steps": 0,
    }

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    cudnn.benchmark = True

    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    dataset_val, _ = build_dataset(is_train=False, args=args)

    if True:  # args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        if args.repeated_aug:
            sampler_train = RASampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        else:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print(
                    "Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. "
                    "This will slightly alter validation results as extra duplicate entries are added to achieve "
                    "equal num of samples per-process."
                )
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False
            )
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=512,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0.0 or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup,
            cutmix_alpha=args.cutmix,
            cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob,
            switch_prob=args.mixup_switch_prob,
            mode=args.mixup_mode,
            label_smoothing=args.smoothing,
            num_classes=args.nb_classes,
        )
    else:
        print("Attention: mixup/cutmix are not used")

    base_rate = args.base_rate
    KEEP_RATE = [base_rate, base_rate ** 2, base_rate ** 3]

    # ---------------------
    # DeiT-Small
    # ---------------------

    print("Creating model: {}".format(args.arch))
    if len(args.num_tokens) == 1:
        num_tokens = args.num_tokens * len(args.stages)
    else:
        num_tokens = args.num_tokens
    model = DynamicVisionTransformer(
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        integrate_attn=args.integrate_attn,
        stages=args.stages,
        num_tokens=num_tokens,
    )

    model_path = args.model_path
    teacher_model_path = args.teacher_path
    checkpoint = torch.load(model_path, map_location="cpu")
    ckpt = checkpoint["model"]
    if not args.eval or args.model_path == "./deit_small_patch16_224-cd65a155.pth":
        ckpt = checkpoint_filter_fn(
            checkpoint,
            model,
            integrate_attn=args.integrate_attn,
            stages=args.stages,
        )
    model.default_cfg = _cfg()
    missing_keys, unexpected_keys = model.load_state_dict(ckpt, strict=False)
    print("Missing keys = ", missing_keys)
    print("Unexpected keys = ", unexpected_keys)
    print("Successfully loaded the pre-trained weights: ", model_path)

    # Knowledge distillation
    if args.distill and not args.eval:
        print("Distillation + Token Sampling Mode ...")
        teacher_ckpt = torch.load(teacher_model_path, map_location="cpu")["model"]
        # Teacher model
        model_t = VisionTransformerTeacher(
            patch_size=16,
            embed_dim=384,
            depth=12,
            num_heads=6,
            mlp_ratio=4,
            qkv_bias=True,
        )
        model_t.load_state_dict(teacher_ckpt, strict=False)
        model_t.to(device)
        print("Successfully loaded the pre-trained weights for the teacher model.")


    # Fine-tune from a checkpoint
    if args.finetune:
        if args.finetune.startswith("https"):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location="cpu", check_hash=True
            )
        else:
            checkpoint = torch.load(args.finetune, map_location="cpu")

        checkpoint_model = checkpoint["model"]
        state_dict = model.state_dict()
        for k in ["head.weight", "head.bias", "head_dist.weight", "head_dist.bias"]:
            if (
                k in checkpoint_model
                and checkpoint_model[k].shape != state_dict[k].shape
            ):
                print("Removing key {} from the pre-trained checkpoint.".format(k))
                del checkpoint_model[k]

        # Interpolate positional embedding.
        pos_embed_checkpoint = checkpoint_model["pos_embed"]
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        # only the position tokens are interpolated
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        pos_tokens = pos_tokens.reshape(
            -1, orig_size, orig_size, embedding_size
        ).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode="bicubic", align_corners=False
        )
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        checkpoint_model["pos_embed"] = new_pos_embed

        model.load_state_dict(checkpoint_model, strict=False)

    model.to(device)

    # Calculate model's flops.
    input_data = torch.rand([1, 3, 224, 224])
    input_data = input_data.to(device)
    flops = calculate_flops(model, input_data)
    print("GFlops: {}".format(flops))

    # Maintain moving averages of the trained parameters
    model_ema = None
    if args.model_ema:
        # ** Create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper.
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device="cpu" if args.model_ema_force_cpu else "",
            resume="",
        )

    adv_patch = Patch(w=64, h=64, is_patch=args.is_patch)
    adv_patch.to(device)


    # Distributed training.
    model_without_ddp = model
    if args.distributed:
        adv_patch = torch.nn.parallel.DistributedDataParallel(
            adv_patch, device_ids=[args.gpu], find_unused_parameters=True
        )
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True
        )
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of parameters:", n_parameters)

    # linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 512.0
    # args.lr = linear_scaled_lr

    opt_args = dict(lr=args.lr, weight_decay=args.weight_decay)
    if hasattr(args, "opt_eps") and args.opt_eps is not None:
        opt_args["eps"] = args.opt_eps
    if hasattr(args, "opt_betas") and args.opt_betas is not None:
        opt_args["betas"] = args.opt_betas

    # parameter_group = get_param_groups(
    #     model_without_ddp,
    #     args.weight_decay,
    #     stages=args.stages,
    #     integrate_attn=args.integrate_attn,
    # )

    # optimizer = torch.optim.AdamW(parameter_group, **opt_args)
    loss_scaler = NativeScaler()

    criterion = LabelSmoothingCrossEntropy()

    if args.mixup > 0.0:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    if args.distill and not args.eval:
        criterion = DistillATSLoss(
            teacher_model=model_t,
            base_criterion=criterion,
            distill_weight=args.distillw,
            clf_weight=1.0,
            mse_token=args.use_token_mse,
            print_mode=True,
        )
    elif not args.eval:
        criterion = ATSLoss(base_criterion=criterion)

    output_dir = Path(args.output_dir)

    if args.resume:
        if args.resume.startswith("https"):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location="cpu", check_hash=True
            )
        else:
            checkpoint = torch.load(args.resume, map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"])
        if not args.eval and "optimizer" in checkpoint and "epoch" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
            # lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint["epoch"] + 1
            if args.model_ema:
                utils._load_checkpoint_for_ema(model_ema, checkpoint["model_ema"])
            if "scaler" in checkpoint:
                loss_scaler.load_state_dict(checkpoint["scaler"])








    for m,p in model.named_parameters():

        p.requires_grad = False


    print("LR is {:.2f}".format(args.lr))
    optimizer = torch.optim.AdamW(adv_patch.parameters(), lr=args.lr, weight_decay=0.0)

    # test_stats = evaluate(data_loader_val, adv_patch, model, device, writer_dict, attack=False)
    # print(
    #     f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%"
    # )

    print(f"Start training for {args.epochs} epochs")

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        adjust_learning_rate(
            optimizer.param_groups,
            args.lr,
            args.min_lr,
            epoch,
            args.epochs,
            warmup_predictor=False,
            warming_up_step=0,
            base_multi=0.1,
            summary_writer=writer_dict,
        )



        train_stats = train_one_epoch(
            model,
            adv_patch,
            criterion,
            data_loader_train,
            # data_loader_val,
            optimizer,
            device,
            epoch,
            loss_scaler,
            writer_dict,
            args.clip_grad,
            model_ema,
            mixup_fn,
            set_training_mode=args.finetune
            == "",  # keep in eval mode during finetuning
        )

        test_stats = evaluate(data_loader_val, adv_patch, model, device, writer_dict, attack=True)
        print(
                f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%"
            )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "DynamicViT training and evaluation script", parents=[get_args_parser()]
    )
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
