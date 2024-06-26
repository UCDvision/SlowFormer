# SlowFormer: Universal Adversarial Patch for Attack on Compute and Energy Efficiency of Inference Efficient Vision Transformers [CVPR 2024]


This Repository is an official implementation of [SlowFormer](https://arxiv.org/abs/2310.02544).
Our code is based on [AdaVit](https://github.com/MengLcool/AdaViT), [A-ViT](https://github.com/NVlabs/A-ViT), and [ATS](https://adaptivetokensampling.github.io/). 

## Overview
We propose SlowFormer, an adversarial attack to reduce the computation / energy efficiency gains of efficient inference methods for image classification. The inference efficient methods we consider are input dependent - the network is dynamically altered (e.g. with dropout on tokens / transformer blocks) for each input and the level of efficiency is determined by the image. SlowFormer adds a universal adversarial patch to input images that makes the adaptive inference network to increase its compuation on the image, usually to its maximum possible value. We exeriment with three different inference efficient vision transformer methods - A-ViT, ATS and AdaViT and show that all the methods can be successfully attacked. A-ViT is particularly vulnerable to SlowFormer, with nearly 80% of maximum possible increase in compute with a patch of just 2% of image area. The attack on energy can be performed while preserving the task performance or simultaneously attacking it. The overview of the method is shown in the figure below.

![](teaser_SlowFormer3.jpg)

## Requirements

All our experiments use the PyTorch library. Instructions for PyTorch installation can be found [here](https://pytorch.org/). 

## Dataset

We use the ImageNet-1k dataset in our experiments. Download and prepare the dataset using the [PyTorch ImageNet training example code](https://github.com/pytorch/examples/tree/master/imagenet). The dataset path needs to be set in the bash scripts used for training and evaluation.

## Train Patches

We provide the code for our adversarial patch attack on 3 methods: A-ViT,
AdaViT and ATS. We modify the publicly available codebases for these approaches
to include our attack. Training the universal adversarial patch is extremely fast - it typically converges in just 1-2 epochs.  

### A-ViT

```sh
cd avit
bash run.sh
```

### AdaViT

```sh
cd ada_vit
bash run.sh
```

## Citation

If you make use of the code, please cite the following work:
```
@inproceedings{navaneet2023slowformer,
 author = {Navaneet, K L and Koohpayegani, Soroush Abbasi and Sleiman, Essam and Pirsiavash, Hamed},
 title = {SlowFormer: Universal Adversarial Patch for Attack on Compute and Energy Efficiency of Inference Efficient Vision Transformers},
 year = {2023}
}
```

## License

This project is under the MIT license.
