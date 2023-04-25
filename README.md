# SlowFormer: Universal Adversarial Patch for Attack on Compute and Energy Efficiency of Inference Efficient Vision Transformers

This Repository is an official implementation of SlowFormer.
Our code is based on [AdaVit](https://github.com/MengLcool/AdaViT), [A-ViT](https://github.com/NVlabs/A-ViT), and [ATS](https://adaptivetokensampling.github.io/). 
## Train Patches

### AdaViT

```sh
python3 ada_main.py ../ImageNetOFF/ --model ada_step_t2t_vit_19_lnorm --ada-head --ada-layer --ada-token-with-mlp --flops-dict adavit_ckpt/t2t-19-h-l-tmlp_flops_dict.pth --eval_checkpoint ./adavit_ckpt/ada_step_t2t_vit_19_lnorm-224-adahlt.tar --num-gpu 4 --batch-size 128 --no-aug --amp
```


We provide the code for our adversarial patch attack on 3 methods: A-ViT,
AdaViT and ATS. We modify the publicly available codebases for these approaches
to include our attack.
# SlowFormerComputationAttack
