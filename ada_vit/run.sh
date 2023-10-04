
#CUDA_VISIBLE_DEVICES=0,1,2,3 python main_act.py \
python ada_main.py <path to dataset> \
  --model ada_step_t2t_vit_19_lnorm \
  --ada-head \
  --ada-layer \
  --ada-token-with-mlp \
  --flops-dict adavit_ckpt/t2t-19-h-l-tmlp_flops_dict.pth \
  --eval_checkpoint ./adavit_ckpt/ada_step_t2t_vit_19_lnorm-224-adahlt.tar \
  --num-gpu 4 \
  --batch-size 128 \
  --no-aug \
  --amp
