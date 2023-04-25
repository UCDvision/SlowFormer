
#CUDA_VISIBLE_DEVICES=0,1,2,3 python main_act.py \
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port 8001\
  --use_env main_act.py \
  --model avit_tiny_patch16_224 \
  --finetune <checkpoint_path.pth>\
  --data-path <path to dataset> \
  --output_dir <output path>\
  --batch-size 128 \
  --patch-h 64\
  --patch-w 64\
  --lr 0.2 \
  --tensorboard \
  --epochs 1 \
  --gate_scale 10.0 \
  --gate_center 30 \
  --warmup-epochs 0 \
  --ponder_token_scale 0.0005 \
  --distr_prior_alpha 0.001 \
  --is_patch
