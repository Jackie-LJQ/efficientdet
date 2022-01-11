#!/bin/bash
NUM_GPU=8
srun --job-name testrun --ntasks=1 \
--partition ALL \
--account rpixel \
--qos premium \
--cpus-per-task=40 \
--mem=1G --gres=gpu:$NUM_GPU \
--time=72:00:00 \
python -m torch.distributed.launch --nproc_per_node=$NUM_GPU \
    train.py /nas/home/biaoye/liujiaqi/datasets/coco/ \
    --model tf_efficientdet_d0 \
    --batch-size 16 \
    --amp \
    --resume logs/train/20220101-002328-tf_efficientdet_d0/checkpoint-124.pth.tar \
    --lr 0.09 \
    --warmup-epochs 5 \
    --sync-bn \
    --opt momentum \
    --fill-color mean \
    --model-ema \
    --no-pretrained-backbone \
    --output logs 

