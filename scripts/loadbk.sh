#!/bin/bash
NUM_GPU=4
srun --job-name loadbk --ntasks=1 \
--partition ALL \
--account rpixel \
--qos premium \
--cpus-per-task=20 \
--mem=1G --gres=gpu:$NUM_GPU \
--time=72:00:00 \
python -m torch.distributed.launch --nproc_per_node=$NUM_GPU \
    train.py /nas/home/biaoye/liujiaqi/datasets/coco/ \
    --model pretrainBN_efficientdet_d0 \
    --batch-size 20 \
    --amp \
    --lr 0.12 \
    --warmup-epochs 5 \
    --sync-bn \
    --lr-noise 0.4 0.9 \
    --opt momentum \
    --fill-color mean \
    --model-ema \
    --model-ema-decay 0.9999 \
    --output logs/pretrainbk \
    --epochs 200
    # --resume logs/pretrainbk/train/20220110-135823-pretrainBN_efficientdet_d0/last.pth.tar \
    # --no-pretrained-backbone \

