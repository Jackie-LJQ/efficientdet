#!/bin/bash
NUM_GPU=8
srun --job-name dualbnbk --ntasks=1 \
--partition ALL \
--account rpixel \
--qos premium \
--cpus-per-task=32 \
--mem=1G --gres=gpu:$NUM_GPU \
--time=72:00:00 \
python -m torch.distributed.launch --nproc_per_node=$NUM_GPU \
    train.py /nas/home/biaoye/liujiaqi/datasets/coco/ \
    --model adv_bkbone_efficientdet_d0 \
    --train-type advtrain \
    --dataset coco \
    --batch-size 4 \
    --amp \
    --lr 0.12 \
    --warmup-epochs 5 \
    --sync-bn \
    --lr-noise 0.4 0.9 \
    --opt momentum \
    --fill-color mean \
    --model-ema \
    --model-ema-decay 0.9999 \
    --output logs/dualbn_bkbone \
    --epochs 100 \
|| scontrol requeue $SLURM_JOB_ID
