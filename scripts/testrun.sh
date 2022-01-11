#!/bin/bash
NUM_GPU=1
srun --job-name testrun --ntasks=1 \
--partition ALL \
--account rpixel \
--qos premium \
--cpus-per-task=2 \
--mem=1G --gres=gpu:$NUM_GPU \
--time=72:00:00 \
python -m torch.distributed.launch --nproc_per_node=$NUM_GPU \
    train.py /nas/home/biaoye/liujiaqi/datasets/coco/ \
    --model advprop_efficientdet_d0 \
    --batch-size 5 \
    --amp \
    --lr 0.12 \
    --warmup-epochs 5 \
    --sync-bn \
    --lr-noise 0.4 0.9 \
    --opt momentum \
    --fill-color mean \
    --model-ema \
    --model-ema-decay 0.9999 \
    --output logs/testrun \
    --epochs 200 \
    --train-type advtrain
    # --no-pretrained-backbone \


# backbone_args=dict(drop_path_rate=0.1, norm_layer=auxiliary_bn),
