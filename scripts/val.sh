#!/bin/bash
NUM_GPU=1
srun --job-name val --ntasks=1 \
--partition ALL \
--account rpixel \
--qos premium \
--cpus-per-task=1 \
--mem=1G --gres=gpu:$NUM_GPU \
--time=72:00:00 \
python validate.py /localtion/of/mscoco/ --model tf_efficientdet_d2 --split testdev

