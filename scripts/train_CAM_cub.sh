#!/bin/sh

cd ../exper/

CUDA_VISIBLE_DEVICES=0 python train_CAM.py \
	--arch=vgg \
    --epoch=100 \
    --lr=0.001 \
    --batch_size=30 \
    --num_gpu=1 \
    --dataset=cub \
    --img_dir=../data/CUB_200_2011/images \
    --num_classes=200 \
    --resume=False \
    --snapshot_dir=../snapshots/vgg_CAM \
    --log_dir=../log/vgg_CAM \
    --onehot=False \
    --decay_point=80 \

