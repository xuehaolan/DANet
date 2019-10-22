#!/bin/sh

cd ../exper/

CUDA_VISIBLE_DEVICES=0 python val_CAM.py \
	--arch=vgg \
    --batch_size=1 \
    --num_gpu=1 \
    --dataset=cub \
    --img_dir=../data/CUB_200_2011/images \
    --num_classes=200 \
    --resume=True \
    --snapshot_dir=../snapshots/vgg_CAM \
    --onehot=False \
