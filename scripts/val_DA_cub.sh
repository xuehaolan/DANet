#!/bin/sh

cd ../exper/

CUDA_VISIBLE_DEVICES=0 python val_hierarchy.py \
	--arch=vgg_DA \
    --num_gpu=1 \
    --dataset=cub \
    --img_dir=../data/CUB_200_2011/images \
    --num_classes=200 \
    --snapshot_dir=../snapshots/vgg_DA \
    --onehot=False \
