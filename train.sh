#!/bin/bash
# training:
# - nocrop
# - normal training
# - data augmentation for rotations / translations
# - random
# - worst of 10

# settings for first two
SPATIAL_CONSTRAINT=$2
OUT_DIR=$3

ATTACK_TYPE="random"
DEVICES=0,1,2,3,4,5,6,7
#DEVICES=0
ARCH="resnet18"
#ARCH=alexnet
if [[ -z "${DATA}" ]]; then
    :
else
    DATA="/scratch/engstrom_scratch/imagenet/"
fi

EXP_NAME=${1}_${2}

if [ "${1}" == "nocrop" ]; then
    DATASET="imagenet_nocrop"
    TRIES=0
    USE_BEST=0
    ADV_TRAIN=0
elif [ "${1}" == "standard" ]; then
    DATASET="imagenet"
    TRIES=0
    USE_BEST=0
    ADV_TRAIN=0
elif [ "${1}" == "random" ]; then
    DATASET="imagenet"
    TRIES=1
    USE_BEST=0
    ADV_TRAIN=1
elif [ "${1}" == "worst10" ]; then
# DATA=/data/theory/robustopt/datasets/imagenet
    DATASET="imagenet"
    ADV_TRAIN=1
    TRIES=10
    USE_BEST=1
else
    exit 1
fi

rm -rf $DATA/$EXP_NAME

CUDA_VISIBLE_DEVICES=$DEVICES python -m robustness.main \
       --dataset $DATASET \
       --adv-train $ADV_TRAIN \
       --tries $TRIES \
       --spatial-constraint $SPATIAL_CONSTRAINT \
       --use-best $USE_BEST \
       --attack-type $ATTACK_TYPE \
       --exp-name $EXP_NAME \
       --out-dir $OUT_DIR \
       --arch $ARCH \
       --data $DATA \
       --batch-size 256
