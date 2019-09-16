#!/bin/bash
# training:
# - nocrop
# - normal training
# - data augmentation for rotations / translations
# - random
# - worst of 10

# settings for first two
ORIG_ATTACK=$1
ORIG_SPATIAL_CONSTRAINT=$2
ORIG_OUT_DIR=$3
EVAL_OUT_DIR=$4
EVAL_ATTACK=$5

DEVICES=0,1,2,3,4,5,6,7

ARCH="resnet18"
DATASET="imagenet"

if [[ -z "${DATA}" ]]; then
    :
else
    DATA="/scratch/engstrom_scratch/imagenet/"
fi


EXP_NAME=${ORIG_ATTACK}_${ORIG_SPATIAL_CONSTRAINT}
DATASET="imagenet"
BATCH_SIZE=1024
ATTACK_TYPE="random"


RESUME_PATH=${ORIG_OUT_DIR}/${EXP_NAME}/checkpoint.pt.best

if [ "${EVAL_ATTACK}" == "standard" ]; then
    TRIES=0
    USE_BEST=0
elif [ "${EVAL_ATTACK}" == "random" ]; then
    TRIES=1
    USE_BEST=0
elif [ "${EVAL_ATTACK}" == "worst10" ]; then
    TRIES=10
    USE_BEST=1
elif [ "${EVAL_ATTACK}" == "grid" ]; then
    ATTACK_TYPE="grid"
    TRIES=1
    USE_BEST=1 # doesnt matter
    BATCH_SIZE=8
else
    exit 1
fi

rm -rf $DATA/$EXP_NAME

CUDA_VISIBLE_DEVICES=$DEVICES python -m robustness.main \
       --dataset $DATASET \
       --eval-only 1 \
       --tries $TRIES \
       --spatial-constraint 30 \
       --use-best $USE_BEST \
       --attack-type $ATTACK_TYPE \
       --out-dir $EVAL_OUT_DIR \
       --arch $ARCH \
       --data $DATA \
       --batch-size $BATCH_SIZE \
       --resume $RESUME_PATH \
       --adv-train 1 \
       --adv-eval 1
