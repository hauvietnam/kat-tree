#!/bin/bash
# Lưu file này là finetune_kat_custom.sh

DATA_PATH=/kaggle/input/tree-base/base_data
CHECKPOINT_PATH=/kaggle/working/pretain.pth
NUM_GPUS=2

bash ./dist_train.sh $NUM_GPUS $DATA_PATH \
--model kat_base_patch16_224 \
--resume $CHECKPOINT_PATH \
-b 16 \
--num-classes 250 \
--opt adamw \
--lr 2e-5 \
--weight-decay 0.05 \
--epochs 200 \
--sched cosine \
--aa rand-m9-mstd0.5 \
--amp \
--mean 0.485 0.456 0.406 \
--std 0.229 0.224 0.225 \
--patience-epochs 30