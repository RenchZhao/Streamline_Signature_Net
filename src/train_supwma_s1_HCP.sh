#!/bin/bash

## Stage 1 training 
s1_epoch=20
s1_opt='LARS'
s1_lr=1e-3
s1_wd=0


input_path=/media/rench/MyBook/HCP/HCP105_fiber_15_small


train_batch_size=8192
#train_batch_size=1024
val_batch_size=1024

model_name=Transformer
loss=nll_loss_raw
s1_path="./ModelWeights/"$loss'_'$model_name"_5_fold_weights_batch_1000_sample"

# # only eval on fold zero
# python train_s1.py --eval_fold_zero --input_path ${input_path} --epoch ${s1_epoch} --out_path_base ${s1_path} --opt ${s1_opt} --train_batch_size 1024 --val_batch_size 4096 --lr ${s1_lr} --weight_decay ${s1_wd} --scheduler wucd --T_0 10 --T_mult 2

# 5-fold cross validation
python train_s1.py --input_path ${input_path} --epoch ${s1_epoch} --out_path_base ${s1_path} --opt ${s1_opt} --train_batch_size ${train_batch_size} --val_batch_size ${val_batch_size} --lr ${s1_lr} --weight_decay ${s1_wd} --scheduler wucd --T_0 10 --T_mult 2 --model_name $model_name --loss $loss #-train_per_file
