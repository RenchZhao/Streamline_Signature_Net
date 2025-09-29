#!/bin/bash

## Stage 1 training 
s1_epoch=80
s1_opt=Adam
#s1_opt='SGD'
s1_lr=1e-3
s1_wd=0


input_path=/home/rench/code/WMA-related/ORG_data/custom/eq_inteval_15


train_batch_size=1024
val_batch_size=4096
#val_batch_size=128

model_name=SigNet_tr_inception_with_mask
loss=nll_loss_raw
s1_path="./ModelWeights/"$loss'_'$model_name"_stage1_weights_ORG"

# # only eval on fold zero
# python train_s1.py --eval_fold_zero --input_path ${input_path} --epoch ${s1_epoch} --out_path_base ${s1_path} --opt ${s1_opt} --train_batch_size 1024 --val_batch_size 4096 --lr ${s1_lr} --weight_decay ${s1_wd} --scheduler wucd --T_0 10 --T_mult 2

# 5-fold cross validation
python train_s1.py --input_path ${input_path} --epoch ${s1_epoch} --out_path_base ${s1_path} --train_batch_size ${train_batch_size} --val_batch_size ${val_batch_size} --lr ${s1_lr} --weight_decay ${s1_wd} --scheduler wucd --T_0 10 --T_mult 2 --dataset ORG --model_name  $model_name --loss $loss --opt ${s1_opt}  #-train_per_file --stream 30 --stream=5 --in_dim=20
