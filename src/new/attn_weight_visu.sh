# TODO: change to your own directory
base_dir='/media/rench/MyBook/纤维分类-miccai2025/HCP的每个1000的5折/写上去的实验记录'
weight_dir='/HCP训练权重-第4折验证'
model_name='SigNet_tr_inception_with_mask'
dataset='HCP'
input_path=/media/rench/MyBook/HCP/HCP105_fiber_15_small
model_weight_path=$base_dir/'nll_loss_raw_SigNet_tr_inception_with_mask_5_fold_weights_1000_small'

python new/visu_feat.py --input_path ${input_path} --dataset ${dataset} --model_name ${model_name} --model_weight_path $model_weight_path
