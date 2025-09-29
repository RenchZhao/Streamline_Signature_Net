
base_dir='/media/rench/MyBook/纤维分类-miccai2025/测试'
weight_dir='/HCP训练权重-第4折验证'

inputDir=$base_dir'/HCP_tractograms'
refDir='/media/rench/MyBook/HCP/HCP_many_img' #HCP
ref_rel_path='T1w/T1w_acpc_dc_restore_brain.nii.gz' #HCP
resultTckDir=$base_dir'/HCP_tck_infer' #HCP
resultImgDir=$base_dir'/HCP_tract_img' #HCP


metric='acc'
reco_thresh=0.9

model_name='PointNetCls'
model_weight_path=$base_dir$weight_dir/$model_name'_best_'$metric'_model.pth'

python infer_dataset_main.py -reco_thresh $reco_thresh -inputDir $inputDir -resultTckDir $resultTckDir/$model_name'_thr_'$reco_thresh -resultImgDir $resultImgDir/$model_name'_thr_'$reco_thresh -refDir $refDir -ref_rel_path $ref_rel_path -model_name $model_name -model_weight_path $model_weight_path --MRtrix3_tck

model_name='DeepWMA'
model_weight_path=$base_dir$weight_dir/$model_name'_best_'$metric'_model.pth'


python infer_dataset_main.py -reco_thresh $reco_thresh -inputDir $inputDir -resultTckDir $resultTckDir/$model_name'_thr_'$reco_thresh -resultImgDir $resultImgDir/$model_name'_thr_'$reco_thresh -refDir $refDir -ref_rel_path $ref_rel_path -model_name $model_name -model_weight_path $model_weight_path --MRtrix3_tck  

model_name='SigNet_tr_inception_with_mask'
model_weight_path=$base_dir$weight_dir/$model_name'_best_'$metric'_model.pth'


python infer_dataset_main.py -reco_thresh $reco_thresh -inputDir $inputDir -resultTckDir $resultTckDir/$model_name'_thr_'$reco_thresh -resultImgDir $resultImgDir/$model_name'_thr_'$reco_thresh -refDir $refDir -ref_rel_path $ref_rel_path -model_name $model_name -model_weight_path $model_weight_path --MRtrix3_tck --rm_tmp
# test
#python visu_tracts.py -tckDir /media/rench/MyBook/HCP/test/HCP_data_test/tck_infer/PointNetCls -outputDir /media/rench/MyBook/HCP/test/HCP_data_test/infer_tract_img/PointNetCls -refImgDir /media/rench/MyBook/HCP/HCP_many_img -ref_rel_path T1w/T1w_acpc_dc_restore_brain.nii.gz

