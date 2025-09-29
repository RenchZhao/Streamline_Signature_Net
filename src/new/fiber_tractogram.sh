# requirements
# conda create -n pnlpipe3 python=3.6
# conda install -c conda-forge mrtrix3
conda activate pnlpipe3

5ttgen fsl T1w_acpc_dc_restore_brain.nii.gz 5TT.mif -premasked
dwi2response msmt_5tt Diffusion/data.nii.gz 5TT.mif Diffusion/RF_WM.txt Diffusion/RF_GM.txt Diffusion/RF_CSF.txt -voxels Diffusion/RF_voxels.mif -fslgrad Diffusion/bvecs Diffusion/bvals
dwi2fod msmt_csd Diffusion/data.nii.gz Diffusion/RF_WM.txt Diffusion/WM_FODs.mif Diffusion/RF_GM.txt Diffusion/GM.mif Diffusion/RF_CSF.txt Diffusion/CSF.mif -mask Diffusion/nodif_brain_mask.nii.gz -fslgrad Diffusion/bvecs Diffusion/bvals
tckgen -algorithm iFOD2 Diffusion/WM_FODs.mif output.tck -act 5TT.mif -backtrack -crop_at_gmwmi -seed_image Diffusion/nodif_brain_mask.nii.gz -maxlength 250 -minlength 40 -select 10M -cutoff 0.06
