Code is still under managing. Accepted in [CDMRI 2025] "Streamline Signature Net(SSN): Efficient White Matter Pathway Recognition for Bundles Parcellation Using Path Signature", Computational Diffusion MRI

Uploaded all files.
But comments and instructions are still under developing.

## License  
This repository contains code under two licenses:  
- File in `src/new/lars.py` is licensed under Apache-2.0, see LICENSE.Apache-2.0  
- Files in `src/` except `src/new/lars.py` are licensed under BSD-3-Clause, see LICENSE.BSD-3-Clause 

## run
conda create --name SSN python=3.12
conda activate SSN

pip install git+https://github.com/SlicerDMRI/whitematteranalysis.git

pip install torch torchvision torchaudio
pip install argparse signatory h5py matplotlib scikit-learn==1.5.2
conda install -c conda-forge libstdcxx-ng=13

git clone https://github.com/RenchZhao/Streamline_Signature_Net.git
cd Streamline_Signature_Net

## if need to prepare datasets
conda create -n pnlpipe3 python=3.6
conda install -c mrtrix3 mrtrix3


## possible bugs

ImportError: /home/user/anaconda3/envs/SSN/bin/../lib/libstdc++.so.6: version `GLIBCXX_3.4.32' not found (required by /home/user/anaconda3/envs/SSN/lib/python3.12/site-packages/signatory/_impl.cpython-312-x86_64-linux-gnu.so)
run:
conda install -c conda-forge libstdcxx-ng=13




"cross_val_txt" variable in new/gen_train_h5.py
contains subjects of 5-fold cross validation of HCP dataset: https://zenodo.org/records/1285152
You can copy contents below without quotation mark to your own "5_fold.txt":
'''
fold1 = ['992774', '991267', '987983', '984472', '983773', '979984', '978578', '965771', '965367', '959574', '958976', '957974', '951457', '932554', '930449', '922854', '917255', '912447', '910241', '907656', '904044']
fold2 = ['901442', '901139', '901038', '899885', '898176', '896879', '896778', '894673', '889579', '887373', '877269', '877168', '872764', '872158', '871964', '871762', '865363', '861456', '859671', '857263', '856766']
fold3 = ['849971', '845458', '837964', '837560', '833249', '833148', '826454', '826353', '816653', '814649', '802844', '792766', '792564', '789373', '786569', '784565', '782561', '779370', '771354', '770352', '765056']
fold4 = ['761957', '759869', '756055', '753251', '751348', '749361', '748662', '748258', '742549', '734045', '732243', '729557', '729254', '715647', '715041', '709551', '705341', '704238', '702133', '695768', '690152']
fold5 = ['687163', '685058', '683256', '680957', '679568', '677968', '673455', '672756', '665254', '654754', '645551', '644044', '638049', '627549', '623844', '622236', '620434', '613538', '601127', '599671', '599469']
'''

new/get_label_dict.py is used to generate "label_di" variable in new/gen_train_h5.py