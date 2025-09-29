import sys
sys.path.append('../')

import ast

import utils.tract_feat as tract_feat

import whitematteranalysis as wma
import os
import h5py
import argparse
import numpy as np

from functools import partial

label_di = {
    'AF_left':1,       #(Arcuate fascicle)
    'AF_right':2,
    'ATR_left':3,      #(Anterior Thalamic Radiation)
    'ATR_right':4,
    'CA':5,            #(Commissure Anterior)import utils.tract_feat as tract_feat
    'CC_1':6,          #(Rostrum)
    'CC_2':7,          #(Genu)
    'CC_3':8,          #(Rostral body (Premotor))
    'CC_4':9,          #(Anterior midbody (Primary Motor))
    'CC_5':10,         #(Posterior midbody (Primary Somatosensory))
    'CC_6':11,         #(Isthmus)
    'CC_7':12,         #(Splenium)
    'CG_left':13,      #(Cingulum left)
    'CG_right':14, 
    'CST_left':15,     #(Corticospinal tract
    'CST_right':16,
    'MLF_left':17,     #(Middle longitudinal fascicle)
    'MLF_right':18,
    'FPT_left':19,     #(Fronto-pontine tract)
    'FPT_right':20,
    'FX_left':21,      #(Fornix)
    'FX_right':22,
    'ICP_left':23,     #(Inferior cerebellar peduncle)
    'ICP_right':24,
    'IFO_left':25,     #(Inferior occipito-frontal fascicle) 
    'IFO_right':26,
    'ILF_left':27,     #(Inferior longitudinal fascicle) 
    'ILF_right':28,
    'MCP':29,          #(Middle cerebellar peduncle)
    'OR_left':30,      #(Optic radiation) 
    'OR_right':31,
    'POPT_left':32,    #(Parieto‐occipital pontine)
    'POPT_right':33,
    'SCP_left':34,     #(Superior cerebellar peduncle)
    'SCP_right':35,
    'SLF_I_left':36,   #(Superior longitudinal fascicle I)
    'SLF_I_right':37,
    'SLF_II_left':38,  #(Superior longitudinal fascicle II)
    'SLF_II_right':39,
    'SLF_III_left':40, #(Superior longitudinal fascicle III)
    'SLF_III_right':41,
    'STR_left':42,     #(Superior Thalamic Radiation)
    'STR_right':43,
    'UF_left':44,      #(Uncinate fascicle) 
    'UF_right':45,
    'CC':46,           #(Corpus Callosum - all)
    'T_PREF_left':47,  #(Thalamo-prefrontal)
    'T_PREF_right':48,
    'T_PREM_left':49,  #(Thalamo-premotor)
    'T_PREM_right':50,
    'T_PREC_left':51,  #(Thalamo-precentral)
    'T_PREC_right':52,
    'T_POSTC_left':53, #(Thalamo-postcentral)
    'T_POSTC_right':54,
    'T_PAR_left':55,   #(Thalamo-parietal)
    'T_PAR_right':56,
    'T_OCC_left':57,   #(Thalamo-occipital)
    'T_OCC_right':58,
    'ST_FO_left':59,   #(Striato-fronto-orbital)
    'ST_FO_right':60,
    'ST_PREF_left':61, #(Striato-prefrontal)
    'ST_PREF_right':62,
    'ST_PREM_left':63, #(Striato-premotor)
    'ST_PREM_right':64,
    'ST_PREC_left':65, #(Striato-precentral)
    'ST_PREC_right':66,
    'ST_POSTC_left':67,#(Striato-postcentral)
    'ST_POSTC_right':68,
    'ST_PAR_left':69,  #(Striato-parietal)
    'ST_PAR_right':70,
    'ST_OCC_left':71,  #(Striato-occipital)
    'ST_OCC_right':72,

}

def gen_features(pd_tract, feature_type='RAS', numPoints=15, numRepeats=15, script_name='<>', normalization=False, data_augmentation=False):
    feat =  gen_tract_features_core(pd_tract, feature_type=feature_type, numPoints=numPoints, numRepeats=numRepeats, script_name=script_name)
    if normalization:
        ### will introduce class bias
        # feat = feat - np.expand_dims(np.mean(feat, axis=0), 0)  # center
        # dist = np.max(np.sqrt(np.sum(feat ** 2, axis=1)), 0)
        # feat = feat / dist  # scale

        ### normalize points axis per streamline to [-1,1]
        # feat = feat - np.expand_dims(np.mean(feat, axis=1), 1)  # center
        # feat = feat / np.expand_dims(np.max(abs(feat), axis=1), 1)

        ### normalize all points axis per streamline to [-1,1]
        shape = feat.shape
        feat = feat.reshape((shape[0], -1))
        feat = feat - np.expand_dims(np.mean(feat, axis=1), 1)  # center
        feat = feat / np.expand_dims(np.max(abs(feat), axis=1), 1)
        feat = feat.reshape(shape)
        # print(feat.shape)
        # print(-1<=feat.all()<=1)

        ### normalize all points axis from a global parameter
        mean_val = 0
        dev_val = 0.8
        feat = (feat - mean_val)/dev_val

        # print(feat.shape)
        # print(-1<=feat.all()<=1)


    if data_augmentation:
        theta = np.random.uniform(0, np.pi * 2)
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        feat[:, [0, 2]] = feat[:, [0, 2]].dot(rotation_matrix)  # random rotation
        feat += np.random.normal(0, 0.02, size=feat.shape)  # random jitter

    return feat#.transpose(1,2) # [B, N, 3] to [B, 3, N]

def gen_tract_features_core(pd_tract, feature_type='RAS', numPoints=15, numRepeats=15, script_name='<>'):
    print(script_name, 'Computing feauture:', feature_type)
    if feature_type == 'RAS':
        resample_by_curvatures = partial(tract_feat.resample_trajectory, M=numPoints)
        feat = tract_feat.convert_from_polydata_all(pd_tract, data_transform=resample_by_curvatures)

    elif feature_type == 'RAS_sig':
        feat = tract_feat.convert_from_polydata_all(pd_tract, data_transform=tract_feat.multi_window_signature_feat_avg)

    elif feature_type == 'RAS_raw':
        feat_RAS = tract_feat.feat_RAS(pd_tract, number_of_points=numPoints)
        feat = feat_RAS

        # Reshape from 3D (num of fibers, num of points, num of features) to 4D (num of fibers, num of points, num of features, 1)
        # The 4D array considers the input has only one channel (depth = 1)
        # feat_shape = np.append(feat_RAS.shape, 1)
        # feat = np.reshape(feat_RAS, feat_shape)

    elif feature_type == 'RAS-3D':

        feat_RAS_3D = tract_feat.feat_RAS_3D(pd_tract, number_of_points=numPoints, repeat_time=numRepeats)

        feat = feat_RAS_3D

    elif feature_type == 'RASCurvTors':

        feat_curv_tors = tract_feat.feat_RAS_curv_tors(pd_tract, number_of_points=numPoints)
        
        feat = feat_curv_tors

        # feat_shape = np.append(feat_curv_tors.shape, 1)
        # feat = np.reshape(feat_curv_tors, feat_shape)

    elif feature_type == 'CurvTors':

        feat_curv_tors = tract_feat.feat_curv_tors(pd_tract, number_of_points=numPoints)

        feat = feat_curv_tors

        # feat_shape = np.append(feat_curv_tors.shape, 1)
        # feat = np.reshape(feat_curv_tors, feat_shape)

    else:
        raise ValueError('Please enter valid feature names.')

    # print(script_name, 'Feature matrix shape:', feat.shape)

    return feat

def save_data_h5(datalist, h5_fname, dataset_name='feat'):
    with h5py.File(h5_fname, 'w') as f:
        f.create_dataset(dataset_name, data=datalist)

if __name__ == "__main__":
    inputVtkDataset = '/media/rench/MyBook/HCP/HCP105_Zenodo_NewVtkFormat'
    outputDir = '/path/to/your/h5'
    cross_val_txt = '/media/rench/MyBook/HCP/5_fold.txt'
    #numPoints=15
    numPoints=30
    feature_type='RAS_raw'
    less_data_per_class = 1000 # as "Tract Dictionary Learning for Fast and Robust Recognition of Fiber Bundles" do


    if not os.path.exists(outputDir):
        os.makedirs(outputDir)

    fold_ls = []
    file = open(cross_val_txt, 'r')
    for line in file.readlines():
        data = line.split(' = ')[-1]
        fold_ls.append(ast.literal_eval(data)) #fold_1 ... fold_5
        # print(len(ast.literal_eval(data)))
        # print(len(fold_ls))
    file.close()

    for i in range(len(fold_ls)):
        fold = i+1
        feat_h5_fname = 'HCP_featMatrix_fold_{}.h5'.format(fold)
        feat_h5_fname = os.path.join(outputDir, feat_h5_fname)
        label_h5_fname = 'HCP_labelMatrix_fold_{}.h5'.format(fold)
        label_h5_fname = os.path.join(outputDir, label_h5_fname)
        if not os.path.exists(label_h5_fname) or not os.path.exists(feat_h5_fname):
            label_ls = []
            feat_ls = None
            for j in range(len(fold_ls[i])):
                subj_dir = os.path.join(inputVtkDataset, fold_ls[i][j])
                for vtk_fname in os.listdir(subj_dir):
                    pd_tract = wma.io.read_polydata(os.path.join(subj_dir, vtk_fname))
                    feat = gen_features(pd_tract, numPoints=numPoints,feature_type=feature_type)
                    if less_data_per_class is not None:
                        if len(feat) > less_data_per_class:
                            feat = feat[:less_data_per_class]
                    label_fname, _ = os.path.splitext(vtk_fname)
                    label = label_di[label_fname]
                    label_ls = label_ls + [label] * len(feat)#.shape[0]
                    if feat_ls is None:
                        feat_ls = feat
                    else:
                        feat_ls = np.concatenate((feat_ls,feat),axis=0)
            save_data_h5(feat_ls, feat_h5_fname)
            #save_data_h5(label_ls, label_h5_fname, dataset_name='label')
            with h5py.File(label_h5_fname, 'w') as f:
                f.create_dataset('label', data=label_ls)
                f.create_dataset('label_names', data=list(label_di.keys()))

