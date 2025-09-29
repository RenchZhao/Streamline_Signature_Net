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

from sklearn.model_selection import StratifiedKFold
from sklearn.utils import shuffle


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
    inputVtkDataset = '/home/rench/code/WMA-related/ORG_data/vtp'
    outputDir = '/home/rench/code/WMA-related/ORG_data/custom/sig_3579_mean'
    fold_num=5

    #numPoints=15
    numPoints=15
    feature_type='RAS_sig'


    if not os.path.exists(outputDir):
        os.makedirs(outputDir)

    # gen feat and label
    feat_ls = None
    label_ls = []
    for vtk_fname in os.listdir(inputVtkDataset):
        pd_tract = wma.io.read_polydata(os.path.join(inputVtkDataset, vtk_fname))
        feat = gen_features(pd_tract, numPoints=numPoints,feature_type=feature_type)
        label_fname, _ = os.path.splitext(vtk_fname)
        label = int(label_fname[-3:])
        label_ls = label_ls + [label] * len(feat)#.shape[0]
        if feat_ls is None:
            feat_ls = feat
        else:
            feat_ls = np.concatenate((feat_ls,feat),axis=0)
    
    # k fold split and save
    label_ls = np.array(label_ls)
    cv = StratifiedKFold(n_splits=fold_num, shuffle=True, random_state=0) # shuffle in the same way
    i=0
    for _, val in cv.split(feat_ls, label_ls):
        # print(_.shape)
        # print(val.shape)
        i+=1
        val_feat_ls = feat_ls[val]
        val_label = label_ls[val]
        fold = i
        feat_h5_fname = 'sf_clusters_train_featMatrix_{}.h5'.format(fold)
        feat_h5_fname = os.path.join(outputDir, feat_h5_fname)
        label_h5_fname = 'sf_clusters_train_label_{}.h5'.format(fold)
        label_h5_fname = os.path.join(outputDir, label_h5_fname)
            
        # save_data_h5(val_feat_ls, feat_h5_fname,dataset_name='sc_feat')
        with h5py.File(feat_h5_fname, 'w') as f:
            f.create_dataset('sc_feat', data=val_feat_ls)
            f.create_dataset('other_feat', data=np.empty((0,*val_feat_ls.shape[1:])))

        #save_data_h5(label_ls, label_h5_fname, dataset_name='label')
        with h5py.File(label_h5_fname, 'w') as f:
            f.create_dataset('sc_label', data=val_label)
            f.create_dataset('label_names', data=['background']+['cluster_00{:0>3d}'.format(j) for j in range(1,801)])
            f.create_dataset('other_label', data=np.empty((0,*val_label.shape[1:])))

