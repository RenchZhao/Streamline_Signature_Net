from __future__ import print_function
import torch.utils.data as data
import torch
import numpy as np
import h5py
import sys
import os
sys.path.append('../')
import utils.tract_feat as tract_feat

import whitematteranalysis as wma
from new.gen_train_h5 import gen_features

def _feat_to_3D(feat):
    # 1 first; 2 last
    # 12 is the original point order
    # 21 is the fliped point order
    repeat_time=feat.shape[1]
    feat_12 = feat
    feat_21 = np.flip(feat_12, axis=1)

    # concatenate the different orders
    feat_1221 = np.concatenate((feat_12, feat_21), axis=1)
    feat_2112 = np.concatenate((feat_21, feat_12), axis=1)

    # reshape to a 4D array
    feat_shape = (feat_1221.shape[0], 1, feat_1221.shape[1], feat_1221.shape[2])

    feat_1221 = np.reshape(feat_1221, feat_shape)
    feat_2112 = np.reshape(feat_2112, feat_shape)

    # Now the dimension is (# of fibers, 2, # of points, 3)
    # the second D is [1221; 2112]; the fourth D is RAS
    feat_1221_2112 = np.concatenate((feat_1221, feat_2112), axis=1)

    # Repeat the send D;
    # In the tmp variable, it is [1221; 1221; ...; 2112; 2112; ....],
    # but we want [1221; 2112; 1221; 2112; ....]
    feat_1221_2112_repeat_tmp = np.repeat(feat_1221_2112, repeat_time, axis=1)

    feat_1221_2112_repeat = np.zeros(feat_1221_2112_repeat_tmp.shape)

    feat_1221_2112_repeat[:, 0::2, :, :] = feat_1221_2112_repeat_tmp[:, 0:repeat_time, :, :]
    feat_1221_2112_repeat[:, 1::2, :, :] = feat_1221_2112_repeat_tmp[:, repeat_time:, :, :]

    feat_1221_2112_repeat=np.transpose(feat_1221_2112_repeat, (0,3,1,2)) # newly added [B, C, 2N, 2N]
    return feat_1221_2112_repeat


class h5Dataset(data.Dataset):
    def __init__(self, feat_h5, label_h5, logger, transform=None, dataset_name='HCP'):
        # load feature data
        self.logger = logger
        if dataset_name=='HCP':
            feat_h5 = h5py.File(feat_h5, 'r')
            self.features = feat_h5['feat'][:]#.squeeze(-1)
            feat_h5.close()
            # load label data
            label_h5 = h5py.File(label_h5, 'r')
            self.labels = label_h5['label'][:]

            # label names list
            self.label_names = [*label_h5['label_names'][:]]
            self.label_names = [b'background'] + self.label_names
            label_h5.close()

        else:
            # ORG
            # load feature data
            feat_h5 = h5py.File(feat_h5, 'r')
            self.features = np.concatenate((feat_h5['sc_feat'], feat_h5['other_feat']), axis=0)
            # load label data
            label_h5 = h5py.File(label_h5, 'r')
            self.labels = np.concatenate((label_h5['sc_label'], label_h5['other_label']), axis=0)
            # label names list
            self.label_names = [*label_h5['label_names']]
            label_h5.close()
        
        if transform:
            self.features = transform(self.features)
        # self.logger.info('The size of feature is {}'.format(self.features.shape))

    def __getitem__(self, index):
        point_set = self.features[index]
        label = self.labels[index]
        if point_set.dtype == 'float32':
            point_set = torch.from_numpy(point_set)
        else:
            point_set = torch.from_numpy(point_set.astype(np.float32))
            #print('Feature is not in float32 format')

        if label.dtype == 'int64':
            label = torch.from_numpy(np.array([label]))
        else:
            label = torch.from_numpy(np.array([label]).astype(np.int64))
            #print('Label is not in int64 format')

        return point_set, label

    def __len__(self):
        return len(self.labels)
    
    def obtain_label_names(self):
        return self.label_names
    

class HCPDataset(data.Dataset):
    def __init__(self, root, logger, num_fold=1, k=5, split='train', transform=None):
        self.root = root
        self.split = split
        self.num_fold = num_fold
        self.k = k
        self.logger = logger
        features_combine = None
        labels_combine = None
        if self.split == 'train':
            train_fold = 0
            train_fold_lst = []
            for i in range(self.k):
                if i+1 != self.num_fold:
                    feat_h5 = h5py.File(os.path.join(root, 'HCP_featMatrix_fold_{}.h5'.format(str(i+1))), 'r')
                    features = feat_h5['feat'][:]#.squeeze(-1)
                    feat_h5.close()
                    # load label data
                    label_h5 = h5py.File(os.path.join(root, 'HCP_labelMatrix_fold_{}.h5'.format(str(i+1))), 'r')
                    labels = label_h5['label'][:]
                    if transform:
                        features = transform(features)
                    if train_fold == 0:
                        features_combine = features
                        labels_combine = labels
                    else:
                        features_combine = np.concatenate((features_combine, features), axis=0)
                        labels_combine = np.concatenate((labels_combine, labels), axis=0)
                    train_fold_lst.append(i+1)
                    train_fold += 1
            self.features = features_combine
            self.labels = labels_combine
            logger.info('use {} fold as train data'.format(train_fold_lst))
        else:
            # load feature data
            # TODO: Feel free to change the path
            feat_h5 = h5py.File(os.path.join(root, 'HCP_featMatrix_fold_{}.h5'.format(self.num_fold)), 'r')
            self.features = feat_h5['feat'][:]#.squeeze(-1)
            feat_h5.close()
            if transform:
                self.features = transform(self.features)
            # load label data
            label_h5 = h5py.File(os.path.join(root, 'HCP_labelMatrix_fold_{}.h5'.format(self.num_fold)), 'r')
            self.labels = label_h5['label'][:]
            logger.info('use {} fold as validation data'.format(self.num_fold))

        # label names list
        self.label_names = [*label_h5['label_names'][:]]
        self.label_names = [b'background'] + self.label_names # modified
        label_h5.close()
        self.logger.info('The size of feature for {} is {}'.format(self.split, self.features.shape))
        # if split == 'val':
        #     print('The label names are: {}'.format(self.label_names))

    def __getitem__(self, index):
        point_set = self.features[index]
        label = self.labels[index]
        if point_set.dtype == 'float32':
            point_set = torch.from_numpy(point_set)
        else:
            point_set = torch.from_numpy(point_set.astype(np.float32))
            #print('Feature is not in float32 format')

        if label.dtype == 'int64':
            label = torch.from_numpy(np.array([label]))
        else:
            label = torch.from_numpy(np.array([label]).astype(np.int64))
            #print('Label is not in int64 format')

        return point_set, label

    def __len__(self):
        return len(self.labels)
    
    def obtain_label_names(self):
        return self.label_names


# class SupConDataset(data.Dataset):
#     """Obtain data from ORG dataset and then generate bilateral pair for each fiber"""
#     # TODO: Feel free to change the data loading module to fit your data.
#     def __init__(self, root, logger, num_fold=1, k=5, split='train'):
#         self.root = root
#         self.split = split
#         self.num_fold = num_fold
#         self.k = k
#         self.logger = logger
#         features_combine = None
#         labels_combine = None
#         if self.split == 'train':
#             train_fold = 0
#             train_fold_lst = []
#             for i in range(self.k):
#                 if i+1 != self.num_fold:
#                     # load feature data
#                     feat_h5 = h5py.File(os.path.join(root, 'sf_clusters_train_featMatrix_{}.h5'.format(str(i+1))), 'r')
#                     features = np.concatenate((feat_h5['sc_feat'], feat_h5['other_feat']), axis=0)
#                     # load label data
#                     label_h5 = h5py.File(os.path.join(root, 'sf_clusters_train_label_{}.h5'.format(str(i+1))), 'r')
#                     labels = np.concatenate((label_h5['sc_label'], label_h5['other_label']), axis=0)
#                     if train_fold == 0:
#                         features_combine = features
#                         labels_combine = labels
#                     else:
#                         features_combine = np.concatenate((features_combine, features), axis=0)
#                         labels_combine = np.concatenate((labels_combine, labels), axis=0)
#                     train_fold_lst.append(i+1)
#                     train_fold += 1
#             self.features = features_combine
#             self.labels = labels_combine
#             logger.info('use {} fold as train data'.format(train_fold_lst))
#         else:
#             # load feature data
#             feat_h5 = h5py.File(os.path.join(root, 'sf_clusters_train_featMatrix_{}.h5'.format(self.num_fold)), 'r')
#             self.features = np.concatenate((feat_h5['sc_feat'], feat_h5['other_feat']), axis=0)
#             # load label data
#             label_h5 = h5py.File(os.path.join(root, 'sf_clusters_train_label_{}.h5'.format(self.num_fold)), 'r')
#             self.labels = np.concatenate((label_h5['sc_label'], label_h5['other_label']), axis=0)
#             logger.info('use {} fold as validation data'.format(self.num_fold))

#         # label names list
#         self.label_names = [*label_h5['label_names']]
#         self.logger.info('The size of feature for {} is {}'.format(self.split, self.features.shape))

#     def __getitem__(self, index):
#         point_set = self.features[index]
#         label = self.labels[index]
#         if point_set.dtype == 'float32':
#             point_set = torch.from_numpy(point_set)
#         else:
#             point_set = torch.from_numpy(point_set.astype(np.float32))
#             print('Feature is not in float32 format')

#         if label.dtype == 'int64':
#             label = torch.from_numpy(np.array([label]))
#         else:
#             label = torch.from_numpy(np.array([label]).astype(np.int64))
#             print('Label is not in int64 format')

        # bilateral pair for pointset
class SupConDataset(data.Dataset):
    def __init__(self, root, logger, num_fold=1, k=5, split='train', transform=None, dataset_name='HCP'):
        self.root = root
        self.split = split
        self.num_fold = num_fold
        self.k = k
        self.logger = logger
        features_combine = None
        labels_combine = None
        if self.split == 'train':
            train_fold = 0
            train_fold_lst = []
            for i in range(self.k):
                if i+1 != self.num_fold:
                    if dataset_name=='HCP':
                        feat_h5 = h5py.File(os.path.join(root, 'HCP_featMatrix_fold_{}.h5'.format(str(i+1))), 'r')
                        label_h5 = h5py.File(os.path.join(root, 'HCP_labelMatrix_fold_{}.h5'.format(str(i+1))), 'r')
                        features = feat_h5['feat'][:]
                        feat_h5.close()
                        # load label data
                        labels = label_h5['label'][:]
                    else: #if dataset_name=='ORG':
                        feat_h5 = h5py.File(os.path.join(root, 'sf_clusters_train_featMatrix_{}.h5'.format(str(i+1))), 'r')
                        label_h5 = h5py.File(os.path.join(root, 'sf_clusters_train_label_{}.h5'.format(str(i+1))), 'r')
                        features = np.concatenate((feat_h5['sc_feat'], feat_h5['other_feat']), axis=0)
                        feat_h5.close()
                        # load label data
                        labels = np.concatenate((label_h5['sc_label'], label_h5['other_label']), axis=0)
                    if transform:
                        features = transform(features)
                    if train_fold == 0:
                        features_combine = features
                        labels_combine = labels
                    else:
                        features_combine = np.concatenate((features_combine, features), axis=0)
                        labels_combine = np.concatenate((labels_combine, labels), axis=0)
                    train_fold_lst.append(i+1)
                    train_fold += 1
            self.features = features_combine
            self.labels = labels_combine
            logger.info('use {} fold as train data'.format(train_fold_lst))
        else:
            # load feature data
            # TODO: Feel free to change the path
            if dataset_name=='HCP':
                feat_h5 = h5py.File(os.path.join(root, 'HCP_featMatrix_fold_{}.h5'.format(self.num_fold)), 'r')
                label_h5 = h5py.File(os.path.join(root, 'HCP_labelMatrix_fold_{}.h5'.format(self.num_fold)), 'r')
            else: #if dataset_name=='ORG':
                feat_h5 = h5py.File(os.path.join(root, 'sf_clusters_train_featMatrix_{}.h5'.format(self.num_fold)), 'r')
                label_h5 = h5py.File(os.path.join(root, 'sf_clusters_train_label_{}.h5'.format(self.num_fold)), 'r')
            self.features = feat_h5['feat'][:]
            feat_h5.close()
            # load label data
            self.labels = label_h5['label'][:]
            logger.info('use {} fold as validation data'.format(self.num_fold))

        # label names list
        self.label_names = [*label_h5['label_names'][:]]
        self.label_names = [b'background'] + self.label_names # modified
        label_h5.close()
        self.logger.info('The size of feature for {} is {}'.format(self.split, self.features.shape))
        # if split == 'val':
        #     print('The label names are: {}'.format(self.label_names))

    def __getitem__(self, index):
        point_set = self.features[index]
        label = self.labels[index]
        if point_set.dtype == 'float32':
            point_set = torch.from_numpy(point_set)
        else:
            point_set = torch.from_numpy(point_set.astype(np.float32))
            #print('Feature is not in float32 format')

        if label.dtype == 'int64':
            label = torch.from_numpy(np.array([label]))
        else:
            label = torch.from_numpy(np.array([label]).astype(np.int64))
            #print('Label is not in int64 format')
        point_set_bilateral = point_set.detach().clone()
        point_set_bilateral[:, 0] = -point_set_bilateral[:, 0]
        new_point_set = [point_set, point_set_bilateral]

        return new_point_set, label

    def __len__(self):
        return len(self.labels)

    def obtain_label_names(self):
        return self.label_names


class ORGDataset(data.Dataset):
    def __init__(self, root, logger, num_fold=1, k=5, split='train', transform=None):
        # TODO: Feel free to change the data loading module to fit your data.
        # TODO: I saved my data into .h5 file, the size of "features" is [num_samples, num_points, 3], and the size of "labels" is [num_samples, ]
        self.root = root
        self.split = split
        self.num_fold = num_fold
        self.k = k
        self.logger = logger
        features_combine = None
        labels_combine = None
        if self.split == 'train':
            train_fold = 0
            train_fold_lst = []
            for i in range(self.k):
                if i+1 != self.num_fold:
                    # load feature data
                    # TODO: Feel free to change the path
                    feat_h5 = h5py.File(os.path.join(root, 'sf_clusters_train_featMatrix_{}.h5'.format(str(i+1))), 'r')
                    features = np.concatenate((feat_h5['sc_feat'], feat_h5['other_feat']), axis=0)
                    if transform:
                        features = transform(features)
                    # load label data
                    label_h5 = h5py.File(os.path.join(root, 'sf_clusters_train_label_{}.h5'.format(str(i+1))), 'r')
                    labels = np.concatenate((label_h5['sc_label'], label_h5['other_label']), axis=0)
                    if train_fold == 0:
                        features_combine = features
                        labels_combine = labels
                    else:
                        features_combine = np.concatenate((features_combine, features), axis=0)
                        labels_combine = np.concatenate((labels_combine, labels), axis=0)
                    train_fold_lst.append(i+1)
                    train_fold += 1
            self.features = features_combine
            self.labels = labels_combine
            logger.info('use {} fold as train data'.format(train_fold_lst))
        else:
            # load feature data
            # TODO: Feel free to change the path
            feat_h5 = h5py.File(os.path.join(root, 'sf_clusters_train_featMatrix_{}.h5'.format(self.num_fold)), 'r')
            self.features = np.concatenate((feat_h5['sc_feat'], feat_h5['other_feat']), axis=0)
            if transform:
                self.features = transform(self.features)
            # load label data
            label_h5 = h5py.File(os.path.join(root, 'sf_clusters_train_label_{}.h5'.format(self.num_fold)), 'r')
            self.labels = np.concatenate((label_h5['sc_label'], label_h5['other_label']), axis=0)
            logger.info('use {} fold as validation data'.format(self.num_fold))

        # label names list
        self.label_names = [*label_h5['label_names']]
        self.logger.info('The size of feature for {} is {}'.format(self.split, self.features.shape))
        # if split == 'val':
        #     print('The label names are: {}'.format(self.label_names))

    def __getitem__(self, index):
        point_set = self.features[index]
        label = self.labels[index]
        if point_set.dtype == 'float32':
            point_set = torch.from_numpy(point_set)
        else:
            point_set = torch.from_numpy(point_set.astype(np.float32))
            #print('Feature is not in float32 format')

        if label.dtype == 'int64':
            label = torch.from_numpy(np.array([label]))
        else:
            label = torch.from_numpy(np.array([label]).astype(np.int64))
            #print('Label is not in int64 format')

        return point_set, label

    def __len__(self):
        return len(self.labels)

    def obtain_label_names(self):
        return self.label_names


class TestDataset(data.Dataset):
    def __init__(self, test_features_path, test_label_path,
                 label_names_path, script_name, normalization=False,
                 data_augmentation=False, transform=None):
        self.feat_p = test_features_path
        self.label_p = test_label_path
        self.label_names_p = label_names_path
        self.script_name = script_name
        self.data_augmentation = data_augmentation
        self.normalization = normalization
        # load label names
        print(self.script_name, 'Load clusters names along with the model.')
        labels_names_in_model_h5 = h5py.File(self.label_names_p, 'r')
        self.label_names_in_model = labels_names_in_model_h5['y_names']
        self.label_names_in_model = [name.decode('UTF-8') for name in self.label_names_in_model]
        # load feature data
        print(self.script_name, 'Load input feature.')
        feat_h5 = h5py.File(self.feat_p, 'r')
        # self.features = np.asarray(feat_h5['feat']).squeeze()
        self.features = np.asarray(feat_h5['feat']).squeeze(-1)
        if transform:
            self.features = transform(self.features)
        # load label data
        if self.label_p is not None:
            print(self.script_name, 'Load input label.')
            label_h5 = h5py.File(self.label_p, 'r')
            label_test_gt = label_h5['label_array'].value.astype(int)
            label_names = label_h5['label_names'].value
            # Generate final ground truth label
            print(self.script_name, 'Generate FINAL ground truth label for evaluation.')
            self.label_test_final = tract_feat.update_y_test_based_on_model_y_names(label_test_gt, label_names,
                                                                                    self.label_names_in_model)
        else:
            self.label_test_final = None
        print('The size of feature for is {}'.format(self.features.shape))

    def __getitem__(self, index):
        point_set = self.features[index]
        if self.label_p is not None:
            label = self.label_test_final[index]
        else:
            label = None
        if self.normalization:
            point_set = point_set - np.expand_dims(np.mean(point_set, axis=0), 0)  # center
            dist = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)), 0)
            point_set = point_set / dist  # scale

        if self.data_augmentation:
            theta = np.random.uniform(0, np.pi * 2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            point_set[:, [0, 2]] = point_set[:, [0, 2]].dot(rotation_matrix)  # random rotation
            point_set += np.random.normal(0, 0.02, size=point_set.shape)  # random jitter

        if point_set.dtype == 'float32':
            point_set = torch.from_numpy(point_set)
        else:
            point_set = torch.from_numpy(point_set.astype(np.float32))
            # print('Feature is not in float32 format')

        if label is None:
            # None can not be passed
            label = torch.from_numpy(np.asarray([1]))
        elif label.dtype == 'int64':
            label = torch.from_numpy(np.array([label]))
        else:
            label = torch.from_numpy(np.array([label]).astype(np.int64))
            # print('Label is not in int64 format')

        return point_set, label

    def __len__(self):
        return self.features.shape[0]

class InferVtkDataset(data.Dataset):
    def __init__(self, vtk_dataset, script_name='<inference>', normalization=False,
                 data_augmentation=False, transform=None):
        pd_tract = wma.io.read_polydata(vtk_dataset)
        self.features = np.asarray(gen_features(pd_tract))
        if transform:
            self.features = transform(self.features)
        self.script_name = script_name
        self.data_augmentation = data_augmentation
        self.normalization = normalization
        
        print('The size of feature for is {}'.format(self.features.shape))

    def __getitem__(self, index):
        point_set = self.features[index]
        
        if self.normalization:
            point_set = point_set - np.expand_dims(np.mean(point_set, axis=0), 0)  # center
            dist = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)), 0)
            point_set = point_set / dist  # scale

        if self.data_augmentation:
            theta = np.random.uniform(0, np.pi * 2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            point_set[:, [0, 2]] = point_set[:, [0, 2]].dot(rotation_matrix)  # random rotation
            point_set += np.random.normal(0, 0.02, size=point_set.shape)  # random jitter

        if point_set.dtype == 'float32':
            point_set = torch.from_numpy(point_set)
        else:
            point_set = torch.from_numpy(point_set.astype(np.float32))
            # print('Feature is not in float32 format')


        return point_set

    def __len__(self):
        return self.features.shape[0]
