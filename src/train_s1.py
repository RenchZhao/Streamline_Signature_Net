import argparse
import os
import pickle
import time
import h5py

import numpy as np

import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data

from utils.dataset import h5Dataset, HCPDataset, ORGDataset, _feat_to_3D
from utils.model import get_model, get_free_gpu, load_and_freeze_layers, load_model_weights
from utils.logger import create_logger
from utils.metrics_plots import classify_report, per_class_metric, process_curves, \
    calculate_prec_recall_f1, best_swap, save_best_weights, calculate_average_metric, gen_199_classify_report
from utils.funcs import unify_path, makepath, fix_seed
from utils.custom_loss import *

import torch.nn.functional as F

from new.lars import LARS

# GPU check
device = get_free_gpu() #torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_data(dataset='HCP', transform=None):
    """load train and validation data"""
    # load feature and label data
    if dataset=='ORG':
        train_dataset = ORGDataset(
            root=args.input_path,
            logger=logger,
            num_fold=num_fold,
            k=args.k_fold,
            transform=transform,
            split='train')
        val_dataset = ORGDataset(
            root=args.input_path,
            logger=logger,
            num_fold=num_fold,
            k=args.k_fold,
            transform=transform,
            split='val')
    else:
        train_dataset = HCPDataset(
            root=args.input_path,
            logger=logger,
            num_fold=num_fold,
            k=args.k_fold,
            transform=transform,
            split='train')

        val_dataset = HCPDataset(
            root=args.input_path,
            logger=logger,
            num_fold=num_fold,
            k=args.k_fold,
            transform=transform,
            split='val')

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.train_batch_size,
        shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.val_batch_size,
        shuffle=False)

    train_data_size = len(train_dataset)
    val_data_size = len(val_dataset)
    logger.info('The training data size is:{}'.format(train_data_size))
    logger.info('The validation data size is:{}'.format(val_data_size))
    num_classes = len(train_dataset.label_names)
    logger.info('The number of classes is:{}'.format(num_classes))

    # load label names
    train_label_names = train_dataset.obtain_label_names()
    val_label_names = val_dataset.obtain_label_names()
    assert train_label_names == val_label_names
    label_names = train_label_names
    label_names_h5 = h5py.File(os.path.join(args.out_path, 'label_names.h5'), 'w')
    print(list(label_names))
    label_names_h5['y_names'] = list(label_names)
    logger.info('The label names are: {}'.format(str(label_names)))

    return train_loader, val_loader, label_names, num_classes, train_data_size, val_data_size

def load_data_val(dataset='HCP', transform=None):
    """load validation data"""
    # load feature and label data
    if dataset=='HCP':
        feat_h5_prefix=os.path.join(args.input_path, 'HCP_featMatrix_fold_')
        label_h5_prefix=os.path.join(args.input_path, 'HCP_labelMatrix_fold_')
    else:
        # ORG
        feat_h5_prefix=os.path.join(args.input_path, 'sf_clusters_train_featMatrix_')
        label_h5_prefix=os.path.join(args.input_path, 'sf_clusters_train_label_')

    val_dataset = h5Dataset(feat_h5_prefix+str(num_fold)+'.h5', label_h5_prefix+str(num_fold)+'.h5', logger=logger, transform=transform, dataset_name=dataset)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.val_batch_size,
        shuffle=False)

    val_data_size = len(val_dataset)
    logger.info('The validation data size is:{}'.format(val_data_size))
    num_classes = len(val_dataset.label_names)
    logger.info('The number of classes is:{}'.format(num_classes))

    # load label names
    label_names = val_dataset.obtain_label_names()
    label_names_h5 = h5py.File(os.path.join(args.out_path, 'label_names.h5'), 'w')
    print(list(label_names))
    label_names_h5['y_names'] = list(label_names)
    logger.info('The label names are: {}'.format(str(label_names)))

    return  val_loader, label_names, num_classes, val_data_size

def train_val_net(net, running_transform=None):
    """train and validation of the network"""
    time_start = time.time()
    train_num_batch = train_data_size / args.train_batch_size
    val_num_batch = val_data_size / args.val_batch_size
    # save training and validating process data
    train_loss_lst, val_loss_lst, train_acc_lst, val_acc_lst, \
    train_precision_lst, val_precision_lst, train_recall_lst, val_recall_lst, \
    train_f1_lst, val_f1_lst = [], [], [], [], [], [], [], [], [], []
    # save weights with best metrics
    best_acc, best_f1_mac = 0, 0
    best_acc_epoch, best_f1_epoch = 1, 1
    best_acc_wts, best_f1_wts = None, None
    best_acc_val_labels_lst, best_f1_val_labels_lst = [], []
    best_acc_val_pred_lst, best_f1_val_pred_lst = [], []

    for epoch in range(args.epoch):
        train_start_time = time.time()
        epoch += 1
        total_train_loss, total_val_loss = 0, 0
        train_labels_lst, train_predicted_lst = [], []
        total_train_correct, total_val_correct = 0, 0
        val_labels_lst, val_predicted_lst = [], []

        # training
        for i, data in enumerate(train_loader, 0):
            points, label = data  # points [B, N, 3]
            label = label[:, 0]  # [B,1] rank2 to (, B) rank1
            if running_transform:
                points = torch.from_numpy(running_transform(points.numpy()).astype(np.float32))
            points, label = points.to(device), label.to(device)
            optimizer.zero_grad()
            net = net.train()
            pred = net(points)
            loss = criterion(pred, label)
            loss.backward()
            optimizer.step()
            if not isinstance(pred, torch.Tensor):
                pred = pred[0]
            if args.scheduler == 'wucd':
                scheduler.step(epoch-1 + i/train_num_batch)
            _, pred_idx = torch.max(pred, dim=1)
            correct = pred_idx.eq(label.data).cpu().sum()
            # for calculating training accuracy and loss
            total_train_correct += correct.item()
            total_train_loss += loss.item()
            # for calculating training weighted and macro metrics
            label = label.cpu().detach().numpy().tolist()
            train_labels_lst.extend(label)
            pred_idx = pred_idx.cpu().detach().numpy().tolist()
            train_predicted_lst.extend(pred_idx)

        # train accuracy loss
        avg_train_acc = total_train_correct / float(train_data_size)
        avg_train_loss = total_train_loss / float(train_num_batch)
        train_acc_lst.append(avg_train_acc)
        train_loss_lst.append(avg_train_loss)

        if args.scheduler == 'step':
            scheduler.step()
        elif args.scheduler == 'ReduceLROnPlateau':
            scheduler.step(avg_train_acc)


        # train macro p, r, f1
        mac_train_precision, mac_train_recall, mac_train_f1 = calculate_prec_recall_f1(train_labels_lst, train_predicted_lst)
        train_precision_lst.append(mac_train_precision)
        train_recall_lst.append(mac_train_recall)
        train_f1_lst.append(mac_train_f1)
        train_end_time = time.time()
        train_time = round(train_end_time-train_start_time, 2)
        logger.info('{} epoch [{}/{}] time: {}s train loss: {} accuracy: {} f1: {}'.format(
            script_name, epoch, args.epoch, train_time, round(avg_train_loss, 4), round(avg_train_acc, 4), round(mac_train_f1, 4)))

        # validation
        with torch.no_grad():
            val_start_time = time.time()
            for j, data in (enumerate(val_loader, 0)):
                points, label = data
                label = label[:, 0]
                if running_transform:
                    points = torch.from_numpy(running_transform(points.numpy()).astype(np.float32))
                points, label = points.to(device), label.to(device)
                net = net.eval()
                try:
                    pred = net(points)
                except RuntimeError as exception:
                    if "out of memory" in str(exception):
                        logger.info("WARNING: out of memory")
                        if hasattr(torch.cuda, 'empty_cache'):
                            torch.cuda.empty_cache()
                    else:
                        raise exception
                    
                loss = criterion(pred, label)
                if not isinstance(pred, torch.Tensor):
                    pred = pred[0]
                _, pred_idx = torch.max(pred, dim=1)
                correct = pred_idx.eq(label.data).cpu().sum()
                # for calculating validation accuracy and loss
                total_val_correct += correct.item()
                total_val_loss += loss.item()
                # for calculating validation weighted and macro metrics
                label = label.cpu().detach().numpy().tolist()
                val_labels_lst.extend(label)
                pred_idx = pred_idx.cpu().detach().numpy().tolist()
                val_predicted_lst.extend(pred_idx)
        # calculate the validation accuracy and loss for the epoch
        avg_val_acc = total_val_correct / float(val_data_size)
        avg_val_loss = total_val_loss / float(val_num_batch)
        val_acc_lst.append(avg_val_acc)
        val_loss_lst.append(avg_val_loss)
        # calculate the validation macro metrics
        mac_val_precision, mac_val_recall, mac_val_f1 = calculate_prec_recall_f1(val_labels_lst, val_predicted_lst)
        val_precision_lst.append(mac_val_precision)
        val_recall_lst.append(mac_val_recall)
        val_f1_lst.append(mac_val_f1)
        val_end_time = time.time()
        val_time = round(val_end_time-val_start_time, 2)
        logger.info('{} epoch [{}/{}] time: {}s val loss: {} accuracy: {} f1: {}'.format(
            script_name, epoch, args.epoch, val_time, round(avg_val_loss, 4), round(avg_val_acc, 4), round(mac_val_f1, 4)))
        # swap and save the best metric
        if avg_val_acc > best_acc:
            best_acc, best_acc_epoch, best_acc_wts, best_acc_val_labels_lst, best_acc_val_pred_lst = \
                best_swap(avg_val_acc, epoch, net, val_labels_lst, val_predicted_lst)
        if mac_val_f1 > best_f1_mac:
            best_f1_mac, best_f1_epoch, best_f1_wts, best_f1_val_labels_lst, best_f1_val_pred_lst = \
                best_swap(mac_val_f1, epoch, net, val_labels_lst, val_predicted_lst)
    # save best weights
    save_best_weights(net, best_acc_wts, args.out_path, 'acc', best_acc_epoch, best_acc, logger)
    save_best_weights(net, best_f1_wts, args.out_path, 'f1', best_f1_epoch, best_f1_mac, logger)
    # calculate classification report and plot class analysis curves for different metrics
    label_names_str = [label_name.decode() for label_name in label_names]
    # accuracy
    classify_report(best_acc_val_labels_lst, best_acc_val_pred_lst, label_names_str, logger, args.out_path, 'acc')
    per_class_metric(best_acc_val_labels_lst, best_acc_val_pred_lst, label_names_str, val_data_size, logger,
                     args.out_path, 'acc')
    # macro f1
    classify_report(best_f1_val_labels_lst, best_f1_val_pred_lst, label_names_str, logger, args.out_path, 'f1')
    per_class_metric(best_f1_val_labels_lst, best_f1_val_pred_lst, label_names_str, val_data_size, logger,
                     args.out_path, 'f1')
    if args.redistribute_class:
        gen_199_classify_report(best_acc_val_labels_lst, best_acc_val_pred_lst, label_names_str, logger, args.out_path,
                                'acc')
        gen_199_classify_report(best_f1_val_labels_lst, best_f1_val_pred_lst, label_names_str, logger, args.out_path,
                                'f1')
    # plot process curves
    process_curves(args.epoch, train_loss_lst, val_loss_lst, train_acc_lst, val_acc_lst,
                   train_precision_lst, val_precision_lst, train_recall_lst, val_recall_lst,
                    train_f1_lst, val_f1_lst, best_acc, best_acc_epoch, best_f1_mac, best_f1_epoch, args.out_path)
    # total processing time
    time_end = time.time()
    total_time = round(time_end-time_start, 2)
    logger.info('Total processing time is {}s'.format(total_time))

def train_val_net_per_file(net, dataset='HCP', data_transform=None, running_transform=None):
    """train and validation of the network"""
    time_start = time.time()
    # train_num_batch = train_data_size / args.train_batch_size
    train_num_batch = None # unknown now because we did not load all the files into train_iter once
    train_data_size = 0
    val_num_batch = val_data_size / args.val_batch_size
    # save training and validating process data
    train_loss_lst, val_loss_lst, train_acc_lst, val_acc_lst, \
    train_precision_lst, val_precision_lst, train_recall_lst, val_recall_lst, \
    train_f1_lst, val_f1_lst = [], [], [], [], [], [], [], [], [], []
    # save weights with best metrics
    best_acc, best_f1_mac = 0, 0
    best_acc_epoch, best_f1_epoch = 1, 1
    best_acc_wts, best_f1_wts = None, None
    best_acc_val_labels_lst, best_f1_val_labels_lst = [], []
    best_acc_val_pred_lst, best_f1_val_pred_lst = [], []

    if dataset=='HCP':
        feat_h5_prefix=os.path.join(args.input_path, 'HCP_featMatrix_fold_')
        label_h5_prefix=os.path.join(args.input_path, 'HCP_labelMatrix_fold_')
    else:
        # ORG
        feat_h5_prefix=os.path.join(args.input_path, 'sf_clusters_train_featMatrix_')
        label_h5_prefix=os.path.join(args.input_path, 'sf_clusters_train_label_')
    
    # initialize
    train_dataset = None
    train_loader = None
    train_fold_ls = list(range(args.k_fold))
    train_fold_ls = [x for x in train_fold_ls if x+1!=num_fold]
    logger.info('use {} fold as train data'.format([x+1 for x in train_fold_ls]))
    logger.info('use {} fold as validation data'.format(num_fold))
    for epoch in range(args.epoch):
        train_start_time = time.time()
        epoch += 1
        total_train_loss, total_val_loss = 0, 0
        train_labels_lst, train_predicted_lst = [], []
        total_train_correct, total_val_correct = 0, 0
        val_labels_lst, val_predicted_lst = [], []
        
        for fold_counter, train_fold in enumerate(train_fold_ls):
            if (train_dataset is not None and len(train_fold_ls)>1) or train_dataset is None:
                train_dataset = None
                train_dataset = h5Dataset(feat_h5_prefix+str(train_fold+1)+'.h5', label_h5_prefix+str(train_fold+1)+'.h5', 
                                      logger=logger, transform=data_transform, dataset_name=dataset)
            if (train_loader is not None and len(train_fold_ls)>1) or train_loader is None:
                train_loader = None
                train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch_size,shuffle=True)
            len_iters = len(train_loader)
            if train_num_batch is None:
                train_data_size += len(train_dataset)
            # training
            for i, data in enumerate(train_loader, 0):
                points, label = data  # points [B, N, 3]
                label = label[:, 0]  # [B,1] rank2 to (, B) rank1
                if running_transform:
                    points = torch.from_numpy(running_transform(points.numpy()).astype(np.float32))
                points, label = points.to(device), label.to(device)
                optimizer.zero_grad()
                net = net.train()
                pred = net(points)
                loss = criterion(pred, label)
                loss.backward()
                optimizer.step()
                if not isinstance(pred, torch.Tensor):
                    pred = pred[0]
                if args.scheduler == 'wucd':
                    scheduler.step(epoch-1 + (fold_counter+i/len_iters)/len(train_fold_ls))
                _, pred_idx = torch.max(pred, dim=1)
                correct = pred_idx.eq(label.data).cpu().sum()
                # for calculating training accuracy and loss
                total_train_correct += correct.item()
                total_train_loss += loss.item()
                # for calculating training weighted and macro metrics
                label = label.cpu().detach().numpy().tolist()
                train_labels_lst.extend(label)
                pred_idx = pred_idx.cpu().detach().numpy().tolist()
                train_predicted_lst.extend(pred_idx)

        # train accuracy loss
        avg_train_acc = total_train_correct / float(train_data_size)
        if train_num_batch is None:
            train_num_batch = train_data_size / args.train_batch_size
        avg_train_loss = total_train_loss / float(train_num_batch)
        train_acc_lst.append(avg_train_acc)
        train_loss_lst.append(avg_train_loss)

        if args.scheduler == 'step':
            scheduler.step()
        elif args.scheduler == 'ReduceLROnPlateau':
            scheduler.step(avg_train_acc)


        # train macro p, r, f1
        mac_train_precision, mac_train_recall, mac_train_f1 = calculate_prec_recall_f1(train_labels_lst, train_predicted_lst)
        train_precision_lst.append(mac_train_precision)
        train_recall_lst.append(mac_train_recall)
        train_f1_lst.append(mac_train_f1)
        train_end_time = time.time()
        train_time = round(train_end_time-train_start_time, 2)
        logger.info('{} epoch [{}/{}] time: {}s train loss: {} accuracy: {} f1: {}'.format(
            script_name, epoch, args.epoch, train_time, round(avg_train_loss, 4), round(avg_train_acc, 4), round(mac_train_f1, 4)))

        # validation
        train_dataset = None
        train_loader = None
        with torch.no_grad():
            val_start_time = time.time()
            for j, data in (enumerate(val_loader, 0)):
                points, label = data
                label = label[:, 0]
                if running_transform:
                    points = torch.from_numpy(running_transform(points.numpy()).astype(np.float32))
                points, label = points.to(device), label.to(device)
                net = net.eval()
                try:
                    pred = net(points)
                except RuntimeError as exception:
                    if "out of memory" in str(exception):
                        logger.info("WARNING: out of memory")
                        if hasattr(torch.cuda, 'empty_cache'):
                            torch.cuda.empty_cache()
                    else:
                        raise exception
                    
                loss = criterion(pred, label)
                if not isinstance(pred, torch.Tensor):
                    pred = pred[0]
                _, pred_idx = torch.max(pred, dim=1)
                correct = pred_idx.eq(label.data).cpu().sum()
                # for calculating validation accuracy and loss
                total_val_correct += correct.item()
                total_val_loss += loss.item()
                # for calculating validation weighted and macro metrics
                label = label.cpu().detach().numpy().tolist()
                val_labels_lst.extend(label)
                pred_idx = pred_idx.cpu().detach().numpy().tolist()
                val_predicted_lst.extend(pred_idx)
        # calculate the validation accuracy and loss for the epoch
        avg_val_acc = total_val_correct / float(val_data_size)
        avg_val_loss = total_val_loss / float(val_num_batch)
        val_acc_lst.append(avg_val_acc)
        val_loss_lst.append(avg_val_loss)
        # calculate the validation macro metrics
        mac_val_precision, mac_val_recall, mac_val_f1 = calculate_prec_recall_f1(val_labels_lst, val_predicted_lst)
        val_precision_lst.append(mac_val_precision)
        val_recall_lst.append(mac_val_recall)
        val_f1_lst.append(mac_val_f1)
        val_end_time = time.time()
        val_time = round(val_end_time-val_start_time, 2)
        logger.info('{} epoch [{}/{}] time: {}s val loss: {} accuracy: {} f1: {}'.format(
            script_name, epoch, args.epoch, val_time, round(avg_val_loss, 4), round(avg_val_acc, 4), round(mac_val_f1, 4)))
        # swap and save the best metric
        if avg_val_acc > best_acc:
            best_acc, best_acc_epoch, best_acc_wts, best_acc_val_labels_lst, best_acc_val_pred_lst = \
                best_swap(avg_val_acc, epoch, net, val_labels_lst, val_predicted_lst)
        if mac_val_f1 > best_f1_mac:
            best_f1_mac, best_f1_epoch, best_f1_wts, best_f1_val_labels_lst, best_f1_val_pred_lst = \
                best_swap(mac_val_f1, epoch, net, val_labels_lst, val_predicted_lst)
    # save best weights
    save_best_weights(net, best_acc_wts, args.out_path, 'acc', best_acc_epoch, best_acc, logger)
    save_best_weights(net, best_f1_wts, args.out_path, 'f1', best_f1_epoch, best_f1_mac, logger)
    # calculate classification report and plot class analysis curves for different metrics
    label_names_str = [label_name.decode() for label_name in label_names]
    # accuracy
    classify_report(best_acc_val_labels_lst, best_acc_val_pred_lst, label_names_str, logger, args.out_path, 'acc')
    per_class_metric(best_acc_val_labels_lst, best_acc_val_pred_lst, label_names_str, val_data_size, logger,
                     args.out_path, 'acc')
    # macro f1
    classify_report(best_f1_val_labels_lst, best_f1_val_pred_lst, label_names_str, logger, args.out_path, 'f1')
    per_class_metric(best_f1_val_labels_lst, best_f1_val_pred_lst, label_names_str, val_data_size, logger,
                     args.out_path, 'f1')
    if args.redistribute_class:
        gen_199_classify_report(best_acc_val_labels_lst, best_acc_val_pred_lst, label_names_str, logger, args.out_path,
                                'acc')
        gen_199_classify_report(best_f1_val_labels_lst, best_f1_val_pred_lst, label_names_str, logger, args.out_path,
                                'f1')
    # plot process curves
    process_curves(args.epoch, train_loss_lst, val_loss_lst, train_acc_lst, val_acc_lst,
                   train_precision_lst, val_precision_lst, train_recall_lst, val_recall_lst,
                    train_f1_lst, val_f1_lst, best_acc, best_acc_epoch, best_f1_mac, best_f1_epoch, args.out_path)
    # total processing time
    time_end = time.time()
    total_time = round(time_end-time_start, 2)
    logger.info('Total processing time is {}s'.format(total_time))

    
if __name__ == '__main__':
    # Variable Space
    parser = argparse.ArgumentParser(description="Train stage 1 model",
                                     epilog="by Tengfei Xue txue4133@uni.sydney.edu.au")
    # Paths
    parser.add_argument('--input_path', type=str, default='./TrainData/outliers_data/DEBUG_kp0.1/h5_np15/',
                        help='Input graph data and labels')
    parser.add_argument('--out_path_base', type=str, default='./ModelWeights',
                        help='Save trained models')
    parser.add_argument('--load_model_base', type=str, default=None, required=False,
                        help='Saved trained models for loading')


    # load pretrained model
    parser.add_argument('--load_model_name', type=str, default=None, required=False, help='model for loading in trainging')
    parser.add_argument('--load_model_layers', type=str, default=None, required=False, help='model layers of load_model for loading')
    parser.add_argument('--load_in_model_layers', type=str, default=None, required=False, help='model layers of training model according')
    parser.add_argument('-load_whole', type=bool, default=False, required=False, help='load all parameters')
    parser.add_argument('-load_freeze', default=False, action='store_true', help='freeze loaded layers')

    # parameters
    parser.add_argument('--model_name', type=str, default='PointNetCls', help='model for trainging')
    parser.add_argument('--dataset', type=str, default='HCP', help='dataset for trainging')
    parser.add_argument('-RAS_3D', default=False, action='store_true', help='get DeepWMA 3D RAS feature of dataset')
    parser.add_argument('--loss', type=str, default='nll_loss', help='loss of training')
    parser.add_argument('--in_dim', type=int, default=3, help='dimension of data. PointCloud:3, sig_feat: 12 if sig for 3 dim(custom) 20 if sig for 4 dim(time augcustom)')
    parser.add_argument('--stream', type=int, default=15, help='stream size of data.')
    parser.add_argument('--dict_feat_size', type=int, default=256, help='if use loss function LabelDictionaryAndCrossEntropy. It should be specified.')
    parser.add_argument('--k_fold', type=int, default=5, help='fold of all data')
    parser.add_argument('--val_fold', type=int, default=None, help='fold number of validation data')
    parser.add_argument('--num_workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--opt', type=str, required=True, help='type of optimizer')
    parser.add_argument('--weight_decay', type=float, default=5e-2, help='weight decay for Adam')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum for SGD')
    parser.add_argument('--scheduler', type=str, default='step', help='type of learning rate scheduler')
    parser.add_argument('--step_size', type=int, default=20, help='Period of learning rate decay')
    parser.add_argument('--decay_factor', type=float, default=0.5, help='Multiplicative factor of learning rate decay')
    parser.add_argument('--T_0', type=int, default=10, help='Number of iterations for the first restart (for wucd)')
    parser.add_argument('--T_mult', type=int, default=2, help='A factor increases Ti after a restart (for wucd)')
    parser.add_argument('--train_batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--val_batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--epoch', type=int, default=10, help='the number of epochs')
    parser.add_argument('--best_metric', type=str, default='f1', help='evaluation metric')
    parser.add_argument('--eval_fold_zero', default=False, action='store_true', help='eval on fold 0, train on fold 1 2 3 4')
    parser.add_argument('--redistribute_class', default=False, action='store_true',
                        help="redistribute classes to 199 classes when generate classification reports")
    parser.add_argument('-train_per_file', default=False, action='store_true',
                        help="if multi fold in training set, load one file once. Load dataset every epoch. Exchange time for memory.")

    args = parser.parse_args()

    args.manualSeed = 0  # fix seed
    print("Random Seed: ", args.manualSeed)
    fix_seed(args.manualSeed)

    script_name = '<train_stage1>'

    args.input_path = unify_path(args.input_path)
    args.out_path_base = unify_path(args.out_path_base)

    if args.eval_fold_zero:
        fold_lst = [0]
    elif args.val_fold is not None and 0<=args.val_fold<args.k_fold:
        fold_lst = [args.val_fold]
    else:
        fold_lst = [i for i in range(args.k_fold)]

    # train_loader, val_loader = None, None # initialize 
    val_loader = None
    for num_fold in fold_lst:
        num_fold = num_fold + 1
        args.out_path = os.path.join(args.out_path_base, str(num_fold))
        makepath(args.out_path)

        # Record the training process and values
        logger = create_logger(args.out_path)
        logger.info('=' * 55)
        logger.info(args)
        logger.info('=' * 55)
        logger.info('Implement {} fold experiment'.format(num_fold))

        # load data
        # if train_loader is not None:
            # train_loader = None # save RAM
        if val_loader is not None:
            val_loader = None # save RAM
        
        if not args.train_per_file:
            train_loader, val_loader, label_names, \
            num_classes, train_data_size, val_data_size = load_data(dataset=args.dataset)
        else:
            val_loader, label_names, num_classes, val_data_size = load_data_val(dataset=args.dataset)

        running_transform = None
        if args.RAS_3D:
            running_transform =_feat_to_3D

        
        # load pretrained model
        if args.load_model_base is not None:
            load_model_weight = os.path.join(unify_path(args.load_model_base), str(num_fold), 'best_{}_model.pth'.format(args.best_metric))
            if args.load_whole:
                classifier = load_model_weights(load_model_weight, model_name=args.model_name, num_classes=num_classes)
            else:
                if args.load_model_name is not None and args.load_model_layers is not None and args.load_in_model_layers is not None:
                    load_model_layers = args.load_model_layers.strip().split(',')
                    load_in_model_layers = args.load_in_model_layers.strip().split(',')
                    if len(load_model_layers)==len(load_in_model_layers):
                        classifier = get_model(model_name=args.model_name, num_classes=num_classes, dict_feat_size=args.dict_feat_size, stream=args.stream, input_dim=args.in_dim)
                        para_net = load_model_weights(load_model_weight, model_name=args.load_model_name, num_classes=num_classes, stream=args.stream)
                        classifier = load_and_freeze_layers(para_net,classifier,dict(zip(load_model_layers, load_in_model_layers)), freeze=args.load_freeze)
                    else:
                        raise ValueError('number of load_model_layers and load_in_model_layers should be the same')
        else:
            # model setting
            classifier = get_model(model_name=args.model_name, num_classes=num_classes, dict_feat_size=args.dict_feat_size, stream=args.stream, input_dim=args.in_dim)
        
        criterion = get_criterion(loss_name=args.loss, num_classes=num_classes, dict_feat_size=args.dict_feat_size, device=device)
        # print(device)

        # optimizers
        # filter(lambda p: p.requires_grad, classifier.parameters())
        if args.opt == 'Adam':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, classifier.parameters()), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
        elif args.opt == 'SGD':
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, classifier.parameters()), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        elif args.opt == 'LARS':
            base_optimizer = optim.Adam(filter(lambda p: p.requires_grad, classifier.parameters()), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
            optimizer = LARS(optimizer=base_optimizer, eps=1e-8, trust_coef=0.001)
        else:
            raise ValueError('Please input valid optimizers Adam | SGD | LARS')
        # schedulers
        if args.scheduler == 'step':
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.decay_factor)
        elif args.scheduler == 'wucd':
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.T_0, T_mult=args.T_mult)
        elif args.scheduler=='ReduceLROnPlateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=0.5, verbose=True)
        else:
            raise ValueError('Please input valid schedulers step | wucd | ReduceLROnPlateau')

        classifier.to(device)
        # train and eval net
        if not args.train_per_file:
            train_val_net(classifier, running_transform=running_transform)
        else:
            train_val_net_per_file(classifier, running_transform=running_transform, dataset=args.dataset)

    # clean the logger
    logger.handlers.clear()

    # Generate .pickle file of stage 1 parameters
    num_swm_stage1 = len([name.decode() for name in label_names if 'swm' in name.decode()])
    stage1_params_dict = {'stage1_num_class': num_classes, 'num_swm_stage1': num_swm_stage1, 'fold_lst': fold_lst}
    with open(os.path.join(args.out_path_base, 'stage1_params.pickle'), 'wb') as f:
        pickle.dump(stage1_params_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()
    
    # average metric
    num_files = len(fold_lst)
    calculate_average_metric(args.out_path_base, num_files, args.best_metric, args.redistribute_class)
