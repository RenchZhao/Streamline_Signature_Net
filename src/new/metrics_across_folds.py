import numpy as np
import h5py
import os
import sys
import copy
import torch
import matplotlib.ticker as mtick
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, precision_recall_fscore_support, accuracy_score
from decimal import Decimal, ROUND_HALF_UP, ROUND_HALF_EVEN


sys.path.append('..')
from utils.logger import create_logger
from utils.funcs import round_decimal_percentage, round_decimal

import time


def calculate_prec_recall_f1(labels_lst, predicted_lst):
    # Beta: The strength of recall versus precision in the F-score. beta == 1.0 means recall and precision are equally important, that is F1-score
    mac_precision, mac_recall, mac_f1, _ = precision_recall_fscore_support(y_true=labels_lst, y_pred=predicted_lst, beta=1.0, average='macro')
    return mac_precision, mac_recall, mac_f1


def classify_report(labels_lst, predicted_lst, label_names, logger, out_path, metric_name):
    """Generate classification performance report"""
    #cls_report = classification_report(y_true=labels_lst, y_pred=predicted_lst, digits=5, target_names=label_names)
    cls_report = classification_report(y_true=labels_lst, y_pred=predicted_lst, digits=5, labels=list(range(len(label_names))), target_names=label_names)
    logger.info('=' * 55)
    logger.info('Best {} classification report:\n{}'.format(metric_name, cls_report))
    logger.info('=' * 55)
    logger.info('\n')

    if 'test' in metric_name:
        test_res = h5py.File(out_path, "w")
        test_res['val_predictions'] = predicted_lst
        test_res['val_labels'] = labels_lst
        test_res['label_names'] = label_names
        test_res['classification_report'] = cls_report
    else:
        val_res = h5py.File(os.path.join(out_path, 'entire_data_validation_results_best_{}.h5'.format(metric_name)), "w")
        val_res['val_predictions'] = predicted_lst
        val_res['val_labels'] = labels_lst
        val_res['label_names'] = label_names
        val_res['classification_report'] = cls_report





def _mean_std_across_folds(accuracy_array, precision_array, recall_array, f1_array,
                           h5_base_path, num_average_files, metric_name, logger):
    """Calculate mean and standard deviation for folds"""
    avg_acc = round_decimal(np.mean(accuracy_array)* 100, decimal=3)
    avg_precision = round_decimal(np.mean(precision_array* 100), decimal=3)
    avg_recall = round_decimal(np.mean(recall_array)* 100, decimal=3)
    avg_f1 = round_decimal(np.mean(f1_array)* 100, decimal=3)

    std_acc = round_decimal(np.std(accuracy_array)* 100, decimal=3)
    std_precision = round_decimal(np.std(precision_array)* 100, decimal=3)
    std_recall = round_decimal(np.std(recall_array)* 100, decimal=3)
    std_f1 = round_decimal(np.std(f1_array)* 100, decimal=3)

    logger.info('The number of experiment implementations is {}'.format(num_average_files))
    logger.info('Use the weight with best {} for each fold'.format(metric_name))
    logger.info('='*55)

    logger.info('The average accuracy for {} is {} % and standard deviation is {} %\n'.format(h5_base_path, avg_acc, std_acc))
    logger.info('The average macro precision for {} is {} % and standard deviation is {} %\n'.format(h5_base_path, avg_precision, std_precision))
    logger.info('The average macro recall for {} is {} % and standard deviation is {} %\n'.format(h5_base_path, avg_recall, std_recall))
    logger.info('The average macro f1 for {} is {} % and standard deviation is {} %\n'.format(h5_base_path, avg_f1, std_f1))
    logger.info('='*55)


def calculate_average_metric(h5_base_path, num_average_files, metric_name, redistribute_class):
    accuracy_array = np.zeros(num_average_files)
    precision_array = np.zeros(num_average_files)
    recall_array = np.zeros(num_average_files)
    f1_array = np.zeros(num_average_files)
    logger = create_logger(h5_base_path, '{}_Whole_dataset-MeanStd_Results'.format(metric_name))
    # logger.info('Not calculating stage 1 or stage 2')
    if redistribute_class:
        logger.info('Redistribute class')
    for i in range(num_average_files):
        if redistribute_class:
            h5_path = os.path.join(h5_base_path, str(i + 1), 'validation_results_best_{}_redistribute.h5'.format(metric_name))
        else:
            h5_path = os.path.join(h5_base_path, str(i+1), 'entire_data_validation_results_best_{}.h5'.format(metric_name))
        results = h5py.File(h5_path, 'r')
        labels_lst = results['val_labels']
        predicted_lst = results['val_predictions']
        label_names = results['label_names']
        mac_precision, mac_recall, mac_f1 = calculate_prec_recall_f1(labels_lst, predicted_lst)
        mac_acc = accuracy_score(labels_lst, predicted_lst)

        accuracy_array[i] = mac_acc
        precision_array[i] = mac_precision
        recall_array[i] = mac_recall
        f1_array[i] = mac_f1

    logger.info('The number of classes is {}'.format(len(label_names)))
    _mean_std_across_folds(accuracy_array, precision_array, recall_array, f1_array,
                           h5_base_path, num_average_files, metric_name, logger)

    
    

if __name__=='__main__':
    bases = ['/path/to/ORG_base', '/path/to/HCP_base']
    for base in bases:
        for model in os.listdir(base):
            h5_base_path = os.path.join(base, model)
            print(h5_base_path)
            num_average_files=5

            calculate_average_metric(h5_base_path, num_average_files, 'f1', False)
            # time.sleep(1)

            calculate_average_metric(h5_base_path, num_average_files, 'acc', False)
            # time.sleep(1)
