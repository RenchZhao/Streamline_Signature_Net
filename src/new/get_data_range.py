import sys
sys.path.append('../')

import argparse
import os
import pickle
import time
import h5py

import torch
import torch.nn.parallel

import torch.utils.data

from utils.dataset import HCPDataset # ORGDataset

from utils.logger import create_logger

import numpy as np


if __name__ == "__main__":
    logger = create_logger("./")
    inputDir = '/path/to/your/h5/files'
    train_dataset = HCPDataset(
        root=inputDir,
        logger=logger,
        num_fold=6,
        k=5,
        split='train')
    
    a = -10000
    b = 10000
    counter = 0
    c = 0
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1024,
        shuffle=False)
    for i, input_data in enumerate(train_loader):
        data, label = input_data
        max_data = data.max()
        min_data = data.min()
        if max_data>a:
            a=max_data
        if min_data<b:
            b=min_data
        c = c*(counter/(counter+data.shape[0]))
        counter+=data.shape[0]
        c = c + data.sum()/counter

    # a = max(train_dataset.features)
    # b = min(train_dataset.features)
    # c = np.mean(train_dataset.features)
    print(a)
    print(b)
    print(c)


