import argparse
import os
import torch
import h5py

from utils.model import get_model

from utils.funcs import unify_path, makepath
from utils.dataset import h5Dataset, HCPDataset, ORGDataset, _feat_to_3D

from matplotlib import pyplot as plt
import numpy as np

def load_model_weights(model_weight_path, model_name='PointNetCls', num_classes=2, stream=15):
    model = get_model(model_name, num_classes)
    # weight_path = os.path.join(weight_path_base, str(num_fold), 'best_{}_model.pth'.format(args.best_metric))
    model.load_state_dict(torch.load(model_weight_path, weights_only=True, map_location='cpu'))
    return model

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

    val_dataset = h5Dataset(feat_h5_prefix+str(num_fold)+'.h5', label_h5_prefix+str(num_fold)+'.h5', logger=None, transform=transform, dataset_name=dataset)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.val_batch_size,
        shuffle=False)

    val_data_size = len(val_dataset)
    print('The validation data size is:{}'.format(val_data_size))
    num_classes = len(val_dataset.label_names)
    print('The number of classes is:{}'.format(num_classes))

    # load label names
    label_names = val_dataset.obtain_label_names()
    label_names_h5 = h5py.File(os.path.join(args.out_path, 'label_names.h5'), 'w')

    label_names_h5['y_names'] = list(label_names)
    print('The label names are: {}'.format(str(label_names)))

    return  val_loader, label_names, num_classes, val_data_size


def update_mean_feat(feat_ls, num_ls, X_label, batch_feat):
    for i in range(len(X_label)):
        label = int(X_label[i])
        num_ls[label] += 1
        feat_ls[label] = feat_ls[label]/num_ls[label]*(num_ls[label]-1) + batch_feat[i]/num_ls[label] # 尽量数值稳定地更新平均值
    
# def capture_features(model, layer_name, shape, data_loader, num_classes, device):
#     """capture model features"""
#     print(layer_name)
#     model.eval()
#     model.to(device)
#     feat_ls = [torch.zeros(shape)]*num_classes
#     num_ls = [0]*num_classes
#     to_device(feat_ls, device)
#     # to_device(num_ls, device)
#     # num = 0
#     for i, (X, X_label) in enumerate(data_loader):
#         # print(i)
#         # print(X.shape)
#         # print(shape)
#         # num += X.shape[0]
#         X = X.to(device)
#         X_label = X_label.to(device)
#         with torch.no_grad():
#             batch_feat = torch.cat(capture_model_layer_output(X, X_label, model, layer_name=layer_name),dim=0)
#             # print(batch_feat.shape)
#             batch_feat = batch_feat.reshape((X.shape[0],*shape))
#             # ls[X_label] = ls[X_label]*((num-X.shape[0])/num) + batch_feat.sum(dim=0)/num
#             # ls 对应的label更新对应的batch_feat
#             update_mean_feat(feat_ls, num_ls, X_label, batch_feat)
#             batch_feat = None # 删除缓存
#     return feat_ls

def capture_features(model, layer_name, shape, data_loader, num_classes, device):
    """capture model features with memory leak fixes by doubao"""
    print(layer_name)
    model.eval()
    model.to(device)
    
    # 正确初始化特征列表，每个元素是独立的张量
    feat_ls = [torch.zeros(shape, device=device) for _ in range(num_classes)]
    num_ls = [0] * num_classes
    
    # 注册钩子函数的引用，用于后续移除
    hook_handle = None
    
    for i, (X, X_label) in enumerate(data_loader):
        X = X.to(device)
        X_label = X_label.to(device)
        
        # 注册钩子
        for name, module in model.named_modules():
            if name == layer_name:
                # 创建特征输出列表
                feat_out_hook = []
                
                # 定义钩子函数
                def hook_capture_feat_out(module, feat_in, feat_out):
                    feat_out_hook.append(feat_out.clone().detach())
                    return None
                
                # 注册钩子并保存句柄
                hook_handle = module.register_forward_hook(hook_capture_feat_out)
                break
        
        # 获取模型输出并捕获特征
        with torch.no_grad():
            _ = model(X)
            
            # 处理捕获的特征
            if feat_out_hook:
                batch_feat = torch.cat(feat_out_hook, dim=0)
                batch_feat = batch_feat.reshape((X.shape[0], *shape))
                update_mean_feat(feat_ls, num_ls, X_label, batch_feat)
                
                # 释放batch_feat和feat_out_hook占用的内存
                del batch_feat, feat_out_hook
                torch.cuda.empty_cache()  # 清理缓存
        
        # 移除钩子
        if hook_handle:
            hook_handle.remove()
            hook_handle = None
    
    return feat_ls

def to_device(X, device):
    if isinstance(X, list):
        for i in range(len(X)):
            X[i] = to_device(X[i], device)
    else:
        X = X.to(device)
    return X

def capture_model_layer_output(X, X_label, model, layer_name, registered=False, device=None):
    """获取当神经网络model的输入数据为X时layer_name层的输出结果"""
    
    def hook_capture_feat_out(module, feat_in, feat_out):
        "用钩子获取模型其中一层输出"
        nonlocal feat_out_hook
        feat_out_hook.append(feat_out.clone().detach())
        return None

    def hook_capture_feat_in(module, feat_in, feat_out):
        "用钩子获取模型其中一层输出"
        nonlocal feat_in_hook
        feat_in_hook.append(feat_in[0].clone().detach())
        return None
    
    feat_out_hook = []
    feat_in_hook = []
    if not registered:
        # 注册钩子函数，获取模型指定层输出
        for (name, module) in model.named_modules():
            if name == layer_name:
                module.register_forward_hook(hook=hook_capture_feat_out)
    if isinstance(model, torch.nn.Module):
        model.eval() # 设置为评估模式
        if not device:
            device = next(iter(model.parameters())).device
    model.to(device)
    X = to_device(X, device)
    X_label = to_device(X_label, device)
    _ = model(X) # hook之中已经获得了输出
    return feat_out_hook

if __name__ == '__main__':

    # Variable Space
    parser = argparse.ArgumentParser(description="visulaize avg features")
    # Paths
    parser.add_argument('--input_path', type=str, default='./TrainData/outliers_data/DEBUG_kp0.1/h5_np15/',
                        help='Input graph data and labels')
    parser.add_argument('--out_path_base', type=str, default='./Visusalization',
                        help='Save trained models')

    # parameters
    parser.add_argument('--model_name', type=str, default='PointNetCls', help='model for trainging')
    parser.add_argument('--model_weight_path', help='Saved model weight for inference.')
    parser.add_argument('--dataset', type=str, default='HCP', help='dataset for trainging')
    parser.add_argument('-RAS_3D', default=False, action='store_true', help='get DeepWMA 3D RAS feature of dataset')
    parser.add_argument('--in_dim', type=int, default=3, required=False, help='dimension of data. PointCloud:3, sig_feat: 12 if sig for 3 dim(custom) 20 if sig for 4 dim(time augcustom)')
    parser.add_argument('--stream', type=int, default=15, required=False, help='stream size of data.')
    parser.add_argument('--dict_feat_size', type=int, default=256, help='if use loss function LabelDictionaryAndCrossEntropy. It should be specified.')
    parser.add_argument('--k_fold', type=int, default=5, help='fold of all data')
    parser.add_argument('--val_fold', type=int, default=None, help='fold number of validation data')
    parser.add_argument('--num_workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--train_batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--val_batch_size', type=int, default=128, help='batch size')
    

    args = parser.parse_args()

    args.input_path = unify_path(args.input_path)
    args.out_path_base = unify_path(args.out_path_base)

    fold_lst = [i for i in range(args.k_fold)]

    # train_loader, val_loader = None, None # initialize 
    val_loader = None
    for num_fold in fold_lst:
        num_fold = num_fold + 1
        args.out_path = os.path.join(args.out_path_base, str(num_fold))
        makepath(args.out_path)

        # load data
        # if train_loader is not None:
            # train_loader = None # save RAM
        if val_loader is not None:
            val_loader = None # save RAM
        
        val_loader, label_names, num_classes, val_data_size = load_data_val(dataset=args.dataset)

        running_transform = None
        if args.RAS_3D:
            running_transform =_feat_to_3D

        # model setting
        device='cpu'
        layer_name = ['sig_layer1.mask_sig_blocks.1.0', 'sig_layer2.mask_sig_blocks.1.0', 'sig_layer1.mask_sig_blocks.1.0', 'sig_layer2.mask_sig_blocks.1.0']

        
        model = load_model_weights(os.path.join(args.model_weight_path, str(num_fold), 'best_acc_model.pth'), model_name=args.model_name, num_classes=len(label_names), stream=args.stream)
        model.to(device)

        kernels=[3,5,7,9]
        for i in range(1,3): # tr
            for j in range(4): # Inception kernels
                layer_name = 'sig_layer{}.mask_sig_blocks.{}.0'.format(i,j)
                shape = ((args.stream-kernels[j]) + 1, )
                feat = capture_features(model, layer_name, shape, val_loader, num_classes, device)
                feat = torch.stack(feat).transpose(0,1)

                # save data
                feat = feat.cpu().numpy()
                np.save('{}/{}_{}.npy'.format(args.out_path, layer_name, j), feat)

                # show 
                plt.figure()
                plt.imshow(feat)
                plt.title('{} attention weight feature of window length={}'.format('forward' if i==1 else 'flipped', kernels[j]))
                if not os.path.exists(args.out_path):
                    os.makedirs(args.out_path)
                plt.savefig('{}/{}_{}.png'.format(args.out_path, layer_name, j))
        

        # for name,layer in model.named_modules():
        #     print(name,layer)


        # kernel = [3,5,7,9]
        # for i in range(len(kernel)):
        #     win_num = (args.stream-kernel[i])//1 + 1
        #     shape = (win_num, 1)
        #     capture_features(model, layer_name, shape, val_loader, num_classes, device)
        
        
        
