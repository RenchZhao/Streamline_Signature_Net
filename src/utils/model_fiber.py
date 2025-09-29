import torch
import torch.nn as nn

import torch.nn.functional as F


import torch

class BundleAvgStream(nn.Module):
    def __init__(self, stream, out_dimension, dist_type='mdf', input_dim=3):
        super().__init__()
        self.avg_stream = nn.Parameter(torch.randn(out_dimension, stream, input_dim))
        self.dist_type = dist_type


    def forward(self, x, log_softmax=True):
        ans=[]
        if self.dist_type == 'mam':
            for i in range(self.avg_stream.shape[0]):
                ans.append(mam_distance(x, self.avg_stream[i]))
        elif self.dist_type == 'mdf':
            for i in range(self.avg_stream.shape[0]):
                ans.append(mdf_distance(x, self.avg_stream[i]))
        else:
            raise ValueError("dist_type must in 'mam' | 'mdf'")
        if log_softmax:
            return F.log_softmax(-torch.stack(ans,dim=1),dim=1)
        else:
            return torch.stack(ans,dim=1)



def mdf_distance(a, b, p=2):
    """
    计算批量纤维流线与单条纤维流线之间的MDF距离
    
    参数:
    a (torch.Tensor): 形状为(M,N,3)的张量，表示M条纤维流线
    b (torch.Tensor): 形状为(N,3)的张量，表示单条纤维流线
    p (int): 范数类型，1表示曼哈顿距离(默认)，2表示欧氏距离
    
    返回:
    torch.Tensor: 形状为(M,)的张量，表示每条流线与b的MDF距离
    """
    # 扩展b的维度以匹配a的批量维度
    b_expanded = b.unsqueeze(0)  # 形状变为(1,N,3)
    
    # 计算正向距离
    direct_dist = torch.mean(torch.norm(a - b_expanded, p=p, dim=2), dim=1)
    
    # 计算反向距离（将b翻转）
    b_flipped = torch.flip(b, dims=[0]).unsqueeze(0)  # 形状变为(1,N,3)
    flipped_dist = torch.mean(torch.norm(a - b_flipped, p=p, dim=2), dim=1)
    
    # 返回最小距离
    return torch.min(direct_dist, flipped_dist)

def mam_distance(a, b, variant='mean', p=2):
    """
    计算批量纤维流线与单条纤维流线之间的MAM距离
    
    参数:
    a (torch.Tensor): 形状为(M,N,3)的张量，表示M条纤维流线
    b (torch.Tensor): 形状为(N,3)的张量，表示单条纤维流线
    variant (str): MAM变体类型，可选值为'min'、'max'或'mean'（默认值）
    p (int): 范数类型，1表示曼哈顿距离，2表示欧氏距离（默认）
    
    返回:
    torch.Tensor: 形状为(M,)的张量，表示每条流线与b的MAM距离
    """
    # 计算a中每个点到b的最小距离的平均值
    dists_a_to_b = torch.cdist(a.reshape(-1, 3), b, p=p).reshape(a.shape[0], a.shape[1], b.shape[0])
    # min_dists_a,_ = dists_a_to_b.min(dim=2)#.squeeze(2)
    min_dists_a,_ = torch.min(dists_a_to_b, dim=2)#.squeeze(2) #小心多返回的下标

    d_mean_a = min_dists_a.mean(dim=1)
    
    # 计算b中每个点到a的最小距离的平均值（对每条a单独计算）
    d_mean_b_list = []
    # for i in range(a.shape[0]):
    #     dists_b_to_ai = torch.cdist(b, a[i], p=p)
    #     min_dists_b,_ = torch.min(dists_b_to_ai, dim=1)#.squeeze(1)
    #     d_mean_b_list.append(torch.mean(min_dists_b,dim=0, keepdim=True))
    dists_b_to_a = torch.cdist(b, a.reshape(-1, 3), p=p).reshape(b.shape[0], a.shape[0], a.shape[1])
    min_dists_b,_ = torch.min(dists_b_to_a, dim=2)
    d_mean_b = torch.mean(min_dists_b, dim=0)
    
    # 根据变体类型返回相应的距离
    if variant == 'min':
        return torch.min(d_mean_a, d_mean_b)
    elif variant == 'max':
        return torch.max(d_mean_a, d_mean_b)
    elif variant == 'mean':
        return (d_mean_a + d_mean_b) / 2
    else:
        raise ValueError("Variant must be 'min', 'max', or 'mean'")

if __name__ == '__main__':
    # 生成随机数据
    M = 10  # 批量大小
    N = 100 # 每条流线的点数
    a = torch.randn(M, N, 3)
    b = torch.randn(N, 3)

    # 计算MDF距离
    mdf_distances = mdf_distance(a, b, p=1)
    print(f"MDF distances shape: {mdf_distances.shape}")  # 输出应为(M,)

    # 计算MAM距离
    mam_distances = mam_distance(a, b, variant='mean', p=1)
    print(f"MAM distances shape: {mam_distances.shape}")  # 输出应为(M,)


    layer = BundleAvgStream(stream=N, num_class=73)
    # print(layer(a))
    print(layer(a).shape)

