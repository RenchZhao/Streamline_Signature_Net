
# This is code for Attention Weight in the paper.

# #bug
# Traceback (most recent call last):
#  
#     feat = torch.cat(capture_features(model, layer_name, shape, val_loader, num_classes, device),dim=0).transpose(0,1)
#            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# IndexError: Dimension out of range (expected to be in range of [-1, 0], but got 1)


import torch
import torch.nn as nn
import math
import torch.nn.functional as F
# from .attention import Attention

class product_mask_layer(nn.Module):
    ''' generate mask for product'''
    def __init__(self, gen_mask_layer, mode='mask'):
        # mode: mask. concat, sum
        super().__init__()
        self.gen_mask_layer = gen_mask_layer
        self.mode = mode
    def forward(self, x):
        y = self.gen_mask_layer(x)
        if self.mode=='mask':
            return torch.mul(x, y) #x * y
        elif self.mode=='concat':
            return torch.cat([x, y], dim=-1)
        elif self.mode=='sum':
            return x + y




# class conv_attn_score_no_parameter(nn.Module):
#     def __init__(self, dim, embed_dim=32, num_heads=4, kernels=3): 
#         super().__init__()
#         # assert embed_dim % num_heads == 0, 'embed_dim should be divisible by num_heads'

#         # self.Inception_blocks = []
#         # for i in range(len(kernels)):
#             # self.Inception_blocks.append(nn.Sequential(nn.Conv1d(dim, embed_dim, kernels[i]),cal_attn_mtx(embed_dim, num_heads)))
        
#         # self.para = nn.Parameter(torch.randn(1, (embed_dim-kernels, 1))
        
#         self.conv = nn.Conv1d(dim, embed_dim, kernels)
#         # self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
#         self.attn_mtx = cal_attn_mtx(embed_dim, num_heads)
#     def forward(self, x):
#         # return self.para
    
#         # B,N,C = x.shape
#         x = x.transpose(1,2)
#         x = F.relu(self.conv(x)).transpose(1,2) # [B, N, embed_dim]
#         # cls_tokens = self.cls_token.expand(B, -1, -1)
#         # x = torch.cat((cls_tokens, x), dim=1) # [B, N, embed_dim]
#         x = self.attn_mtx(x) # [B, N, N]
#         x = torch.mean(x, dim=1).unsqueeze(-1)

        
#         # result=[]
#         # for i in range(len(self.Inception_blocks)):
#         #     result.append(self.Inception_blocks[i](x)).transpose(1,2)
#         # return torch.cat(result, dim=1)

#         return x

# it is expected to be the same but its performance is worse than conv_attn_score

class conv_attn_score_classic(nn.Module):
    def __init__(self, dim, embed_dim=32, num_heads=4, kernels=3): 
        super().__init__()
        assert embed_dim % num_heads == 0, 'embed_dim should be divisible by num_heads'
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # self.Inception_blocks = []
        # for i in range(len(kernels)):
            # self.Inception_blocks.append(nn.Sequential(nn.Conv1d(dim, embed_dim, kernels[i]),cal_attn_mtx(embed_dim, num_heads)))
        
        # self.para = nn.Parameter(torch.randn(1, (embed_dim-kernels, 1))
        
        self.conv = nn.Conv1d(dim, embed_dim, kernels)
        self.hidden = nn.Linear(embed_dim, embed_dim)
        # self.attn = nn.Linear(embed_dim, num_heads)
        self.attn = nn.Linear(embed_dim//num_heads, 1)
    def forward(self, x):
        # return self.para
    
        B,N,C = x.shape
        x = x.transpose(1,2)
        x = F.relu(self.conv(x)).transpose(1,2) # [B, N-kernels+1, embed_dim]
        # x = F.relu(self.hidden(x))
        x = self.hidden(x).reshape(B, -1, self.num_heads, self.embed_dim//self.num_heads)
        # x = F.softmax(F.relu(self.attn(x)),dim=1) # [B, N-kernels+1, num_heads]
        x = F.softmax(self.attn(x).squeeze(-1)*(self.embed_dim**-0.5),dim=1) # [B, N-kernels+1, num_heads]
        x = x.mean(dim=2,keepdim=True)
        return x

        


class conv_attn_score(nn.Module):
    def __init__(self, dim, embed_dim=32, num_heads=4, kernels=3): 
        super().__init__()
        # assert embed_dim % num_heads == 0, 'embed_dim should be divisible by num_heads'

        # self.Inception_blocks = []
        # for i in range(len(kernels)):
            # self.Inception_blocks.append(nn.Sequential(nn.Conv1d(dim, embed_dim, kernels[i]),cal_attn_mtx(embed_dim, num_heads)))
        
        # self.para = nn.Parameter(torch.randn(1, (embed_dim-kernels, 1))
        
        self.conv = nn.Conv1d(dim, embed_dim, kernels)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.attn_mtx = cal_attn_mtx(embed_dim, num_heads)
    def forward(self, x):
        # return self.para
    
        B,N,C = x.shape
        x = x.transpose(1,2)
        x = F.relu(self.conv(x)).transpose(1,2) # [B, N-kernels+1, embed_dim]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) # [B, N-kernels+1+1, embed_dim]
        x = self.attn_mtx(x) # [B, N+1, N+1]
        x = x[:,0,1:].squeeze(1).unsqueeze(-1) # [B, N-kernels+1, 1] similarity between cls token and other time point tokens

        
        # result=[]
        # for i in range(len(self.Inception_blocks)):
        #     result.append(self.Inception_blocks[i](x)).transpose(1,2)
        # return torch.cat(result, dim=1)

        return x


class conv_attn_mtx(nn.Module):
    def __init__(self, dim, embed_dim=32, num_heads=4, kernels=3):
        super().__init__()
        assert embed_dim % num_heads == 0, 'embed_dim should be divisible by num_heads'
        # self.Inception_blocks = []
        # for i in range(len(kernels)):
            # self.Inception_blocks.append(nn.Sequential(nn.Conv1d(dim, embed_dim, kernels[i]),cal_attn_mtx(embed_dim, num_heads)))
        self.conv = nn.Conv1d(dim, embed_dim, kernels)
        self.attn_mtx = cal_attn_mtx(embed_dim, num_heads)
    def forward(self, x):
        x = x.transpose(1,2)
        x = F.relu(self.conv(x)).transpose(1,2) # [B, N, embed_dim]
        x = self.attn_mtx(x) # [B, N, embed_dim]
        
        # result=[]
        # for i in range(len(self.Inception_blocks)):
        #     result.append(self.Inception_blocks[i](x)).transpose(1,2)
        # return torch.cat(result, dim=1)

        return x

class cal_attn_mtx(nn.Module):
    def __init__(self, dim, num_heads, attn_drop=0., qk_bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.qk = nn.Linear(dim, dim * 2, bias=qk_bias)
        self.scale = dim ** -0.5
        self.attn_drop = nn.Dropout(attn_drop)
    def forward(self, x):
        #x = torch.cat((x,y),dim=1)
        B, N, C = x.shape
        qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k= qk.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        relative = torch.mean(attn,dim=1) # reduce num_heads dim

        return relative

class Attention(nn.Module):
    def __init__(self, dim, num_heads=4, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x



