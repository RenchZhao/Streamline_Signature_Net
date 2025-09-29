import math
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F

import signatory

import subprocess

from utils.model_mask import conv_attn_score,conv_attn_mtx
from utils.model_fiber import BundleAvgStream


def get_free_gpu():
    # 检查是否有可用的 GPU
    if not torch.cuda.is_available():
        print("No GPU available, using CPU.")
        return 'cpu'
    
    # 使用 nvidia-smi 获取显卡的显存占用情况
    try:
        result = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], encoding='utf-8'
        )
        memory_used = [int(x) for x in result.strip().split('\n')]
        free_gpu = min(range(len(memory_used)), key=lambda i: memory_used[i])  # 找到显存占用最少的 GPU
        print(f"Using GPU: {free_gpu} (memory used: {memory_used[free_gpu]} MB)")
        return f'cuda:{free_gpu}'
    except Exception as e:
        print(f"Error detecting GPU: {e}, defaulting to CPU.")
        return 'cpu'

def load_and_freeze_layers(para_net, model_load, key_dict, freeze=False):
    """
    加载指定层参数并选择性冻结
    参数:
    para_net: 提供要load的参数的模型
    model_load: 需要加载参数的模型
    key_dict: net提供参数的层名字是key, model对应加载参数的层名字是value
    freeze (bool): 是否冻结model加载的参数
    
    返回:
    model_load: 加载完成且对应参数冻结的模型
    """
    net_state_dict = para_net.state_dict()

    load_dict = {}
    for net_k, model_k in key_dict.items():
        v = net_state_dict[net_k]
        load_dict[model_k] = v

    # 旧写法有误，除非key_dict的key==value
    # load_dict = {k: v for k, v in net_state_dict.items() if any(key in k for key in ['avg_stream'])}
    
    model_dict = model_load.state_dict()
    model_dict.update(load_dict)
    model_load.load_state_dict(model_dict)
    
    if freeze:
        for name, param in model_load.named_parameters():
            if any(key==name for key in key_dict.values()):
                print(f'freeze:{name}')
                param.requires_grad = False
    
    return model_load

def load_model_weights(model_weight_path, model_name='PointNetCls', num_classes=2, stream=15):
    model = get_model(model_name, num_classes)
    # weight_path = os.path.join(weight_path_base, str(num_fold), 'best_{}_model.pth'.format(args.best_metric))
    model.load_state_dict(torch.load(model_weight_path, weights_only=True, map_location='cpu'))
    return model

def get_model(model_name='PointNetCls', num_classes=2, stream=15, dict_feat_size=256, input_dim=3):
    if model_name=='PointNetCls':
        return PointNetCls(k=num_classes)  # Remove transformation nets
    elif model_name=='PointNetSig_Cls':
        return PointNetSig_Cls(k=num_classes)
    elif model_name=='SigNet':
        return SigNet(stream=stream, out_dimension=num_classes)
    elif model_name=='SigNet_tr_global_and_inception':
        return SigNet_tr_global_and_inception(stream=stream, out_dimension=num_classes)
    elif model_name=='SigNet_tr_single_inception':
        return SigNet_tr_single_inception(stream=stream, out_dimension=num_classes)
    elif model_name=='SigNet_tr_inception_with_mask':
        return SigNet_tr_inception_with_mask(stream=stream, out_dimension=num_classes)
    elif model_name=='SigNet_tr_inception_with_mask_and_dis_feat':
        return SigNet_tr_inception_with_mask_and_dis_feat(stream=stream, out_dimension=num_classes)
    elif model_name=='SigNet_tr_inception_Transformer':
        return SigNet_tr_inception_Transformer(stream=stream, out_dimension=num_classes)
    elif model_name=='SigNet_tr_single_with_mask':
        return SigNet_tr_single_with_mask(stream=stream, out_dimension=num_classes)
    elif model_name=='SigNet_inception_with_mask':
        return SigNet_inception_with_mask(stream=stream, out_dimension=num_classes)
    elif model_name=='tr_path_SigNet':
        return tr_path_SigNet(stream=stream, out_dimension=num_classes)
    elif model_name=='SigNet_with_norm':
        return SigNet_with_norm(stream=stream, out_dimension=num_classes)
    elif model_name=='SigNet_old_with_norm':
        return SigNet_old_with_norm(stream=stream, out_dimension=num_classes)
    elif model_name=='PointNetAndRawSig':
        return PointNetAndRawSig(stream=stream, out_dimension=num_classes)
    elif model_name=='tr_LSTMSigNet':
        return tr_LSTMSigNet(input_dim=input_dim, hidden_dim=128, stream=stream, window_stream=True, window_size=7, out_dimension=num_classes)
    elif model_name=='Single_SigNet_tr_with_dict':
        return Single_SigNet_tr_with_dict(in_channels=input_dim, stream=stream, out_dimension=num_classes, dict_feat_size=dict_feat_size)
    elif model_name=='Single_SigNet_tr_with_feat':
        return Single_SigNet_tr_with_feat(stream=stream, out_dimension=num_classes, dict_feat_size=dict_feat_size)
    elif model_name=='Sig_before_LSTM':
        return Sig_before_LSTM(in_channels=input_dim, hidden_dim=128, stream=stream, out_dimension=num_classes)
    elif model_name=='Sig_after_LSTM':
        return Sig_after_LSTM(in_channels=input_dim, hidden_dim=128, stream=stream, out_dimension=num_classes)
    elif model_name=='LSTMSigNet':
        return LSTMSigNet(input_dim=input_dim, hidden_dim=128, stream=stream, window_stream=True, window_size=7, out_dimension=num_classes)
    elif model_name=='LSTMSigNet_old':
        return LSTMSigNet_old(input_dim=input_dim, hidden_dim=128, out_dimension=num_classes)
    elif model_name=='PointNetFeatAndSig_cls':
        return PointNetFeatAndSig_cls(stream=stream, out_dimension=num_classes)
    elif model_name=='SigNet_tr_with_norm':
        return SigNet_tr_with_norm(stream=stream, out_dimension=num_classes)
    elif model_name=='LSTM':
        # return clsLSTM(input_dim=input_dim, hidden_dim=512, stream=stream,out_dimension=num_classes)
        return clsLSTM(input_dim=input_dim, hidden_dim=512, stream=stream, out_dimension=num_classes, num_layers=1)
    elif model_name=='tr_LSTM_with_mask':
        # return tr_LSTM_with_mask(input_dim=input_dim, hidden_dim=512, stream=stream,out_dimension=num_classes)
        return tr_LSTM_with_mask(input_dim=input_dim, hidden_dim=512, stream=stream,out_dimension=num_classes, num_layers=1)
    elif model_name=='LSTM_with_mask':
        return LSTM_with_mask(input_dim=input_dim, hidden_dim=512, stream=stream,out_dimension=num_classes)
    elif model_name=='tr_LSTM':
        return tr_LSTM(input_dim=input_dim, hidden_dim=512, stream=stream,out_dimension=num_classes)
    elif model_name=='Transformer':
        # return TransformerClassifier(input_dim=input_dim,seq_len=stream,out_dimension=num_classes)
        return TransformerClassifier(num_layers=1, input_dim=input_dim,seq_len=stream,out_dimension=num_classes)
    elif model_name=='Transformer_tr_with_mask':
        # return Transformer_tr_with_mask(input_dim=input_dim,seq_len=stream,out_dimension=num_classes)
        return Transformer_tr_with_mask(input_dim=input_dim,seq_len=stream,out_dimension=num_classes, num_layers=1)
    elif model_name=='Transformer_tr':
        return Transformer_tr(input_dim=input_dim,seq_len=stream,out_dimension=num_classes)
    elif model_name=='Transformer_with_mask':
        return Transformer_with_mask(input_dim=input_dim,seq_len=stream,out_dimension=num_classes)
    elif model_name=='TransformerMLP':
        return TransformerMLP(input_dim=input_dim,seq_len=stream,out_dimension=num_classes, pooling_type='linear')
    elif model_name=='TCN_tr_inception_with_mask':
        return TCN_tr_inception_with_mask(stream=stream, out_dimension=num_classes)
    elif model_name=='TCN_single':
        return TCN_single(stream=stream, out_dimension=num_classes)
    elif model_name=='BundleAvgStream':
        return BundleAvgStream(input_dim=input_dim,stream=stream,out_dimension=num_classes)
    elif model_name=='DeepWMA':
        return DeepWMA_conv(in_channel=input_dim, stream=stream, out_dimension=num_classes)
    else:
        raise RuntimeError('Invalid model name.')

def _compute_sig_output_channel(in_channels, sig_depth=2, layer_sizes=(), include_original=True, include_time=False):
        # compute new channel
        new_channel = 0
        if layer_sizes:
            if isinstance(layer_sizes, int):
                layer_sizes = (layer_sizes,)
            new_channel += layer_sizes[-1]
        if include_original:
            new_channel += in_channels
        if include_time:
            new_channel += 1
        if new_channel==0:
            raise RuntimeError('Invalid Augment layer:0 new channels. It will result in empty concatenation.')
        out_sig_channels = signatory.signature_channels(channels=new_channel,
                                                     depth=sig_depth)
        return out_sig_channels

def _create_windows(x, window_size=4, stride=1):
        """创建3维数据滑动窗口并添加时间维度"""
        # 滑动窗口划分 [B, N, C] -> [B, num_windows, window_size, C]
        # 师兄任务是window_size=4
        # N = x.shape[1]
        # num_windows = (N-window_size)//stride + 1
        x_unfold = x.unfold(1, window_size, stride).permute(0, 1, 3, 2)
        return x_unfold #, num_windows
        
class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, dp_rate=0.3):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, out_channels)
        self.dropout = nn.Dropout(p=dp_rate)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)

        return x

class TCN_single(nn.Module):
    def __init__(self, stream, out_dimension, in_channels=3, sig_depth=2, window_size=7, window_stride=1):
        super().__init__()
        self.conv_layer = TCNLayer(in_channels=in_channels, sig_depth=sig_depth, include_time=False, window_size=window_size, window_stride=window_stride)
        conv_channels = _compute_sig_output_channel(in_channels, sig_depth=sig_depth, layer_sizes=(), include_time=False)

        window_num=(stream-window_size)//window_stride + 1
        #self.sigBN = nn.BatchNorm1d(sig_channels)
        self.linear = MLP(conv_channels*window_num,out_dimension)

    def forward(self, inp):
        y = self.conv_layer(inp)
        if len(y.shape)>2:
            y = y.reshape(y.shape[0],-1)
        z = self.linear(y)

        return F.log_softmax(z, dim=1)


class TCNLayer(nn.Module):
    '''for signature comparation'''
    def __init__(self, in_channels, out_channel=None, window_size=2, window_stride=1, include_time=True, sig_depth=2):
        super().__init__()
        # compute new channel
        new_channel = in_channels

        # stride=1
        # self.augment = signatory.Augment(in_channels=in_channels,
        #                                   layer_sizes=(),
        #                                   kernel_size=1,
        #                                   include_original=True,
        #                                   include_time=include_time)
        
        self.new_channel = new_channel
        self.hidden_channel = signatory.signature_channels(channels=new_channel,
                                                     depth=sig_depth)
        self.conv = nn.Conv1d(new_channel, self.hidden_channel, kernel_size=window_size, stride=window_stride)
        # self.bn1 = nn.BatchNorm1d(self.out_sig_channels)
        # self.layerNorm1 = nn.LayerNorm(self.hidden_channel)
        if out_channel is not None:
            self.outLinear = nn.Linear(self.hidden_channel, out_channel)
        else:
            self.outLinear = None

    def forward(self, inp):
        # inp is a three dimensional tensor of shape (batch, stream, in_channels)

        # a = self.augment(inp)
        a = inp.transpose(1,2)
        if a.size(1) <= 1:
            raise RuntimeError("Given an input with too short a stream to take the"
                               " signature")
        # a in a three dimensional tensor of shape (batch, (stream+2*padding-kernel_size)/stride + 1, self.new_channel) 
        # b in (batch, stream, self.out_sig_channels) if keep_stream else (batch, self.out_sig_channels)
        b = self.conv(a)
        b=b.transpose(1,2)
        b = F.relu(b)
        if self.outLinear is not None:
            b = F.relu(self.outLinear(b))

        return b
    

class TCNLayer_noLN(nn.Module):
    '''for signature comparation'''
    def __init__(self, in_channels, out_channel=None, window_size=2, window_stride=1, include_time=True, sig_depth=2):
        super().__init__()
        # compute new channel
        new_channel = in_channels+1

        # stride=1
        self.augment = signatory.Augment(in_channels=in_channels,
                                          layer_sizes=(),
                                          kernel_size=1,
                                          include_original=True,
                                          include_time=include_time)
        
        self.new_channel = new_channel
        self.hidden_channel = signatory.signature_channels(channels=new_channel,
                                                     depth=sig_depth)
        self.conv = nn.Conv1d(new_channel, self.hidden_channel, kernel_size=window_size, stride=window_stride)
        # self.bn1 = nn.BatchNorm1d(self.out_sig_channels)
        # self.layerNorm1 = nn.LayerNorm(self.hidden_channel)
        if out_channel is not None:
            self.outLinear = nn.Linear(self.hidden_channel, out_channel)
        else:
            self.outLinear = None

    def forward(self, inp):
        # inp is a three dimensional tensor of shape (batch, stream, in_channels)

        a = self.augment(inp)
        a = a.transpose(1,2)
        if a.size(1) <= 1:
            raise RuntimeError("Given an input with too short a stream to take the"
                               " signature")
        # a in a three dimensional tensor of shape (batch, (stream+2*padding-kernel_size)/stride + 1, self.new_channel) 
        # b in (batch, stream, self.out_sig_channels) if keep_stream else (batch, self.out_sig_channels)
        b = self.conv(a)
        b=b.transpose(1,2)
        b = F.relu(b)
        if self.outLinear is not None:
            b = F.relu(self.outLinear(b))

        return b
    

class TCNLayer_full_as_sig(nn.Module):
    '''for signature comparation'''
    def __init__(self, in_channels, out_channel=None, window_size=2, window_stride=1, include_time=True, sig_depth=2):
        super().__init__()
        # compute new channel
        new_channel = in_channels+1

        # stride=1
        self.augment = signatory.Augment(in_channels=in_channels,
                                          layer_sizes=(),
                                          kernel_size=1,
                                          include_original=True,
                                          include_time=include_time)
        
        self.new_channel = new_channel
        self.hidden_channel = signatory.signature_channels(channels=new_channel,
                                                     depth=sig_depth)
        self.conv = nn.Conv1d(new_channel, self.hidden_channel, kernel_size=window_size, stride=window_stride)
        # self.bn1 = nn.BatchNorm1d(self.out_sig_channels)
        self.layerNorm1 = nn.LayerNorm(self.hidden_channel)
        if out_channel is not None:
            self.outLinear = nn.Linear(self.hidden_channel, out_channel)
        else:
            self.outLinear = None

    def forward(self, inp):
        # inp is a three dimensional tensor of shape (batch, stream, in_channels)

        a = self.augment(inp)
        a = a.transpose(1,2)
        if a.size(1) <= 1:
            raise RuntimeError("Given an input with too short a stream to take the"
                               " signature")
        # a in a three dimensional tensor of shape (batch, (stream+2*padding-kernel_size)/stride + 1, self.new_channel) 
        # b in (batch, stream, self.out_sig_channels) if keep_stream else (batch, self.out_sig_channels)
        b = self.conv(a)
        b=b.transpose(1,2)
        b = F.relu(self.layerNorm1(b))
        if self.outLinear is not None:
            b = F.relu(self.outLinear(b))

        return b



class Parallel_Encoders_and_MLP(nn.Module):
    def __init__(self, out_dimension, **kwargs):
        super().__init__()
        self.encoders = nn.ModuleList()
        hidden_dims = []
        for model_name in kwargs.keys():
            self.encoders.append(
                model_name(
                    **kwargs[model_name]
                )
            )
            hidden_dims.append(kwargs[model_name][out_dimension])
        self.linear = MLP(sum(hidden_dims), out_dimension)
    def  forward(self, x):
        return self.linear(torch.cat([encoder(x) for encoder in self.encoders], dim=-1))

class TransformerMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,          # 输入特征维度 C
        seq_len: int,            # 序列长度 T
        out_dimension: int,      # 输出类别数
        d_model: int = 128,      # 输入层特征通道数，Transformer的d_model
        num_heads: int = 4,      # Transformer头数
        num_layers: int = 2,     # Transformer层数
        dim_feedforward: int = 512,  # FFN隐藏层维度
        dropout: float = 0.1,
        pooling_type: str = "mean"
    ):
        super().__init__()
        
        self.pooling_type = pooling_type
        self.seq_len = seq_len

        self.in_layer = nn.Linear(input_dim,d_model)
        
        # Transformer编码器
        # softmax之后的是注意力分数，可以可视化,T维每个元素对其他元素注意力
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True  # 可以使用(batch, seq, dim)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        if self.pooling_type.lower()=='linear':
            self.pool_linear=nn.Linear(self.seq_len,1)
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, out_dimension)
        )

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        """
        输入: 
            x -> [B, T, C]
        输出: 
            logits -> [B, out_dimension]
        """
        
        # # 调整维度为PyTorch Transformer格式 (T, B, C)
        # x = x.permute(1, 0, 2)  # [T, B, C]

        x=self.in_layer(x)
        
        # Transformer编码
        transformer_out = self.transformer_encoder(x)  # [B, T, C]
        
        # 恢复维度并池化
        transformer_out = transformer_out #.permute(1, 0, 2)  # [B, T, C]
        # 修改池化部分
        if self.pooling_type == "mean":
            pooled = transformer_out.mean(dim=1)
        elif self.pooling_type == "max":
            pooled = transformer_out.max(dim=1).values
        elif self.pooling_type == "cls":
            # 使用第一个token作为分类特征
            pooled = transformer_out[:, 0, :]
        elif self.pooling_type=='linear':
            transformer_out = transformer_out.transpose(2,1)
            pooled = F.relu(self.pool_linear(transformer_out).squeeze(-1))

        # 分类输出
        logits = self.classifier(pooled)  # [B, out_dimension]
        return F.log_softmax(logits, dim=1)



class class_dict_similarity(nn.Module):
    def __init__(self, feat_size, input_dim, num_class, simi_type='cos'):
        super().__init__()
        self.feat_size=feat_size
        self.num_class=num_class
        self.simi_type = simi_type
        self.modulate = nn.Linear(input_dim, feat_size)

        # 各个类别的原型信号，可学习
        self.dict = nn.Parameter(torch.randn(num_class, feat_size))

    def forward(self, x):
        Batch = x.shape[0]
        if len(x.shape)>2:
            x=x.reshape(Batch, -1)
        x_mod = self.modulate(x) # [B, feat_size]
        x_mod = F.normalize(x_mod)

        if self.simi_type=='cos':
            # 扩展维度用于广播计算
            x_mod = x_mod.unsqueeze(1)  # [B, 1, feat_size]
            feat_expanded = self.dict.unsqueeze(0)  # [1, num_class, feat_size]
            out = F.cosine_similarity(x_mod, feat_expanded, dim=-1)
        elif self.simi_type=='dot':
            out = torch.matmul(x_mod, self.dict.transpose(1, 0))
        elif self.simi_type=='L1':
            # 扩展维度用于广播计算
            x_mod = x_mod.unsqueeze(1)  # [B, 1, feat_size]
            feat_expanded = self.dict.unsqueeze(0)  # [1, num_class, feat_size]
            out = torch.abs(x_mod-feat_expanded).sum(dim=-1)
        elif self.simi_type=='L2':
            # 扩展维度用于广播计算
            x_mod = x_mod.unsqueeze(1)  # [B, 1, feat_size]
            feat_expanded = self.dict.unsqueeze(0)  # [1, num_class, feat_size]
            out = (x_mod-feat_expanded)**2
            out = out.sum(dim=-1)
        return out
    
class clsLSTM(nn.Module):
    # cannot run
    def __init__(self, input_dim, hidden_dim, stream, out_dimension, num_layers=3):
        super().__init__()

        # 定义 LSTM 层
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, dropout=0, batch_first=True)
        # self.bn = nn.BatchNorm1d(hidden_dim*stream)
        self.bn = nn.BatchNorm1d(hidden_dim)

        # self.linear = MLP(hidden_dim*stream,out_dimension)
        self.linear = MLP(hidden_dim,out_dimension)

    def forward(self, x):

        # LSTM 前向传播
        # print(f"x.shape={x.shape}")
        lstm_out, (h_n, c_n) = self.lstm(x)
        # y = lstm_out.reshape(lstm_out.shape[0],-1)
        y = lstm_out[:,-1,:]
        y = F.relu(self.bn(y))
        
        z = self.linear(y)
        
        return F.log_softmax(z, dim=1)
    
class PointNetfeat(nn.Module):
    def __init__(self, out_dimension=1024):
        super(PointNetfeat, self).__init__()
        # 3-layer MLP (via 1D-CNN) : encoder points individually
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, out_dimension, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(out_dimension)
        self.out_dimension = out_dimension


    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, self.out_dimension)

        return x

class PointNetSigfeat(nn.Module):
    def __init__(self, out_dimension=256):
        super().__init__()
        # 3-layer MLP (via 1D-CNN) : encoder points individually
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        #self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.conv3 = torch.nn.Conv1d(128, out_dimension, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(out_dimension)
        self.out_dimension = out_dimension

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        return x
    
class PointNetCls(nn.Module):
    def __init__(self, k=2):
        super(PointNetCls, self).__init__()
        self.feat = PointNetfeat()
        # 3 fully connected layers for classification
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.transpose(2, 1) # points [B, 3, N] # newly added beacause train.py discarded it
        x = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)

        return F.log_softmax(x, dim=1)

class SigLayer(nn.Module):
    def __init__(self, in_channels, out_channel=None, sig_depth=2, layer_sizes=(), kernel_size=1, include_original=True, include_time=False, keep_stream=False, window_stream=False, window_size=4, window_stride=1):
        super().__init__()
        self.window_stream = window_stream
        self.window_size=window_size
        self.window_stride=window_stride
        if window_stream:
            keep_stream=False

        # compute new channel
        new_channel = 0
        if layer_sizes:
            if isinstance(layer_sizes, int):
                layer_sizes = (layer_sizes,)
            new_channel += layer_sizes[-1]
        if include_original:
            new_channel += in_channels
        if include_time:
            new_channel += 1
        if new_channel==0:
            raise RuntimeError('Invalid Augment layer:0 new channels. It will result in empty concatenation.')

        self.augment = signatory.Augment(in_channels=in_channels,
                                          layer_sizes=layer_sizes,
                                          kernel_size=kernel_size,
                                          include_original=include_original,
                                          include_time=include_time)
        self.signature = signatory.Signature(depth=sig_depth,
                                              stream=keep_stream)
        
        self.new_channel = new_channel
        self.out_sig_channels = signatory.signature_channels(channels=new_channel,
                                                     depth=sig_depth)
        # self.bn1 = nn.BatchNorm1d(self.out_sig_channels)
        self.layerNorm1 = nn.LayerNorm(self.out_sig_channels)
        if out_channel is not None:
            self.outLinear = nn.Linear(self.out_sig_channels, out_channel)
        else:
            self.outLinear = None

    def forward(self, inp, basepoint=True):
        # inp is a three dimensional tensor of shape (batch, stream, in_channels)

        if self.window_stream:
            B,N,C = inp.shape
            num_windows = (N-self.window_size)//self.window_stride + 1
            inp = _create_windows(inp, window_size=self.window_size, stride=self.window_stride).reshape(B*num_windows, self.window_size, C)
        a = self.augment(inp)
        if a.size(1) <= 1:
            raise RuntimeError("Given an input with too short a stream to take the"
                               " signature")
        # a in a three dimensional tensor of shape (batch, (stream+2*padding-kernel_size)/stride + 1, self.new_channel) 
        # b in (batch, stream, self.out_sig_channels) if keep_stream else (batch, self.out_sig_channels)
        b = self.signature(a, basepoint=basepoint)

        ## bn
        # if len(b.shape)>2:
        #     # keep_stream=True
        #     b = b.transpose(2,1) # (batch, out_sig_channels, stream)
        #     b = F.relu(self.bn1(b))
        #     b = b.transpose(2,1) # (batch, stream, out_sig_channels)
        # else:
        #     # keep_stream=False
        #     b = F.relu(self.bn1(b)) # (batch, out_sig_channels)
        b = F.relu(self.layerNorm1(b))
        if self.outLinear is not None:
            b = F.relu(self.outLinear(b))
        if self.window_stream:
            b = b.reshape(B, num_windows, -1)
        return b

class SigResLayer(nn.Module):
    def __init__(self, in_channels, out_channel=None, sig_depth=2, layer_sizes=(), kernel_size=1, include_time=False, keep_stream=False, window_stream=False, window_size=4, window_stride=1):
        super().__init__()
        self.window_stream = window_stream
        self.window_size=window_size
        self.window_stride=window_stride
        if window_stream:
            keep_stream=False

         # compute new channel
        new_channel = 0
        if layer_sizes:
            if isinstance(layer_sizes, int):
                layer_sizes = (layer_sizes,)
            new_channel += layer_sizes[-1]
        # if include_original:
        #     new_channel += in_channels
        if include_time:
            new_channel += 1
        if new_channel==0:
            raise RuntimeError('Invalid Augment layer:0 new channels. It will result in empty concatenation.')

        self.augment = signatory.Augment(in_channels=in_channels,
                                          layer_sizes=layer_sizes,
                                          kernel_size=kernel_size,
                                          include_original=False,
                                          include_time=include_time)
        self.signature = signatory.Signature(depth=sig_depth,
                                              stream=keep_stream)
        
        self.new_channel = new_channel
        self.out_sig_channels = signatory.signature_channels(channels=new_channel,
                                                     depth=sig_depth)
        # self.bn1 = nn.BatchNorm1d(self.out_sig_channels)
        self.layerNorm1 = nn.LayerNorm(self.out_sig_channels)
        # self.bn2 = nn.BatchNorm1d(self.out_sig_channels)
        #self.layerNorm2 = nn.LayerNorm(self.out_sig_channels)
        if out_channel is not None:
            self.outLinear = nn.Linear(self.out_sig_channels, out_channel)
        else:
            self.outLinear = None
            out_channel = self.out_sig_channels

        if keep_stream or window_stream:
            self.ResLinear = nn.Linear(in_channels, out_channel)
        else:
            self.ResLinear = None

    def forward(self, inp, basepoint=True, readout_method='mean'):
        # inp is a three dimensional tensor of shape (batch, stream, in_channels)
        if self.window_stream:
            B,N,C = inp.shape
            num_windows = (N-self.window_size)//self.window_stride + 1
            inp = _create_windows(inp, window_size=self.window_size, stride=self.window_stride).reshape(B*num_windows, self.window_size, C)

        a = self.augment(inp)
        if a.size(1) <= 1:
            raise RuntimeError("Given an input with too short a stream to take the"
                               " signature")
        # a in a three dimensional tensor of shape (batch, (stream+2*padding-kernel_size)/stride + 1, self.new_channel) 
        # b in (batch, stream, self.out_sig_channels) if keep_stream else (batch, self.out_sig_channels)
        b = self.signature(a, basepoint=basepoint)

        ## bn
        # if len(b.shape)>2:
        #     # keep_stream=True
        #     b = b.transpose(2,1) # (batch, out_sig_channels, stream)
        #     b = F.relu(self.bn1(b))
        #     b = b.transpose(2,1) # (batch, stream, out_sig_channels)
        # else:
        #     # keep_stream=False
        #     b = F.relu(self.bn1(b)) # (batch, out_sig_channels)
        b = F.relu(self.layerNorm1(b))
        if self.outLinear is not None:
            b = F.relu(self.outLinear(b))
        if self.ResLinear is not None:
            inp = inp.reshape(B, num_windows, self.window_size, C)
            if readout_method=='mean':
                inp = inp.mean(dim=2)
            elif readout_method=='max':
                inp = inp.max(dim=2)
            elif readout_method=='min':
                inp = inp.min(dim=2)

            b = F.relu(b + self.ResLinear(inp))
        return b
class SigFeat(nn.Module):
    def __init__(self, in_channels, out_dimension, hidden_layer_num=2, hidden_channel=512, sig_depth=2, in_Conv_Aug_channels=(), hidden_first_Conv_Aug_channels=(), hidden_middle_Conv_Aug_channels=()):
        super().__init__()
        
        if hidden_channel is None:
            # input_layer
            self.in_layer = SigLayer(in_channels=in_channels, sig_depth=sig_depth, layer_sizes=in_Conv_Aug_channels, include_original=True, include_time=True, window_stream=True)
            hidden_channel = _compute_sig_output_channel(in_channels, sig_depth=sig_depth, layer_sizes=in_Conv_Aug_channels, include_original=True, include_time=True)
        else:
            # input_layer
            self.in_layer = SigLayer(in_channels=in_channels, out_channel=hidden_channel, sig_depth=sig_depth, layer_sizes=in_Conv_Aug_channels, include_original=True, include_time=True, window_stream=True)
        
        # hidden layers
        self.hidden_layers, hidden_channel = self._make_hidden(hidden_channel, hidden_layer_num, sig_depth=sig_depth, first_Conv_Aug_channels=hidden_first_Conv_Aug_channels, hidden_Conv_Aug_channels=hidden_middle_Conv_Aug_channels)

        # out_layer
        # print(hidden_channel)
        self.out_sig_layer = SigLayer(in_channels=hidden_channel, out_channel=out_dimension, sig_depth=sig_depth, layer_sizes=(), include_original=True, include_time=False, keep_stream=False)


    def forward(self, inp):
        # inp is a three dimensional tensor of shape (batch, stream, in_channels)
        x = self.in_layer(inp)
        if self.hidden_layers is not None:
            x = self.hidden_layers(x)
        x = self.out_sig_layer(x)
        return x
    
    def _make_hidden(self, hidden_channel, hidden_layer_num, sig_depth=2, first_Conv_Aug_channels=(), hidden_Conv_Aug_channels=None):
        blk=[]
        new_channel = hidden_channel
        if hidden_layer_num==0:
            return None, new_channel
        for i in range(hidden_layer_num):
            # print(new_channel)
            if i==0:
                blk.append(SigLayer(in_channels=new_channel, out_channel=new_channel, sig_depth=sig_depth, layer_sizes=first_Conv_Aug_channels, include_original=True, include_time=False, window_stream=True))
            else:
                blk.append(SigLayer(in_channels=new_channel, out_channel=new_channel, sig_depth=sig_depth, layer_sizes=hidden_Conv_Aug_channels, include_original=True, include_time=False, window_stream=True))
        return nn.Sequential(*blk), new_channel
      
class SigFeat_old(nn.Module):
    def __init__(self, in_channels, out_dimension, hidden_layer_num=2, hidden_channel=64, sig_depth=2, in_Conv_Aug_channels=(8,8,4), hidden_first_Conv_Aug_channels=(128,64,32), hidden_middle_Conv_Aug_channels=(128,64,32)):
        super().__init__()
        
        if hidden_channel is None:
            # input_layer
            self.in_layer = SigLayer(in_channels=in_channels, sig_depth=sig_depth, layer_sizes=in_Conv_Aug_channels, include_original=True, include_time=True, window_stream=True)
            hidden_channel = _compute_sig_output_channel(in_channels, sig_depth=sig_depth, layer_sizes=in_Conv_Aug_channels, include_original=True, include_time=True)
        else:
            # input_layer
            self.in_layer = SigLayer(in_channels=in_channels, out_channel=hidden_channel, sig_depth=sig_depth, layer_sizes=in_Conv_Aug_channels, include_original=True, include_time=True, window_stream=True)
        
        # hidden layers
        self.hidden_layers, hidden_channel = self._make_hidden(hidden_channel, hidden_layer_num, sig_depth=sig_depth, first_Conv_Aug_channels=hidden_first_Conv_Aug_channels, hidden_Conv_Aug_channels=hidden_middle_Conv_Aug_channels)

        # out_layer
        # print(hidden_channel)
        out_layer_channel = min(max(int(pow(out_dimension,1/sig_depth)),16),128) # control feature dimension after out_sig_layer 
        self.out_sig_layer = SigLayer(in_channels=hidden_channel, sig_depth=sig_depth, layer_sizes=(out_layer_channel,), include_original=True, include_time=False, keep_stream=False)
        #self.out_sig_layer = SigResLayer(in_channels=hidden_channel, sig_depth=sig_depth, layer_sizes=(out_layer_channel,), include_time=False, keep_stream=False)
        self.out_sig_channels = _compute_sig_output_channel(hidden_channel, sig_depth=sig_depth, layer_sizes=(out_layer_channel,), include_original=True, include_time=False) # if SigRes include_original=False
        # print(self.out_sig_channels)
        self.linear = torch.nn.Linear(self.out_sig_channels, out_dimension)

    def forward(self, inp):
        # inp is a three dimensional tensor of shape (batch, stream, in_channels)
        x = self.in_layer(inp)
        if self.hidden_layers is not None:
            x = self.hidden_layers(x)
        x = self.out_sig_layer(x)
        x = F.relu(self.linear(x))
        return x
    
    def _make_hidden(self, hidden_channel, hidden_layer_num, sig_depth=2, first_Conv_Aug_channels=(128,64,32), hidden_Conv_Aug_channels=None):
        blk=[]
        new_channel = hidden_channel
        if hidden_layer_num==0:
            return None, new_channel
        for i in range(hidden_layer_num):
            # print(new_channel)
            if i==0:
                #first_Conv_Aug_channels=(128,64,32) # just like SE net
                blk.append(SigLayer(in_channels=new_channel, out_channel=new_channel, sig_depth=sig_depth, layer_sizes=first_Conv_Aug_channels, include_original=False, include_time=False, window_stream=True))
                # blk.append(SigLayer(in_channels=new_channel, sig_depth=sig_depth, layer_sizes=first_Conv_Aug_channels, include_original=False, include_time=False, window_stream=True))
                #blk.append(SigResLayer(in_channels=new_channel, sig_depth=sig_depth, layer_sizes=first_Conv_Aug_channels, include_time=False, window_stream=True))
                # new_channel = _compute_sig_output_channel(new_channel, sig_depth=sig_depth, layer_sizes=first_Conv_Aug_channels, include_original=False, include_time=False)
            else:
                # i=1 new_channel=16
                if hidden_Conv_Aug_channels is None:
                    new_layer_size = min(new_channel,128) # control feature dimension or it will explode
                    hidden_Conv_Aug_channels = (new_layer_size,new_layer_size)
                #blk.append(SigLayer(in_channels=new_channel, sig_depth=sig_depth, layer_sizes=hidden_Conv_Aug_channels, include_original=False, include_time=False, window_stream=True))
                #blk.append(SigResLayer(in_channels=new_channel, out_channel=new_channel, sig_depth=sig_depth, layer_sizes=hidden_Conv_Aug_channels, include_time=False, window_stream=True))
                blk.append(SigLayer(in_channels=new_channel, out_channel=new_channel, sig_depth=sig_depth, layer_sizes=hidden_Conv_Aug_channels, include_original=False, include_time=False, window_stream=True))
                # new_channel = _compute_sig_output_channel(new_channel, sig_depth=sig_depth, layer_sizes=hidden_Conv_Aug_channels, include_original=False, include_time=False)
        return nn.Sequential(*blk), new_channel

class signature_inception(nn.Module):
    def __init__(self, in_channels, sig_channel=512, sig_depth=2, include_whole=True):
        super().__init__()
        self.include_whole=include_whole
        self.sig_layer_win3 = SigLayer(in_channels=in_channels, out_channel=sig_channel, sig_depth=sig_depth, layer_sizes=(), include_original=True, include_time=True, window_stream=True, window_size=3 , window_stride=1)
        self.sig_layer_win5 = SigLayer(in_channels=in_channels, out_channel=sig_channel, sig_depth=sig_depth, layer_sizes=(), include_original=True, include_time=True, window_stream=True, window_size=5 , window_stride=1)
        self.sig_layer_win7 = SigLayer(in_channels=in_channels, out_channel=sig_channel, sig_depth=sig_depth, layer_sizes=(), include_original=True, include_time=True, window_stream=True, window_size=7 , window_stride=1)
        self.sig_layer_win9 = SigLayer(in_channels=in_channels, out_channel=sig_channel, sig_depth=sig_depth, layer_sizes=(), include_original=True, include_time=True, window_stream=True, window_size=9 , window_stride=1)
        if include_whole:
            self.sig_layer_whole = SigLayer(in_channels=in_channels, out_channel=sig_channel, sig_depth=sig_depth, layer_sizes=(), include_original=True, include_time=True, window_stream=False, keep_stream=False)
        

    def forward(self, inp):
        z_win3 = self.sig_layer_win3(inp)
        z_win5 = self.sig_layer_win5(inp)
        z_win7 = self.sig_layer_win7(inp)
        z_win9 = self.sig_layer_win9(inp)
        result = torch.cat((z_win3, z_win5, z_win7, z_win9),dim=1)
        if self.include_whole:
            z_whole = self.sig_layer_whole(inp)
            z_whole = z_whole.unsqueeze(1)
            result = torch.cat((result, z_whole),dim=1)

        return result

class sig_inception_with_mask(nn.Module):
    def __init__(self, in_channels, sig_channel=512, sig_depth=2, kernels=[3,5,7,9], include_whole=True, mask_type='dot'):
        super().__init__()
        self.mask_type = mask_type
        self.include_whole=include_whole
        self.mask_sig_blocks = nn.ModuleList()
        if mask_type=='dot':
            for i in range(len(kernels)):
                self.mask_sig_blocks.append(nn.ModuleList([
                    conv_attn_score(dim=in_channels, kernels=kernels[i]),
                    SigLayer(in_channels=in_channels, out_channel=sig_channel, sig_depth=sig_depth, layer_sizes=(), include_original=True, include_time=True, window_stream=True, window_size=kernels[i] , window_stride=1)
                    ]))
        elif mask_type=='matmul':
            for i in range(len(kernels)):
                self.mask_sig_blocks.append(nn.ModuleList([
                    conv_attn_mtx(dim=in_channels, kernels=kernels[i]),
                    SigLayer(in_channels=in_channels, out_channel=sig_channel, sig_depth=sig_depth, layer_sizes=(), include_original=True, include_time=True, window_stream=True, window_size=kernels[i] , window_stride=1)
                    ]))
        if include_whole: # whole series signature use all elements and do not need to be masked(mask: [B, 1, 1])
            self.sig_layer_whole = SigLayer(in_channels=in_channels, out_channel=sig_channel, sig_depth=sig_depth, layer_sizes=(), include_original=True, include_time=True, window_stream=False, keep_stream=False)

        

    def forward(self, inp):
        result = []
        for i in range(len(self.mask_sig_blocks)):
            mask = self.mask_sig_blocks[i][0](inp)
            sig_out = self.mask_sig_blocks[i][1](inp)
            if self.mask_type=='dot':
                sig_out = mask * sig_out
            elif self.mask_type=='matmul':
                sig_out =  torch.matmul(mask,sig_out)
            result.append(sig_out)
        result = torch.cat(result,dim=1)
        if self.include_whole:
            z_whole = self.sig_layer_whole(inp)
            z_whole = z_whole.unsqueeze(1)
            result = torch.cat((result, z_whole),dim=1)

        return result


class TCN_inception_with_mask_layer(nn.Module):
    def __init__(self, in_channels, stream=15, out_channels=512, sig_depth=2, kernels=[3,5,7,9], include_whole=True, mask_type='dot'):
        '''new: stream for include whole'''
        super().__init__()
        self.mask_type = mask_type
        self.include_whole=include_whole
        self.mask_conv_blocks = nn.ModuleList()
        if mask_type=='dot':
            for i in range(len(kernels)):
                self.mask_conv_blocks.append(nn.ModuleList([
                    conv_attn_score(dim=in_channels, kernels=kernels[i]),
                    TCNLayer(in_channels=in_channels, out_channel=out_channels, sig_depth=sig_depth, include_time=True, window_size=kernels[i], window_stride=1)
                    ]))
        elif mask_type=='matmul':
            for i in range(len(kernels)):
                self.mask_conv_blocks.append(nn.ModuleList([
                    conv_attn_mtx(dim=in_channels, kernels=kernels[i]),
                    TCNLayer(in_channels=in_channels, out_channel=out_channels, sig_depth=sig_depth, include_time=True, window_size=kernels[i], window_stride=1)
                    ]))
        if include_whole: # whole series convolution use all elements and do not need to be masked(mask: [B, 1, 1])
            self.conv_layer_whole = TCNLayer(in_channels=in_channels, out_channel=out_channels, sig_depth=sig_depth, include_time=True, window_size=stream, window_stride=1)

        

    def forward(self, inp):
        result = []
        for i in range(len(self.mask_conv_blocks)):
            mask = self.mask_conv_blocks[i][0](inp)
            conv_out = self.mask_conv_blocks[i][1](inp)
            if self.mask_type=='dot':
                conv_out = mask * conv_out
            elif self.mask_type=='matmul':
                conv_out =  torch.matmul(mask,conv_out)
            result.append(conv_out)
        result = torch.cat(result,dim=1)
        if self.include_whole:
            z_whole = self.conv_layer_whole(inp)
            # z_whole = z_whole.unsqueeze(1)
            result = torch.cat((result, z_whole),dim=1)

        return result  

class inception_SigNet_and_MLP(nn.Module):
    def __init__(self, stream, out_dimension, in_channels=3):
        super().__init__()
        ss

    def forward(self, inp):
        # inp is a three dimensional tensor of shape (batch, stream, in_channels)
        # inp = inp.unsqueeze(-1)
        # print(f"inp.shape={inp.shape}")

        casc
        return F.log_softmax(z, dim=1)
    
class SigNet(nn.Module):
    def __init__(self, stream, out_dimension, in_channels=3):
        super().__init__()
        self.sigFeat = SigFeat(in_channels=in_channels, hidden_channel=512, hidden_layer_num=2, out_dimension=512)
        self.fcOutput = MLP(512, out_dimension)

    def forward(self, inp):
        # inp is a three dimensional tensor of shape (batch, stream, in_channels)
        # inp = inp.unsqueeze(-1)
        # print(f"inp.shape={inp.shape}")

        # sig Feat
        z = self.sigFeat(inp)

        # output
        z = self.fcOutput(z)
        return F.log_softmax(z, dim=1)

class tr_path_SigNet(nn.Module):
    def __init__(self, stream, out_dimension, in_channels=3):
        super().__init__()
        self.sigFeat1 = SigFeat(in_channels=in_channels, hidden_channel=512, hidden_layer_num=1, out_dimension=512)
        self.sigFeat2 = SigFeat(in_channels=in_channels, hidden_channel=512, hidden_layer_num=1, out_dimension=512)
        self.fcOutput = MLP(1024, out_dimension)

    def forward(self, inp):
        # inp is a three dimensional tensor of shape (batch, stream, in_channels)
        # inp = inp.unsqueeze(-1)
        # print(f"inp.shape={inp.shape}")

        # sig Feat
        z1 = self.sigFeat1(inp)
        filp_inp = inp.flip(dims=[1])
        z2 = self.sigFeat2(filp_inp)

        z = torch.cat((z1,z2),dim=1)
        # output
        z = self.fcOutput(z)
        return F.log_softmax(z, dim=1)

class PointNetSig_Cls(nn.Module):
    # cannot run
    def __init__(self, k=2, sig_depth=2, kernel_size=1, include_orignial=True, include_time=True):
        super().__init__()
        self.feat = PointNetSigfeat() # [B, 1024, N=15]

        in_channels = 256
        self.augment = signatory.Augment(in_channels=in_channels,
                                        layer_sizes=(),
                                        kernel_size=kernel_size,
                                        include_original=include_orignial,
                                        include_time=include_time)

        self.signature = signatory.Signature(depth=sig_depth)

        if include_time:
            # +1 because signatory.Augment is used to add time as well
            sig_channels = signatory.signature_channels(channels=in_channels + 1,
                                                        depth=sig_depth)
        else:
            sig_channels = signatory.signature_channels(channels=in_channels,
                                                        depth=sig_depth)
        # 3 fully connected layers for classification
        #self.fc1 = nn.Linear(1024, 512)
        # signature will reduce a dim
        self.fc1 = nn.Linear(sig_channels, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x, basepoint=True):
        x = x.transpose(2, 1) # points [B, 3, N]
        x = self.feat(x)
        x = x.transpose(2, 1) # points [B, N, 1024]
        x = self.augment(x)
        if x.size(1) <= 1:
            raise RuntimeError("Given an input with too short a stream to take the"
                               " signature")
        x = self.signature(x, basepoint=basepoint)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)

        return F.log_softmax(x, dim=1)



class PointNetFeatAndSig_cls_old(nn.Module):
    # cannot run
    def __init__(self, stream, out_dimension, sig_depth=2, kernel_size=1, include_orignial=True,include_time=True,fc=True):
        super().__init__()
        self.pointFeat = PointNetSigfeat()
        in_channels = 1024
        self.augment = signatory.Augment(in_channels=in_channels,
                                         layer_sizes=(),
                                         kernel_size=kernel_size,
                                         include_original=include_orignial,
                                         include_time=include_time)

        self.signature = signatory.Signature(depth=sig_depth)

        if include_time:
            # +1 because signatory.Augment is used to add time as well
            sig_channels = signatory.signature_channels(channels=in_channels + 1,
                                                        depth=sig_depth)
        else:
            sig_channels = signatory.signature_channels(channels=in_channels,
                                                        depth=sig_depth)
        self.fc = fc
        #self.dropout = nn.Dropout(p=dropout)
        self.sigFeatLinear = MLP(sig_channels,128)
        self.pointFeatLinear = MLP(1024*stream,128)
        self.fcOutput = nn.Linear(256, out_dimension)

    def forward(self, inp, basepoint=True):
        # inp is a three dimensional tensor of shape (batch, stream, in_channels)
        # inp = inp.unsqueeze(-1)
        # print(f"inp.shape={inp.shape}")

        # PointNet Feat
        inp = inp.transpose(2, 1)
        inp = self.pointFeat(inp)
        inp = inp.transpose(2, 1)

        # sig Feat
        x = self.augment(inp)

        if x.size(1) <= 1:
            raise RuntimeError("Given an input with too short a stream to take the"
                               " signature")
        # x in a three dimensional tensor of shape (batch, stream, in_channels + 1),
        # as time has been added as a value
        y = self.signature(x, basepoint=basepoint)

        # y is a two dimensional tensor of shape (batch, terms), corresponding to
        # the terms of the signature
        if self.fc:
            z = F.relu(self.sigFeatLinear(y))
        # z is a two dimensional tensor of shape (batch, out_dimension)
        else:
            z = y
        
        p = inp.reshape(inp.shape[0], -1)
        p = F.relu(self.pointFeatLinear(p))
        # concat
        z = torch.cat((z, p), dim=1)

        # output
        z = self.fcOutput(z)
        return F.log_softmax(z, dim=1)


class PointNetSig_Cls(nn.Module):
    # cannot run
    def __init__(self, k=2, sig_depth=2, kernel_size=1, include_orignial=True, include_time=True):
        super().__init__()
        self.feat = PointNetSigfeat() # [B, 1024, N=15]

        in_channels = 256
        self.augment = signatory.Augment(in_channels=in_channels,
                                        layer_sizes=(),
                                        kernel_size=kernel_size,
                                        include_original=include_orignial,
                                        include_time=include_time)

        self.signature = signatory.Signature(depth=sig_depth)

        if include_time:
            # +1 because signatory.Augment is used to add time as well
            sig_channels = signatory.signature_channels(channels=in_channels + 1,
                                                        depth=sig_depth)
        else:
            sig_channels = signatory.signature_channels(channels=in_channels,
                                                        depth=sig_depth)
        
        self.layernorm = nn.LayerNorm(sig_channels)
        # 3 fully connected layers for classification
        #self.fc1 = nn.Linear(1024, 512)
        # signature will reduce a dim
        self.fc1 = nn.Linear(sig_channels, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x, basepoint=True):
        x = x.transpose(2, 1) # points [B, 3, N]
        x = self.feat(x)
        x = x.transpose(2, 1) # points [B, N, 1024]
        x = self.augment(x)
        if x.size(1) <= 1:
            raise RuntimeError("Given an input with too short a stream to take the"
                               " signature")
        x = self.signature(x, basepoint=basepoint)
        x = F.relu(self.layernorm(x))
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)

        return F.log_softmax(x, dim=1)

class Sig_before_LSTM(nn.Module):
    # cannot run
    def __init__(self, hidden_dim, out_dimension, in_channels=3, stream=15, num_layers=2, sig_depth=2, include_whole_sig=False, complicate_out=True):
        # num_layers： layers of LSTM
        super().__init__()

        sig_out = 32
        self.sig_layer_Global = SigLayer(in_channels=in_channels, out_channel=sig_out, sig_depth=sig_depth, layer_sizes=(), include_original=True, include_time=True, keep_stream=True)
        #self.sig_layer_Local =  signature_inception(in_channels=in_channels, sig_channel=sig_out, sig_depth=sig_depth, include_whole=include_whole_sig)
        # 定义 LSTM 层
        self.lstm = nn.LSTM(sig_out, hidden_dim, num_layers=num_layers, dropout=0, batch_first=True)
        self.bn = nn.BatchNorm1d(hidden_dim)  # 对每个时间步的隐藏状态进行BN

        #self.sigBN = nn.BatchNorm1d(sig_channels)
        # window num calculate
        self.win3_num = (stream-3)//1 + 1
        self.win5_num = (stream-5)//1 + 1
        self.win7_num = (stream-7)//1 + 1
        self.win9_num = (stream-9)//1 + 1
        window_num = sum([self.win3_num, self.win5_num, self.win7_num, self.win9_num])
        if include_whole_sig:
            window_num+=1

        if complicate_out:
            self.linear = MLP(stream*hidden_dim,out_dimension) #+(window_num)*hidden_dim
        else:
            self.linear = nn.Linear(stream*hidden_dim,out_dimension) #+(window_num)*hidden_dim

    def forward(self, x):
        y_global = self.sig_layer_Global(x)
        # y_local = self.sig_layer_Local(x)
        # y = torch.cat((y_global, y_local),dim=1)
        #y=y_local
        y=y_global


        # LSTM 前向传播
        # print(f"x.shape={x.shape}")
        lstm_out, (h_n, c_n) = self.lstm(y)
        # print(f"lstm_out.shape={lstm_out.shape}") # [64, 41, 128]

        # # lstm_out的形状是 (batch_size, seq_len, hidden_size)
        # # 将lstm_out的形状调整为 (batch_size * seq_len, hidden_size)，以便应用BN
        # lstm_out = lstm_out.reshape(-1, lstm_out.size(2))
        
        # # 进行批量归一化
        # lstm_out = self.bn(lstm_out)
        
        # # 将lstm_out的形状还原回 (batch_size, seq_len, hidden_size)
        # lstm_out = lstm_out.view(x.size(0), x.size(1), -1)

        if len(lstm_out.shape)>2:
            lstm_out = lstm_out.reshape(y.shape[0],-1)
        
        z = self.linear(lstm_out)
        
        return F.log_softmax(z, dim=1)

class Sig_after_LSTM(nn.Module):
    # cannot run
    def __init__(self, hidden_dim, out_dimension, in_channels=3, stream=15, num_layers=1, sig_depth=2, sig_out=64, include_whole_sig=False, complicate_out=True):
        # num_layers： layers of LSTM
        super().__init__()


        # 定义 LSTM 层
        self.lstm = nn.LSTM(in_channels, hidden_dim, num_layers=num_layers, dropout=0, batch_first=True)

        sig_out = 32
        self.sig_layer_Global = SigLayer(in_channels=hidden_dim, out_channel=sig_out, sig_depth=sig_depth, layer_sizes=(), include_original=True, include_time=True, keep_stream=True)
        self.sig_layer_Local =  signature_inception(in_channels=hidden_dim, sig_channel=sig_out, sig_depth=sig_depth, include_whole=include_whole_sig)

        # window num calculate
        self.win3_num = (stream-3)//1 + 1
        self.win5_num = (stream-5)//1 + 1
        self.win7_num = (stream-7)//1 + 1
        self.win9_num = (stream-9)//1 + 1
        window_num = sum([self.win3_num, self.win5_num, self.win7_num, self.win9_num])
        if include_whole_sig:
            window_num+=1
        mlp_in = sig_out*(window_num) #stream+

        if complicate_out:
            self.linear = MLP(mlp_in,out_dimension)
        else:
            self.linear = nn.Linear(mlp_in,out_dimension) 

    def forward(self, x):
        

        # LSTM 前向传播
        # print(f"x.shape={x.shape}")
        lstm_out, (h_n, c_n) = self.lstm(x)
        # print(f"lstm_out.shape={lstm_out.shape}") # [64, 41, 128]

        # # lstm_out的形状是 (batch_size, seq_len, hidden_size)
        # # 将lstm_out的形状调整为 (batch_size * seq_len, hidden_size)，以便应用BN
        # lstm_out = lstm_out.reshape(-1, lstm_out.size(2))
        
        # # 进行批量归一化
        # lstm_out = self.bn(lstm_out)
        
        # # 将lstm_out的形状还原回 (batch_size, seq_len, hidden_size)
        # lstm_out = lstm_out.view(x.size(0), x.size(1), -1)
        
        #y_global = self.sig_layer_Global(lstm_out)
        y_local = self.sig_layer_Local(lstm_out)
        #y = torch.cat((y_global, y_local),dim=1)
        y=y_local
        #y=y_global

        if len(y.shape)>2:
            y = y.reshape(y.shape[0],-1)

        z = self.linear(y)
        
        return F.log_softmax(z, dim=1)
    
class tr_LSTMSigNet(nn.Module):
    # cannot run
    def __init__(self, input_dim, hidden_dim, out_dimension, num_layers=2, stream=15, sig_depth=2, window_stream=False, window_size=4, window_stride=1):
        super().__init__()

        self.forward_feature = LSTMSigNet(input_dim, hidden_dim, 256, stream, num_layers, sig_depth, window_stream=window_stream, window_size=window_size, window_stride=window_stride, complicate_out=False)
        self.reverse_feature = LSTMSigNet(input_dim, hidden_dim, 256, stream, num_layers, sig_depth, window_stream=window_stream, window_size=window_size, window_stride=window_stride, complicate_out=False)
        self.fcOutput = MLP(512, out_dimension)


    def forward(self, inp):
        # sig Feat
        z1 = self.forward_feature(inp)
        filp_inp = inp.flip(dims=[1])
        z2 = self.reverse_feature(filp_inp)

        z = torch.cat((z1,z2),dim=1)
        # output
        z = self.fcOutput(z)
        
        return F.log_softmax(z, dim=1)

class LSTMSigNet(nn.Module):
    # cannot run
    def __init__(self, input_dim, hidden_dim, out_dimension, stream=15, num_layers=2, sig_depth=2, window_stream=False, window_size=4, window_stride=1, complicate_out=True):
        super().__init__()

        # 定义 LSTM 层
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, dropout=0, batch_first=True)
        self.bn = nn.BatchNorm1d(hidden_dim)  # 对每个时间步的隐藏状态进行BN

        sig_out = 128
        self.sig_layer = SigLayer(in_channels=hidden_dim, out_channel=sig_out, sig_depth=sig_depth, layer_sizes=(128,64,32), include_original=True, include_time=False, window_stream=window_stream, window_size=window_size, window_stride=window_stride)
        # self.sig_channels = _compute_sig_output_channel(hidden_dim, sig_depth=sig_depth, layer_sizes=(128,64,32), include_original=True, include_time=False) # if SigRes include_original=False
        window_num = 1
        if window_stream:
            window_num=(stream-window_size)//window_stride + 1
        if complicate_out:
            self.linear = MLP(window_num*sig_out,out_dimension)
        else:
            self.linear = nn.Linear(window_num*sig_out,out_dimension)

    def forward(self, x):

        # LSTM 前向传播
        # print(f"x.shape={x.shape}")
        lstm_out, (h_n, c_n) = self.lstm(x)
        # print(f"lstm_out.shape={lstm_out.shape}") # [64, 41, 128]

        # # lstm_out的形状是 (batch_size, seq_len, hidden_size)
        # # 将lstm_out的形状调整为 (batch_size * seq_len, hidden_size)，以便应用BN
        # lstm_out = lstm_out.reshape(-1, lstm_out.size(2))
        
        # # 进行批量归一化
        # lstm_out = self.bn(lstm_out)
        
        # # 将lstm_out的形状还原回 (batch_size, seq_len, hidden_size)
        # lstm_out = lstm_out.view(x.size(0), x.size(1), -1)

        y = self.sig_layer(lstm_out)
        if len(y.shape)>2:
            y = y.reshape(y.shape[0],-1)
        
        z = self.linear(y)
        
        return F.log_softmax(z, dim=1)
    
class LSTMSigNet_old(nn.Module):
    # wrong usage but good effect
    def __init__(self, input_dim, hidden_dim, out_dimension, in_channels=3, num_layers=2, sig_depth=2, dropout=0, kernel_size=1, include_orignial=True,include_time=True,fc=True):
        super().__init__()
        # 当只有 1 层时，禁用 Dropout
        effective_dropout = 0.0 if num_layers == 1 else dropout

        # 定义 LSTM 层
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, dropout=effective_dropout, batch_first=True)
        self.bn = nn.BatchNorm1d(hidden_dim)  # 对每个时间步的隐藏状态进行BN

        self.augment = signatory.Augment(in_channels=in_channels,
                                         layer_sizes=(),
                                         kernel_size=kernel_size,
                                         include_original=include_orignial,
                                         include_time=include_time)

        self.signature = signatory.Signature(depth=sig_depth)

        if include_time:
            # +1 because signatory.Augment is used to add time as well
            sig_channels = signatory.signature_channels(channels=hidden_dim + 1,
                                                        depth=sig_depth)
        else:
            sig_channels = signatory.signature_channels(channels=hidden_dim,
                                                        depth=sig_depth)
        self.fc = fc
        self.layernorm = nn.LayerNorm(sig_channels)
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(sig_channels,out_dimension)

    def forward(self, x, basepoint=True):
        # x=x.transpose(2,1) # [B,C,T]
        # LSTM 前向传播
        # print(f"x.shape={x.shape}")
        lstm_out, (h_n, c_n) = self.lstm(x)
        # print(f"lstm_out.shape={lstm_out.shape}") # [64, 41, 128]

        # # lstm_out的形状是 (batch_size, seq_len, hidden_size)
        # # 将lstm_out的形状调整为 (batch_size * seq_len, hidden_size)，以便应用BN
        # lstm_out = lstm_out.reshape(-1, lstm_out.size(2))

        # # 进行批量归一化
        # lstm_out = self.bn(lstm_out)
        
        # # 将lstm_out的形状还原回 (batch_size, seq_len, hidden_size)
        # lstm_out = lstm_out.view(x.size(0), x.size(1), -1)

        x = self.augment(lstm_out)
        if x.size(1) <= 1:
            raise RuntimeError("Given an input with too short a stream to take the"
                               " signature")
        y = self.signature(x, basepoint=basepoint)
        y = F.relu(self.layernorm(y))


        if self.fc:
            z = self.linear(self.dropout(y))
        else:
            z = y
        
        return F.log_softmax(z, dim=1)


class PointNetFeatAndSig_cls(nn.Module):
    # cannot run
    def __init__(self, stream, out_dimension, sig_depth=2):
        super().__init__()
        self.pointFeat = PointNetSigfeat(out_dimension=64)
        in_channels = 64
        self.sigFeat = SigFeat(in_channels=in_channels, sig_depth=sig_depth, out_dimension=1024)
        self.fcOutput = MLP(1024,out_dimension)

    def forward(self, inp):
        # inp is a three dimensional tensor of shape (batch, stream, in_channels)
        # inp = inp.unsqueeze(-1)
        # print(f"inp.shape={inp.shape}")

        # PointNet Feat
        inp = inp.transpose(2, 1)
        inp = self.pointFeat(inp)
        inp = inp.transpose(2, 1)

        # sig Feat
        z = self.sigFeat(inp)

        # output
        z = self.fcOutput(z)
        return F.log_softmax(z, dim=1)

class PointNetAndRawSig(nn.Module):
    # cannot run
    def __init__(self, stream, out_dimension, sig_depth=2, kernel_size=1, include_orignial=True,include_time=True,fc=True):
        super().__init__()
        self.pointFeat = PointNetfeat()
        in_channels = 3
        self.augment = signatory.Augment(in_channels=in_channels,
                                         layer_sizes=(),
                                         kernel_size=kernel_size,
                                         include_original=include_orignial,
                                         include_time=include_time)

        self.signature = signatory.Signature(depth=sig_depth)

        if include_time:
            # +1 because signatory.Augment is used to add time as well
            sig_channels = signatory.signature_channels(channels=in_channels + 1,
                                                        depth=sig_depth)
        else:
            sig_channels = signatory.signature_channels(channels=in_channels,
                                                        depth=sig_depth)
        self.fc = fc
        #self.dropout = nn.Dropout(p=dropout)
        self.layernorm = nn.LayerNorm(sig_channels)
        self.sigFeatLinear = MLP(sig_channels,128)
        self.pointFeatLinear = MLP(1024,128)
        self.fcOutput = nn.Linear(256, out_dimension)

    def forward(self, inp, basepoint=True):
        # inp is a three dimensional tensor of shape (batch, stream, in_channels)
        # inp = inp.unsqueeze(-1)
        # print(f"inp.shape={inp.shape}")

        # PointNet Feat
        inp = inp.transpose(2, 1)
        p = self.pointFeat(inp)
        inp = inp.transpose(2, 1)

        # sig Feat
        x = self.augment(inp)

        if x.size(1) <= 1:
            raise RuntimeError("Given an input with too short a stream to take the"
                               " signature")
        # x in a three dimensional tensor of shape (batch, stream, in_channels + 1),
        # as time has been added as a value
        y = self.signature(x, basepoint=basepoint)
        y = F.relu(self.layernorm(y))

        # y is a two dimensional tensor of shape (batch, terms), corresponding to
        # the terms of the signature
        if self.fc:
            z = F.relu(self.sigFeatLinear(y))
        # z is a two dimensional tensor of shape (batch, out_dimension)
        else:
            z = y
        
        p = p.reshape(p.shape[0], -1)
        p = F.relu(self.pointFeatLinear(p))
        # concat
        z = torch.cat((z, p), dim=1)

        # output
        z = self.fcOutput(z)
        return F.log_softmax(z, dim=1)
     
class PointNetFeatAndSig_cls_old(nn.Module):
    # cannot run
    def __init__(self, stream, out_dimension, sig_depth=2, kernel_size=1, include_orignial=True,include_time=True,fc=True):
        super().__init__()
        self.pointFeat = PointNetSigfeat()
        in_channels = 1024
        self.augment = signatory.Augment(in_channels=in_channels,
                                         layer_sizes=(),
                                         kernel_size=kernel_size,
                                         include_original=include_orignial,
                                         include_time=include_time)

        self.signature = signatory.Signature(depth=sig_depth)

        if include_time:
            # +1 because signatory.Augment is used to add time as well
            sig_channels = signatory.signature_channels(channels=in_channels + 1,
                                                        depth=sig_depth)
        else:
            sig_channels = signatory.signature_channels(channels=in_channels,
                                                        depth=sig_depth)
        self.fc = fc
        #self.dropout = nn.Dropout(p=dropout)
        self.sigFeatLinear = MLP(sig_channels,128)
        self.pointFeatLinear = MLP(1024*stream,128)
        self.fcOutput = nn.Linear(256, out_dimension)

    def forward(self, inp, basepoint=True):
        # inp is a three dimensional tensor of shape (batch, stream, in_channels)
        # inp = inp.unsqueeze(-1)
        # print(f"inp.shape={inp.shape}")

        # PointNet Feat
        inp = inp.transpose(2, 1)
        inp = self.pointFeat(inp)
        inp = inp.transpose(2, 1)

        # sig Feat
        x = self.augment(inp)

        if x.size(1) <= 1:
            raise RuntimeError("Given an input with too short a stream to take the"
                               " signature")
        # x in a three dimensional tensor of shape (batch, stream, in_channels + 1),
        # as time has been added as a value
        y = self.signature(x, basepoint=basepoint)

        # y is a two dimensional tensor of shape (batch, terms), corresponding to
        # the terms of the signature
        if self.fc:
            z = F.relu(self.sigFeatLinear(y))
        # z is a two dimensional tensor of shape (batch, out_dimension)
        else:
            z = y
        
        p = inp.reshape(inp.shape[0], -1)
        p = F.relu(self.pointFeatLinear(p))
        # concat
        z = torch.cat((z, p), dim=1)

        # output
        z = self.fcOutput(z)
        return F.log_softmax(z, dim=1)

    
class SigNet_awkward(nn.Module):
    def __init__(self, stream, out_dimension, in_channels=3, sig_depth=2, kernel_size=1, include_orignial=True,include_time=True,fc=True):
        super().__init__()
        self.augment = signatory.Augment(in_channels=in_channels,
                                         layer_sizes=(),
                                         kernel_size=kernel_size,
                                         include_original=include_orignial,
                                         include_time=include_time)

        self.signature = signatory.Signature(depth=sig_depth)

        if include_time:
            # +1 because signatory.Augment is used to add time as well
            sig_channels = signatory.signature_channels(channels=in_channels + 1,
                                                        depth=sig_depth)
        else:
            sig_channels = signatory.signature_channels(channels=in_channels,
                                                        depth=sig_depth)
        self.fc = fc
        #self.dropout = nn.Dropout(p=dropout)
        self.sigNorm = nn.LayerNorm(sig_channels)
        self.sigFeatLinear = MLP(sig_channels,128)
        self.rawFeatLinear = MLP(stream*in_channels,128) # 3D dimension so *3
        self.fcOutput = nn.Linear(256, out_dimension)

    def forward(self, inp, basepoint=True):
        # inp is a three dimensional tensor of shape (batch, stream, in_channels)
        # inp = inp.unsqueeze(-1)
        # print(f"inp.shape={inp.shape}")

        # sig Feat
        x = self.augment(inp)

        if x.size(1) <= 1:
            raise RuntimeError("Given an input with too short a stream to take the"
                               " signature")
        # x in a three dimensional tensor of shape (batch, stream, in_channels + 1),
        # as time has been added as a value
        y = self.signature(x, basepoint=basepoint)

        # y is a two dimensional tensor of shape (batch, terms), corresponding to
        # the terms of the signature
        if self.fc:
            z = F.relu(self.sigNorm(self.sigFeatLinear(y)))
        # z is a two dimensional tensor of shape (batch, out_dimension)
        else:
            z = y
        
        # raw Feat
        p = inp.reshape(inp.shape[0], -1)
        p = F.relu(self.rawFeatLinear(p))

        # concat
        z = torch.cat((z, p), dim=1)

        # output
        z = self.fcOutput(z)
        return F.log_softmax(z, dim=1)


class Single_SigNet_tr_with_feat(nn.Module):
    def __init__(self, stream, out_dimension, in_channels=3, sig_depth=2, in_Conv_Aug_channels=(8,8,4), window_stream=True, window_size=7, window_stride=1, dict_feat_size=256):
        super().__init__()
        # input_layer
        self.sig_layer1 = SigLayer(in_channels=in_channels, sig_depth=sig_depth, layer_sizes=in_Conv_Aug_channels, include_original=True, include_time=True, window_stream=window_stream, window_size=window_size, window_stride=window_stride)
        self.sig_layer2 = SigLayer(in_channels=in_channels, sig_depth=sig_depth, layer_sizes=in_Conv_Aug_channels, include_original=True, include_time=True, window_stream=window_stream, window_size=window_size, window_stride=window_stride)
        sig_channels = _compute_sig_output_channel(in_channels, sig_depth=sig_depth, layer_sizes=in_Conv_Aug_channels, include_original=True, include_time=True)

        #self.sigBN = nn.BatchNorm1d(sig_channels)
        window_num = 1
        if window_stream:
            window_num=(stream-window_size)//window_stride + 1
        self.linear = MLP(2*sig_channels*window_num,out_dimension)
        self.modulate = nn.Linear(2*sig_channels*window_num, dict_feat_size)

    def forward(self, inp):
        # inp is a three dimensional tensor of shape (batch, stream, in_channels)
        # inp = inp.unsqueeze(-1)
        # print(f"inp.shape={inp.shape}")
        # sig Feat
        z1 = self.sig_layer1(inp)
        filp_inp = inp.flip(dims=[1])
        z2 = self.sig_layer2(filp_inp)
        if len(z1.shape)>2:
            z1 = z1.reshape(z1.shape[0],-1)
        if len(z2.shape)>2:
            z2 = z2.reshape(z2.shape[0],-1)

        z = torch.cat((z1,z2),dim=1)
        # output
        cls_out = self.linear(z)

        return cls_out, F.normalize(self.modulate(z), dim=1)
    
class Single_SigNet_tr_with_dict(nn.Module):
    def __init__(self, stream, out_dimension, in_channels=3, sig_depth=2, in_Conv_Aug_channels=(8,8,4), window_stream=True, window_size=7, window_stride=1, dict_feat_size=256):
        super().__init__()
        # input_layer
        self.sig_layer1 = SigLayer(in_channels=in_channels, sig_depth=sig_depth, layer_sizes=in_Conv_Aug_channels, include_original=True, include_time=True, window_stream=window_stream, window_size=window_size, window_stride=window_stride)
        self.sig_layer2 = SigLayer(in_channels=in_channels, sig_depth=sig_depth, layer_sizes=in_Conv_Aug_channels, include_original=True, include_time=True, window_stream=window_stream, window_size=window_size, window_stride=window_stride)
        sig_channels = _compute_sig_output_channel(in_channels, sig_depth=sig_depth, layer_sizes=in_Conv_Aug_channels, include_original=True, include_time=True)

        #self.sigBN = nn.BatchNorm1d(sig_channels)
        window_num = 1
        if window_stream:
            window_num=(stream-window_size)//window_stride + 1
        self.linear = MLP(2*sig_channels*window_num,out_dimension)
        self.feat_dict = class_dict_similarity(feat_size=dict_feat_size, input_dim=2*sig_channels*window_num, num_class=out_dimension)

    def forward(self, inp):
        # inp is a three dimensional tensor of shape (batch, stream, in_channels)
        # inp = inp.unsqueeze(-1)
        # print(f"inp.shape={inp.shape}")
        # sig Feat
        z1 = self.sig_layer1(inp)
        filp_inp = inp.flip(dims=[1])
        z2 = self.sig_layer2(filp_inp)
        if len(z1.shape)>2:
            z1 = z1.reshape(z1.shape[0],-1)
        if len(z2.shape)>2:
            z2 = z2.reshape(z2.shape[0],-1)

        z = torch.cat((z1,z2),dim=1)
        # output
        cls_out = self.linear(z)
        simi = self.feat_dict(z)

        return cls_out, simi

class SigNet_tr_global_and_inception(nn.Module):
    def __init__(self, stream, out_dimension, in_channels=3, sig_depth=2, sig_channel=128):
        super().__init__()
        # input_layer
        # in_Conv_Aug_channels=(8,8,4)
        # or in_Conv_Aug_channels=()
        self.sig_inception1 = signature_inception(in_channels=in_channels, sig_channel=sig_channel, sig_depth=sig_depth, include_whole=False)
        self.sig_inception2 = signature_inception(in_channels=in_channels, sig_channel=sig_channel, sig_depth=sig_depth, include_whole=False)
        self.sig_layer_Global1 = SigLayer(in_channels=in_channels, out_channel=sig_channel, sig_depth=sig_depth, layer_sizes=(), include_original=True, include_time=True, keep_stream=True)
        self.sig_layer_Global2 = SigLayer(in_channels=in_channels, out_channel=sig_channel, sig_depth=sig_depth, layer_sizes=(), include_original=True, include_time=True, keep_stream=True)
        
        #self.sigBN = nn.BatchNorm1d(sig_channels)
        # window num calculate
        self.win3_num = (stream-3)//1 + 1
        self.win5_num = (stream-5)//1 + 1
        self.win7_num = (stream-7)//1 + 1
        self.win9_num = (stream-9)//1 + 1
        window_num = sum([self.win3_num, self.win5_num, self.win7_num, self.win9_num])

        self.linear = MLP(2*sig_channel*(window_num+stream),out_dimension)

    def forward(self, inp):
        # inp is a three dimensional tensor of shape (batch, stream, in_channels)
        # inp = inp.unsqueeze(-1)
        # print(f"inp.shape={inp.shape}")
        # sig Feat
        z1_local = self.sig_inception1(inp)
        z1_global = self.sig_layer_Global1(inp)
        filp_inp = inp.flip(dims=[1])
        z2_local = self.sig_inception2(filp_inp)
        z2_global = self.sig_layer_Global2(filp_inp)

        z1=torch.cat((z1_local,z1_global),dim=1)
        z2=torch.cat((z2_local,z2_global),dim=1)
        if len(z1.shape)>2:
            z1 = z1.reshape(z1.shape[0],-1)
        if len(z2.shape)>2:
            z2 = z2.reshape(z2.shape[0],-1)

        z = torch.cat((z1,z2),dim=1)
        # output
        z = self.linear(z)
        return F.log_softmax(z, dim=1)
    
class SigNet_tr_inception_Transformer(nn.Module):
    def __init__(self, stream, out_dimension, in_channels=3, sig_depth=2, sig_channel=512, include_whole_sig=True):
        super().__init__()
        # input_layer
        # in_Conv_Aug_channels=(8,8,4)
        # or in_Conv_Aug_channels=()
        self.sig_layer1 = signature_inception(in_channels=in_channels, sig_channel=sig_channel, sig_depth=sig_depth, include_whole=include_whole_sig)
        self.sig_layer2 = signature_inception(in_channels=in_channels, sig_channel=sig_channel, sig_depth=sig_depth, include_whole=include_whole_sig)
        
        #self.sigBN = nn.BatchNorm1d(sig_channels)
        # window num calculate
        self.win3_num = (stream-3)//1 + 1
        self.win5_num = (stream-5)//1 + 1
        self.win7_num = (stream-7)//1 + 1
        self.win9_num = (stream-9)//1 + 1
        window_num = sum([self.win3_num, self.win5_num, self.win7_num, self.win9_num])
        if include_whole_sig:
            window_num+=1
        # self.linear = MLP(2*sig_channel*window_num,out_dimension)
        self.linear = TransformerMLP(input_dim=2*sig_channel,seq_len=window_num,out_dimension=out_dimension, num_layers=1, pooling_type='linear') #signal attention block

    def forward(self, inp):
        # inp is a three dimensional tensor of shape (batch, stream, in_channels)
        # inp = inp.unsqueeze(-1)
        # print(f"inp.shape={inp.shape}")
        # sig Feat
        z1 = self.sig_layer1(inp)
        filp_inp = inp.flip(dims=[1])
        z2 = self.sig_layer2(filp_inp)
        # if len(z1.shape)>2:
        #     z1 = z1.reshape(z1.shape[0],-1)
        # if len(z2.shape)>2:
        #     z2 = z2.reshape(z2.shape[0],-1)

        z = torch.cat((z1,z2),dim=-1)
        # output
        z = self.linear(z)
        return F.log_softmax(z, dim=1)
    
class SigNet_tr_single_inception(nn.Module):
    def __init__(self, stream, out_dimension, in_channels=3, sig_depth=2, sig_channel=512, include_whole_sig=True):
        super().__init__()
        # input_layer
        # in_Conv_Aug_channels=(8,8,4)
        # or in_Conv_Aug_channels=()
        self.sig_layer1 = signature_inception(in_channels=in_channels, sig_channel=sig_channel, sig_depth=sig_depth, include_whole=include_whole_sig)
        self.sig_layer2 = signature_inception(in_channels=in_channels, sig_channel=sig_channel, sig_depth=sig_depth, include_whole=include_whole_sig)
        
        #self.sigBN = nn.BatchNorm1d(sig_channels)
        # window num calculate
        kernels = [3,5,7,9]
        window_num = sum([(stream-k)//1 + 1 for k in kernels])
        if include_whole_sig:
            window_num+=1
        self.linear = MLP(2*sig_channel*window_num,out_dimension)

    def forward(self, inp):
        # inp is a three dimensional tensor of shape (batch, stream, in_channels)
        # inp = inp.unsqueeze(-1)
        # print(f"inp.shape={inp.shape}")
        # sig Feat
        z1 = self.sig_layer1(inp)
        filp_inp = inp.flip(dims=[1])
        z2 = self.sig_layer2(filp_inp)
        if len(z1.shape)>2:
            z1 = z1.reshape(z1.shape[0],-1)
        if len(z2.shape)>2:
            z2 = z2.reshape(z2.shape[0],-1)

        z = torch.cat((z1,z2),dim=1)
        # output
        z = self.linear(z)
        return F.log_softmax(z, dim=1)

class SigNet_tr_inception_with_mask(nn.Module):
    def __init__(self, stream, out_dimension, in_channels=3, sig_depth=2, sig_channel=512, include_whole_sig=True):
        super().__init__()
        # input_layer
        # in_Conv_Aug_channels=(8,8,4)
        # or in_Conv_Aug_channels=()
        self.sig_layer1 = sig_inception_with_mask(in_channels=in_channels, sig_channel=sig_channel, sig_depth=sig_depth, include_whole=include_whole_sig)
        self.sig_layer2 = sig_inception_with_mask(in_channels=in_channels, sig_channel=sig_channel, sig_depth=sig_depth, include_whole=include_whole_sig)
        
        #self.sigBN = nn.BatchNorm1d(sig_channels)
        # window num calculate
        kernels = [3,5,7,9]
        window_num = sum([(stream-k)//1 + 1 for k in kernels])
        if include_whole_sig:
            window_num+=1
        self.linear = MLP(2*sig_channel*window_num,out_dimension)

    def forward(self, inp):
        # inp is a three dimensional tensor of shape (batch, stream, in_channels)
        # inp = inp.unsqueeze(-1)
        # print(f"inp.shape={inp.shape}")
        # sig Feat
        z1 = self.sig_layer1(inp)
        filp_inp = inp.flip(dims=[1])
        z2 = self.sig_layer2(filp_inp)
        if len(z1.shape)>2:
            z1 = z1.reshape(z1.shape[0],-1)
        if len(z2.shape)>2:
            z2 = z2.reshape(z2.shape[0],-1)

        z = torch.cat((z1,z2),dim=1)
        # output
        z = self.linear(z)
        return F.log_softmax(z, dim=1)

class SigNet_tr_inception_with_mask_edit(nn.Module):
    def __init__(self, stream, out_dimension, in_channels=3, sig_depth=2, sig_channel=12, include_whole_sig=True, num_encoder_layers=2):
        super().__init__()
        # input_layer
        # in_Conv_Aug_channels=(8,8,4)
        # or in_Conv_Aug_channels=()
        
        
        #self.sigBN = nn.BatchNorm1d(sig_channels)
        # window num calculate
        kernels = [3,5,7]
        cal_out_stream = lambda  stream, kernel, num_layers: (stream-kernel) + 1 if num_layers==1 else cal_out_stream((stream-kernel) + 1, kernel, num_layers-1)
        window_num = sum([cal_out_stream(stream, k, num_encoder_layers) for k in kernels])
        self.sig_layer1 = nn.Sequential(*(
            [sig_inception_with_mask(in_channels=in_channels, sig_channel=sig_channel, sig_depth=sig_depth, include_whole=include_whole_sig, kernels=kernels)]
            +[sig_inception_with_mask(in_channels=sig_channel, sig_channel=sig_channel, sig_depth=sig_depth, include_whole=include_whole_sig, kernels=kernels)]*(num_encoder_layers-1)
            ))
        self.sig_layer2 = nn.Sequential(*(
            [sig_inception_with_mask(in_channels=in_channels, sig_channel=sig_channel, sig_depth=sig_depth, include_whole=include_whole_sig, kernels=kernels)]
            +[sig_inception_with_mask(in_channels=sig_channel, sig_channel=sig_channel, sig_depth=sig_depth, include_whole=include_whole_sig, kernels=kernels)]*(num_encoder_layers-1)
            ))
        if include_whole_sig:
            window_num+=1
        self.linear = MLP(2*sig_channel*window_num,out_dimension)

    def forward(self, inp):
        # inp is a three dimensional tensor of shape (batch, stream, in_channels)
        # inp = inp.unsqueeze(-1)
        # print(f"inp.shape={inp.shape}")
        # sig Feat
        z1 = self.sig_layer1(inp)
        filp_inp = inp.flip(dims=[1])
        z2 = self.sig_layer2(filp_inp)
        if len(z1.shape)>2:
            z1 = z1.reshape(z1.shape[0],-1)
        if len(z2.shape)>2:
            z2 = z2.reshape(z2.shape[0],-1)

        z = torch.cat((z1,z2),dim=1)
        # output
        z = self.linear(z)
        return F.log_softmax(z, dim=1)
    
class SigNet_tr_inception_with_mask_and_dis_feat(nn.Module):
    def __init__(self, stream, out_dimension, in_channels=3, sig_depth=2, sig_channel=512, include_whole_sig=True):
        super().__init__()
        # input_layer
        # in_Conv_Aug_channels=(8,8,4)
        # or in_Conv_Aug_channels=()
        self.sig_layer1 = sig_inception_with_mask(in_channels=in_channels, sig_channel=sig_channel, sig_depth=sig_depth, include_whole=include_whole_sig)
        self.sig_layer2 = sig_inception_with_mask(in_channels=in_channels, sig_channel=sig_channel, sig_depth=sig_depth, include_whole=include_whole_sig)
        
        #self.sigBN = nn.BatchNorm1d(sig_channels)
        # window num calculate
        kernels = [3,5,7,9]
        window_num = sum([(stream-k)//1 + 1 for k in kernels])
        if include_whole_sig:
            window_num+=1
        self.dis_feat=BundleAvgStream(input_dim=in_channels,stream=stream,out_dimension=out_dimension)
        self.linear = MLP(2*sig_channel*window_num+out_dimension,out_dimension)

    def forward(self, inp):
        # inp is a three dimensional tensor of shape (batch, stream, in_channels)
        # inp = inp.unsqueeze(-1)
        # print(f"inp.shape={inp.shape}")
        # sig Feat
        z1 = self.sig_layer1(inp)
        filp_inp = inp.flip(dims=[1])
        z2 = self.sig_layer2(filp_inp)
        if len(z1.shape)>2:
            z1 = z1.reshape(z1.shape[0],-1)
        if len(z2.shape)>2:
            z2 = z2.reshape(z2.shape[0],-1)

        dis = self.dis_feat(inp, log_softmax=False)
        z = torch.cat((z1,z2,dis),dim=1)
        # output
        z = self.linear(z)
        return F.log_softmax(z, dim=1)

class tr_LSTM_with_mask(nn.Module):
    def __init__(self, input_dim, hidden_dim, stream, out_dimension, num_layers=3):
        super().__init__()

        # 定义 LSTM 层
        self.lstm1 = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, dropout=0, batch_first=True)
        self.attn1 = conv_attn_score(dim=input_dim, kernels=1)

        self.lstm2 = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, dropout=0, batch_first=True)
        self.attn2 = conv_attn_score(dim=input_dim, kernels=1)


        # self.bn = nn.BatchNorm1d(2*hidden_dim*stream)
        self.bn = nn.BatchNorm1d(2*hidden_dim)
        # self.linear = MLP(2*hidden_dim*stream,out_dimension)
        self.linear = MLP(2*hidden_dim,out_dimension)

    def forward(self, x):

        y1, (h_n, c_n) = self.lstm1(x)
        score1 = self.attn1(x)
        y1 = y1 * score1

        flip_x = x.flip(dims=[1])
        y2, (h_n, c_n) = self.lstm2(flip_x)
        score2 = self.attn2(flip_x)
        y2 = y2 * score2

        # last time step
        y1 = y1[:,-1,:]
        y2 = y2[:,-1,:]

        y=torch.cat((y1,y2),dim=1)
        y = y.reshape(y.shape[0],-1)
        y = F.relu(self.bn(y))
        y = self.linear(y)
        
        return F.log_softmax(y, dim=1)

class tr_LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, stream, out_dimension, num_layers=3):
        super().__init__()

        # 定义 LSTM 层
        self.lstm1 = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, dropout=0, batch_first=True)

        self.lstm2 = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, dropout=0, batch_first=True)

        self.bn = nn.BatchNorm1d(2*hidden_dim*stream)
        self.linear = MLP(2*hidden_dim*stream,out_dimension)

    def forward(self, x):

        y1, (h_n, c_n) = self.lstm1(x)

        flip_x = x.flip(dims=[1])
        y2, (h_n, c_n) = self.lstm2(flip_x)

        y=torch.cat((y1,y2),dim=1)
        y = y.reshape(y.shape[0],-1)
        y = self.bn(y)
        y = self.linear(y)
        return F.log_softmax(y, dim=1)
    
class LSTM_with_mask(nn.Module):
    # cannot run
    def __init__(self, input_dim, hidden_dim, stream, out_dimension, num_layers=3):
        super().__init__()

        # 定义 LSTM 层
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, dropout=0, batch_first=True)
        self.bn = nn.BatchNorm1d(hidden_dim*stream)
        self.attn = conv_attn_score(dim=input_dim, kernels=1)

        self.linear = MLP(hidden_dim*stream,out_dimension)

    def forward(self, x):

        # LSTM 前向传播
        # print(f"x.shape={x.shape}")
        lstm_out, (h_n, c_n) = self.lstm(x)
        score = self.attn(x)
        y = lstm_out * score
        y = y.reshape(lstm_out.shape[0],-1)
        y = self.bn(y)
        
        z = self.linear(y)
        
        return F.log_softmax(z, dim=1)

class SigNet_tr_with_norm(nn.Module):
    def __init__(self, stream, out_dimension, in_channels=3, sig_depth=2, in_Conv_Aug_channels=(), window_stream=True, window_size=7, window_stride=1):
        super().__init__()
        # input_layer
        # in_Conv_Aug_channels=(8,8,4)
        # or in_Conv_Aug_channels=()
        self.sig_layer1 = SigLayer(in_channels=in_channels, sig_depth=sig_depth, layer_sizes=in_Conv_Aug_channels, include_original=True, include_time=True, window_stream=window_stream, window_size=window_size, window_stride=window_stride)
        self.sig_layer2 = SigLayer(in_channels=in_channels, sig_depth=sig_depth, layer_sizes=in_Conv_Aug_channels, include_original=True, include_time=True, window_stream=window_stream, window_size=window_size, window_stride=window_stride)
        sig_channels = _compute_sig_output_channel(in_channels, sig_depth=sig_depth, layer_sizes=in_Conv_Aug_channels, include_original=True, include_time=True)

        #self.sigBN = nn.BatchNorm1d(sig_channels)
        window_num = 1
        if window_stream:
            window_num=(stream-window_size)//window_stride + 1
        self.linear = MLP(2*sig_channels*window_num,out_dimension)

    def forward(self, inp):
        # inp is a three dimensional tensor of shape (batch, stream, in_channels)
        # inp = inp.unsqueeze(-1)
        # print(f"inp.shape={inp.shape}")
        # sig Feat
        z1 = self.sig_layer1(inp)
        filp_inp = inp.flip(dims=[1])
        z2 = self.sig_layer2(filp_inp)
        if len(z1.shape)>2:
            z1 = z1.reshape(z1.shape[0],-1)
        if len(z2.shape)>2:
            z2 = z2.reshape(z2.shape[0],-1)

        z = torch.cat((z1,z2),dim=1)
        # output
        z = self.linear(z)
        return F.log_softmax(z, dim=1)
    
class SigNet_tr_with_norm_old(nn.Module):
    def __init__(self, stream, out_dimension, in_channels=3, sig_depth=4, in_Conv_Aug_channels=(8,8,4)):
        super().__init__()
        # input_layer
        self.sig_layer1 = SigLayer(in_channels=in_channels, sig_depth=sig_depth, layer_sizes=in_Conv_Aug_channels, include_original=True, include_time=True, keep_stream=False)
        self.sig_layer2 = SigLayer(in_channels=in_channels, sig_depth=sig_depth, layer_sizes=in_Conv_Aug_channels, include_original=True, include_time=True, keep_stream=False)
        sig_channels = _compute_sig_output_channel(in_channels, sig_depth=sig_depth, layer_sizes=in_Conv_Aug_channels, include_original=True, include_time=True)

        #self.sigBN = nn.BatchNorm1d(sig_channels)
        self.linear = MLP(2*sig_channels,out_dimension)

    def forward(self, inp):
        # inp is a three dimensional tensor of shape (batch, stream, in_channels)
        # inp = inp.unsqueeze(-1)
        # print(f"inp.shape={inp.shape}")
        # sig Feat
        z1 = self.sig_layer1(inp)
        filp_inp = inp.flip(dims=[1])
        z2 = self.sig_layer2(filp_inp)

        z = torch.cat((z1,z2),dim=1)
        # output
        z = self.linear(z)
        return F.log_softmax(z, dim=1)

class SigNet_with_norm(nn.Module):
    def __init__(self, stream, out_dimension, in_channels=3, sig_depth=2, in_Conv_Aug_channels=(), window_stream=True, window_size=7, window_stride=1):
        super().__init__()
        self.sig_layer = SigLayer(in_channels=in_channels, sig_depth=sig_depth, layer_sizes=in_Conv_Aug_channels, include_original=True, include_time=True, keep_stream=False, window_stream=window_stream, window_size=window_size, window_stride=window_stride)
        sig_channels = _compute_sig_output_channel(in_channels, sig_depth=sig_depth, layer_sizes=in_Conv_Aug_channels, include_original=True, include_time=True)

        window_num = 1
        if window_stream:
            window_num=(stream-window_size)//window_stride + 1
        #self.sigBN = nn.BatchNorm1d(sig_channels)
        self.linear = MLP(sig_channels*window_num,out_dimension)

    def forward(self, inp):
        y = self.sig_layer(inp)
        if len(y.shape)>2:
            y = y.reshape(y.shape[0],-1)
        z = self.linear(y)

        return F.log_softmax(z, dim=1)
    
class SigNet_old_with_norm(nn.Module):
    def __init__(self, stream, out_dimension, in_channels=3, sig_depth=4, kernel_size=1, include_orignial=True,include_time=True,fc=True, dropout=0):
        super().__init__()
        self.augment = signatory.Augment(in_channels=in_channels,
                                         layer_sizes=(),
                                         kernel_size=kernel_size,
                                         include_original=include_orignial,
                                         include_time=include_time)

        self.signature = signatory.Signature(depth=sig_depth)

        if include_time:
            # +1 because signatory.Augment is used to add time as well
            sig_channels = signatory.signature_channels(channels=in_channels + 1,
                                                        depth=sig_depth)
        else:
            sig_channels = signatory.signature_channels(channels=in_channels,
                                                        depth=sig_depth)
        self.fc = fc
        self.dropout = nn.Dropout(p=dropout)
        self.sigBN = nn.LayerNorm(sig_channels)
        #self.sigBN = nn.BatchNorm1d(sig_channels)
        self.linear = MLP(sig_channels,out_dimension)

    def forward(self, inp, basepoint=True):
        # inp is a three dimensional tensor of shape (batch, stream, in_channels)
        # inp = inp.unsqueeze(-1)
        # print(f"inp.shape={inp.shape}")
        x = self.augment(inp)

        if x.size(1) <= 1:
            raise RuntimeError("Given an input with too short a stream to take the"
                               " signature")
        # x in a three dimensional tensor of shape (batch, stream, in_channels + 1),
        # as time has been added as a value
        y = self.signature(x, basepoint=basepoint)
        y = F.relu(self.sigBN(y))

        # y is a two dimensional tensor of shape (batch, terms), corresponding to
        # the terms of the signature
        if self.fc:
            z = self.linear(self.dropout(y))
        # z is a two dimensional tensor of shape (batch, out_dimension)
        else:
            z = y
        return F.log_softmax(z, dim=1)
    
class SigNet_old(nn.Module):
    def __init__(self, stream, out_dimension, in_channels=3, sig_depth=2, kernel_size=1, include_orignial=True,include_time=True,fc=True, dropout=0):
        super().__init__()
        self.augment = signatory.Augment(in_channels=in_channels,
                                         layer_sizes=(),
                                         kernel_size=kernel_size,
                                         include_original=include_orignial,
                                         include_time=include_time)

        self.signature = signatory.Signature(depth=sig_depth)

        if include_time:
            # +1 because signatory.Augment is used to add time as well
            sig_channels = signatory.signature_channels(channels=in_channels + 1,
                                                        depth=sig_depth)
        else:
            sig_channels = signatory.signature_channels(channels=in_channels,
                                                        depth=sig_depth)
        self.fc = fc
        self.dropout = nn.Dropout(p=dropout)
        self.linear = MLP(sig_channels,out_dimension)

    def forward(self, inp, basepoint=True):
        # inp is a three dimensional tensor of shape (batch, stream, in_channels)
        # inp = inp.unsqueeze(-1)
        # print(f"inp.shape={inp.shape}")
        x = self.augment(inp)

        if x.size(1) <= 1:
            raise RuntimeError("Given an input with too short a stream to take the"
                               " signature")
        # x in a three dimensional tensor of shape (batch, stream, in_channels + 1),
        # as time has been added as a value
        y = self.signature(x, basepoint=basepoint)

        # y is a two dimensional tensor of shape (batch, terms), corresponding to
        # the terms of the signature
        if self.fc:
            z = self.linear(self.dropout(y))
        # z is a two dimensional tensor of shape (batch, out_dimension)
        else:
            z = y
        return F.log_softmax(z, dim=1)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 100):
        # TODO stream should be less than max_len: 15<100 correct
        assert d_model % 2 == 0, 'odd d_model is not supported'
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(1), :]
        return self.dropout(x)
        
class TransformerClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,          # 输入特征维度 C
        seq_len: int,            # 序列长度 T
        out_dimension: int,      # 输出类别数
        d_model: int = 512,      # 输入层特征通道数，Transformer的d_model
        num_heads: int = 16,      # Transformer头数
        num_layers: int = 2,     # Transformer层数
        dim_feedforward: int = 512,  # FFN隐藏层维度
        dropout: float = 0,
        pooling_type: str = "mean"
    ):
        super().__init__()
        
        self.pooling_type = pooling_type

        # 经典的位置编码
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # 输入编码
        self.input_embedding = nn.Linear(input_dim, d_model)

        # 位置编码（可学习）
        # self.positional_embedding = nn.Parameter(torch.randn(1, seq_len, d_model))
        
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True  # 可以使用(batch, seq, dim)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 分类头
        # self.classifier = nn.Sequential(
        #     nn.Linear(d_model, dim_feedforward),
        #     nn.ReLU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(dim_feedforward, out_dimension)
        # )
        self.classifier = MLP(d_model, out_dimension)

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        """
        输入: 
            x -> [B, T, C]
        输出: 
            logits -> [B, out_dimension]
        """
        # 如果提供padding mask (形状[B, T])
        if mask is not None:
            src_key_padding_mask = mask.bool()
            transformer_out = self.transformer_encoder(
                x, 
                src_key_padding_mask=src_key_padding_mask
            )
        # x [B,T,input_dim]
        x = self.input_embedding(x) # [B, T, C=d_model]
        B, T, C = x.shape
        
        # 添加位置编码
        # x = x + self.positional_embedding[:, :T, :] # 位置编码（可学习） # 自动广播到[B, T, C]
        x = self.pos_encoder(x) # 经典的位置编码
        
        # # 调整维度为PyTorch Transformer格式 (T, B, C)
        # x = x.permute(1, 0, 2)  # [T, B, C]
        
        # Transformer编码
        transformer_out = self.transformer_encoder(x)  # [B, T, C]
        
        # 恢复维度并池化
        transformer_out = transformer_out #.permute(1, 0, 2)  # [B, T, C]
        # 修改池化部分
        if self.pooling_type == "mean":
            pooled = transformer_out.mean(dim=1)
        elif self.pooling_type == "max":
            pooled = transformer_out.max(dim=1).values
        elif self.pooling_type == "cls":
            # 使用第一个token作为分类特征
            pooled = transformer_out[:, 0, :]

        # 分类输出
        logits = self.classifier(pooled)  # [B, out_dimension]
        return F.log_softmax(logits, dim=1)

class Transformer_tr_with_mask(nn.Module):
    def __init__(
        self,
        input_dim: int,          # 输入特征维度 C
        seq_len: int,            # 序列长度 T
        out_dimension: int,      # 输出类别数
        d_model: int = 512,      # 输入层特征通道数，Transformer的d_model
        num_heads: int = 16,      # Transformer头数
        num_layers: int = 2,     # Transformer层数
        dim_feedforward: int = 512,  # FFN隐藏层维度
        dropout: float = 0,
        pooling_type: str = "mean"
    ):
        super().__init__()
        
        self.pooling_type = pooling_type

        # 输入编码
        self.input_embedding = nn.Linear(input_dim, d_model)

        # 位置编码（可学习）
        # self.positional_embedding = nn.Parameter(torch.randn(1, 2*seq_len, d_model))
        # 经典的位置编码
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True  # 可以使用(batch, seq, dim)
        )
        self.transformer_encoder1 = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.transformer_encoder2 = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.score1 = conv_attn_score(dim=d_model, kernels=1)
        self.score2 = conv_attn_score(dim=d_model, kernels=1)
        
        # 分类头
        # self.classifier = nn.Sequential(
        #     nn.Linear(d_model, dim_feedforward),
        #     nn.ReLU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(dim_feedforward, out_dimension)
        # )
        self.classifier = MLP(2*d_model, out_dimension)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入: 
            x -> [B, T, C]
        输出: 
            logits -> [B, out_dimension]
        """

        # x [B,T,input_dim]
        B, T, C = x.shape
        x = torch.cat([x,x.flip(dims=[1])],dim=1)
        x = self.input_embedding(x) # [B, 2*T, C=d_model]
        
        
        
        # 添加位置编码
        # x = x + self.positional_embedding[:, :2*T, :]  # 自动广播到[B, 2*T, C]
        x = self.pos_encoder(x) # 经典的位置编码
        
        # # 调整维度为PyTorch Transformer格式 (T, B, C)
        # x = x.permute(1, 0, 2)  # [T, B, C]
        
        x1=x[:,:T,:]
        x2=x[:,T:,:]
        # Transformer编码
        y1= self.transformer_encoder1(x1)  # [B, T, C]
        y2= self.transformer_encoder2(x2)  # [B, T, C]
        
        
        # mask
        y1 = y1 * self.score1(x1)
        y2 = y2 * self.score1(x2)

        # 修改池化部分
        if self.pooling_type == "mean":
            pooled = torch.cat([y1.mean(dim=1), y2.mean(dim=1)],dim=-1)
        elif self.pooling_type == "max":
            pooled = torch.cat([y1.max(dim=1).values, y2.max(dim=1).values],dim=-1)
        elif self.pooling_type == "cls":
            # 使用第一个token作为分类特征
            pooled = torch.cat([y1[:, 0, :], y2[:, 0, :]],dim=-1)

        # 分类输出
        logits = self.classifier(pooled)  # [B, out_dimension]
        return F.log_softmax(logits, dim=1)

class Transformer_tr(nn.Module):
    def __init__(
        self,
        input_dim: int,          # 输入特征维度 C
        seq_len: int,            # 序列长度 T
        out_dimension: int,      # 输出类别数
        d_model: int = 512,      # 输入层特征通道数，Transformer的d_model
        num_heads: int = 4,      # Transformer头数
        num_layers: int = 2,     # Transformer层数
        dim_feedforward: int = 512,  # FFN隐藏层维度
        dropout: float = 0.1,
        pooling_type: str = "mean"
    ):
        super().__init__()
        
        self.pooling_type = pooling_type

        # 输入编码
        self.input_embedding = nn.Linear(input_dim, d_model)

        # 位置编码（可学习）
        self.positional_embedding = nn.Parameter(torch.randn(1, 2*seq_len, d_model))
        # 经典的位置编码
        #self.pos_encoder = PositionalEncoding(input_dim, dropout)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True  # 可以使用(batch, seq, dim)
        )
        self.transformer_encoder1 = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.transformer_encoder2 = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 分类头
        # self.classifier = nn.Sequential(
        #     nn.Linear(d_model, dim_feedforward),
        #     nn.ReLU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(dim_feedforward, out_dimension)
        # )
        self.classifier = MLP(2*d_model, out_dimension)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入: 
            x -> [B, T, C]
        输出: 
            logits -> [B, out_dimension]
        """

        # x [B,T,input_dim]
        B, T, C = x.shape
        x = torch.cat([x,x.flip(dims=[1])],dim=1)
        x = self.input_embedding(x) # [B, 2*T, C=d_model]
        
        
        # 添加位置编码
        x = x + self.positional_embedding[:, :2*T, :]  # 自动广播到[B, 2*T, C]
        
        # # 调整维度为PyTorch Transformer格式 (T, B, C)
        # x = x.permute(1, 0, 2)  # [T, B, C]
        
        x1=x[:,:T,:]
        x2=x[:,T:,:]
        # Transformer编码
        y1= self.transformer_encoder1(x1)  # [B, T, C]
        y2= self.transformer_encoder2(x2)  # [B, T, C]
        

        # 修改池化部分
        if self.pooling_type == "mean":
            pooled = torch.cat([y1.mean(dim=1), y2.mean(dim=1)],dim=-1)
        elif self.pooling_type == "max":
            pooled = torch.cat([y1.max(dim=1).values, y2.max(dim=1).values],dim=-1)
        elif self.pooling_type == "cls":
            # 使用第一个token作为分类特征
            pooled = torch.cat([y1[:, 0, :], y2[:, 0, :]],dim=-1)

        # 分类输出
        logits = self.classifier(pooled)  # [B, out_dimension]
        return F.log_softmax(logits, dim=1)

class Transformer_with_mask(nn.Module):
    def __init__(
        self,
        input_dim: int,          # 输入特征维度 C
        seq_len: int,            # 序列长度 T
        out_dimension: int,      # 输出类别数
        d_model: int = 512,      # 输入层特征通道数，Transformer的d_model
        num_heads: int = 4,      # Transformer头数
        num_layers: int = 2,     # Transformer层数
        dim_feedforward: int = 512,  # FFN隐藏层维度
        dropout: float = 0.1,
        pooling_type: str = "mean"
    ):
        super().__init__()
        
        self.pooling_type = pooling_type

        # 输入编码
        self.input_embedding = nn.Linear(input_dim, d_model)

        # 位置编码（可学习）
        self.positional_embedding = nn.Parameter(torch.randn(1, seq_len, d_model))
        # 经典的位置编码
        #self.pos_encoder = PositionalEncoding(input_dim, dropout)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True  # 可以使用(batch, seq, dim)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.score = conv_attn_score(dim=d_model, kernels=1)
        
        # 分类头
        # self.classifier = nn.Sequential(
        #     nn.Linear(d_model, dim_feedforward),
        #     nn.ReLU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(dim_feedforward, out_dimension)
        # )
        self.classifier = MLP(d_model, out_dimension)

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        """
        输入: 
            x -> [B, T, C]
        输出: 
            logits -> [B, out_dimension]
        """
        # 如果提供padding mask (形状[B, T])
        if mask is not None:
            src_key_padding_mask = mask.bool()
            transformer_out = self.transformer_encoder(
                x, 
                src_key_padding_mask=src_key_padding_mask
            )
        # x [B,T,input_dim]
        x = self.input_embedding(x) # [B, T, C=d_model]
        B, T, C = x.shape
        
        # 添加位置编码
        x = x + self.positional_embedding[:, :T, :]  # 自动广播到[B, T, C]
        
        # # 调整维度为PyTorch Transformer格式 (T, B, C)
        # x = x.permute(1, 0, 2)  # [T, B, C]
        
        # Transformer编码
        transformer_out = self.transformer_encoder(x)  # [B, T, C]
        
        # 恢复维度并池化
        transformer_out = transformer_out #.permute(1, 0, 2)  # [B, T, C]
        
        # mask
        transformer_out = transformer_out * self.score(x)
        # 修改池化部分
        if self.pooling_type == "mean":
            pooled = transformer_out.mean(dim=1)
        elif self.pooling_type == "max":
            pooled = transformer_out.max(dim=1).values
        elif self.pooling_type == "cls":
            # 使用第一个token作为分类特征
            pooled = transformer_out[:, 0, :]

        # 分类输出
        logits = self.classifier(pooled)  # [B, out_dimension]
        return F.log_softmax(logits, dim=1)

class DeepWMA_conv(nn.Module):
    def __init__(self, in_channel, stream, out_dimension):
        super().__init__()
        self.conv_32_1 = nn.Conv2d(in_channel, 32, kernel_size=3, padding=1)
        self.conv_32_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.maxpool_32 = nn.MaxPool2d(kernel_size=2)
        self.dropout_32 = nn.Dropout(p=0.25)

        self.conv_64_1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv_64_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.maxpool_64 = nn.MaxPool2d(kernel_size=2)
        self.dropout_64 = nn.Dropout(p=0.25)

        self.conv_128_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv_128_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.maxpool_128 = nn.MaxPool2d(kernel_size=2)
        self.dropout_128 = nn.Dropout(p=0.25)

        self.conv_256_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv_256_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.maxpool_256 = nn.MaxPool2d(kernel_size=2)
        #self.dropout_256 = nn.Dropout(p=0.25)

        # self.conv_512_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        # self.conv_512_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        # self.maxpool_512 = nn.MaxPool2d(kernel_size=2)
        # #self.dropout_512 = nn.Dropout(p=0.25)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256*(int((2*stream)//2//2//2//2)**2), 128)
        self.dp1 = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(128, 256)
        self.dp2 = nn.Dropout(p=0.2)
        self.fc3 = nn.Linear(256, 512)
        self.dp3 = nn.Dropout(p=0.2)
        self.fc4 = nn.Linear(512, out_dimension)

    def forward(self, x):
        # 32 channels
        x = self.conv_32_1(x)
        x = F.relu(x)
        x = self.conv_32_2(x)
        x = F.relu(x)
        x = self.maxpool_32(x)
        x = self.dropout_32(x)

        # 64 channels
        x = self.conv_64_1(x)
        x = F.relu(x)
        x = self.conv_64_2(x)
        x = F.relu(x)
        x = self.maxpool_64(x)
        x = self.dropout_64(x)

        # 128 channels
        x = self.conv_128_1(x)
        x = F.relu(x)
        x = self.conv_128_2(x)
        x = F.relu(x)
        x = self.maxpool_128(x)
        x = self.dropout_128(x)

        # 256 channels
        x = self.conv_256_1(x)
        x = F.relu(x)
        x = self.conv_256_2(x)
        x = F.relu(x)
        x = self.maxpool_256(x)

        # x = self.dropout_256(x)

        # # 512 channels
        # x = self.conv_512_1(x)
        # x = F.relu(x)
        # x = self.conv_512_2(x)
        # x = F.relu(x)
        # x = self.maxpool_512(x)

        # Dense
        x = self.flatten(x)
        x = self.dp1(F.relu(self.fc1(x)))
        x = self.dp2(F.relu(self.fc2(x)))
        x = self.dp3(F.relu(self.fc3(x)))
        x = self.fc4(x)
        x = F.log_softmax(x,dim=1)

        return x

#### Ablation Study ####

class SigNet_inception_with_mask(nn.Module):
    def __init__(self, stream, out_dimension, in_channels=3, sig_depth=2, sig_channel=512, include_whole_sig=True):
        super().__init__()
        # input_layer
        # in_Conv_Aug_channels=(8,8,4)
        # or in_Conv_Aug_channels=()
        self.sig_layer = sig_inception_with_mask(in_channels=in_channels, sig_channel=sig_channel, sig_depth=sig_depth, include_whole=include_whole_sig)
           
        #self.sigBN = nn.BatchNorm1d(sig_channels)
        # window num calculate
        kernels = [3,5,7,9]
        window_num = sum([(stream-k)//1 + 1 for k in kernels])
        if include_whole_sig:
            window_num+=1
        self.linear = MLP(sig_channel*window_num,out_dimension)

    def forward(self, inp):
        # inp is a three dimensional tensor of shape (batch, stream, in_channels)
        # inp = inp.unsqueeze(-1)
        # print(f"inp.shape={inp.shape}")
        # sig Feat
        z = self.sig_layer(inp)
        z = z.view(z.shape[0],-1)

        # output
        z = self.linear(z)
        return F.log_softmax(z, dim=1)

class SigNet_tr_single_with_mask(nn.Module):
    def __init__(self, stream, out_dimension, in_channels=3, sig_depth=2, sig_channel=512, include_whole_sig=False):
        super().__init__()
        # input_layer
        # in_Conv_Aug_channels=(8,8,4)
        # or in_Conv_Aug_channels=()

        # window num calculate
        kernels = [5]
        window_num = sum([(stream-k)//1 + 1 for k in kernels])
        if include_whole_sig:
            window_num+=1
        self.sig_layer1 = sig_inception_with_mask(in_channels=in_channels, sig_channel=sig_channel, kernels=kernels, sig_depth=sig_depth, include_whole=include_whole_sig)
        self.sig_layer2 = sig_inception_with_mask(in_channels=in_channels, sig_channel=sig_channel, kernels=kernels, sig_depth=sig_depth, include_whole=include_whole_sig)
        
        #self.sigBN = nn.BatchNorm1d(sig_channels)
        
        self.linear = MLP(2*sig_channel*window_num,out_dimension)

    def forward(self, inp):
        # inp is a three dimensional tensor of shape (batch, stream, in_channels)
        # inp = inp.unsqueeze(-1)
        # print(f"inp.shape={inp.shape}")
        # sig Feat
        z1 = self.sig_layer1(inp)
        filp_inp = inp.flip(dims=[1])
        z2 = self.sig_layer2(filp_inp)
        if len(z1.shape)>2:
            z1 = z1.reshape(z1.shape[0],-1)
        if len(z2.shape)>2:
            z2 = z2.reshape(z2.shape[0],-1)

        z = torch.cat((z1,z2),dim=1)
        # output
        z = self.linear(z)
        return F.log_softmax(z, dim=1)

class TCN_tr_single(nn.Module):
    def __init__(self, stream, out_dimension, in_channels=3, sig_depth=2, window_size=7, window_stride=1):
        super().__init__()
        pass

    def forward(self, inp):
        pass

class TCN_tr_inception(nn.Module):
    def __init__(self, stream, out_dimension, in_channels=3, sig_depth=2, window_size=7, window_stride=1):
        super().__init__()
        pass

    def forward(self, inp):
        pass

class TCN_tr_inception_with_mask(nn.Module):
    def __init__(self, stream, out_dimension, in_channels=3, sig_depth=2, feat_channel=512, include_whole=True):
        super().__init__()
        # input_layer
        # in_Conv_Aug_channels=(8,8,4)
        # or in_Conv_Aug_channels=()
        self.conv_layer1 = TCN_inception_with_mask_layer(in_channels=in_channels, stream=stream, out_channels=feat_channel, sig_depth=sig_depth, include_whole=include_whole)
        self.conv_layer2 = TCN_inception_with_mask_layer(in_channels=in_channels, stream=stream, out_channels=feat_channel, sig_depth=sig_depth, include_whole=include_whole)
        
        #self.sigBN = nn.BatchNorm1d(sig_channels)
        # window num calculate
        kernels = [3,5,7,9]
        window_num = sum([(stream-k)//1 + 1 for k in kernels])
        if include_whole:
            window_num+=1
        self.linear = MLP(2*feat_channel*window_num,out_dimension)

    def forward(self, inp):
        # inp is a three dimensional tensor of shape (batch, stream, in_channels)
        # inp = inp.unsqueeze(-1)
        # print(f"inp.shape={inp.shape}")
        # sig Feat
        z1 = self.conv_layer1(inp)
        filp_inp = inp.flip(dims=[1])
        z2 = self.conv_layer2(filp_inp)
        if len(z1.shape)>2:
            z1 = z1.reshape(z1.shape[0],-1)
        if len(z2.shape)>2:
            z2 = z2.reshape(z2.shape[0],-1)

        z = torch.cat((z1,z2),dim=1)
        # output
        z = self.linear(z)
        return F.log_softmax(z, dim=1)


if __name__=='__main__':
    batch = 32
    stream = 15
    channels = 3
    # model = SigFeat(3,1024)#SigLayer(3)
    # # print(model.new_channel)
    # # print(model.out_sig_channels)
    # path = torch.rand(batch, stream, channels)
    # y = model(path)
    # print(y.shape)


    feat_size=256
    input_dim=15*3
    num_class=73
    input_1 = torch.rand(batch, stream*channels)

    layer = class_dict_similarity(feat_size, input_dim, num_class, simi_type='dot')
    print(layer(input_1))
    print(layer(input_1).shape)
    print(F.softmax(layer(input_1),dim=-1))

