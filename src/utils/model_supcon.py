"""Reference from https://github.com/fxia22/pointnet.pytorch"""
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F

from utils.model import *
import utils.model as model

def get_supCon_model(model_name='PointNet_SupCon', contra_feat_dim=128, cls_feat_dim=1024, stream=15, dict_feat_size=1024):
    # 所有模型有 self.encoder 属性作为前部
    if model_name=='PointNet_SupCon' or model_name=='PointNet':
        return PointNet_SupCon(feat_dim=contra_feat_dim)  # Remove transformation nets
    elif model_name=='LSTM_SupCon':
        return LSTM_SupCon(stream=stream, encoder_out=cls_feat_dim, feat_dim=contra_feat_dim)
    elif model_name=='Transformer_SupCon':
        return Transformer_SupCon(stream=stream, encoder_out=cls_feat_dim, feat_dim=contra_feat_dim)
    elif model_name=='Single_SigNet_tr_SupCon':
        return Single_SigNet_tr_SupCon(stream=stream, encoder_out=cls_feat_dim, feat_dim=contra_feat_dim)
    elif model_name=='Single_SigNet_tr_encoder':
        return Single_SigNet_tr_encoder(stream=stream ,out_dimension=cls_feat_dim)
    elif model_name=='DeepWMA':
        return DeepWMA_encoder(in_channel=3, stream=stream, out_dimension=cls_feat_dim)
    elif model_name=='Transformer_encoder':
        return Transformer_encoder(input_dim=3,seq_len=stream,out_dimension=cls_feat_dim)
    elif model_name=='LSTM_encoder':
        return LSTM_encoder(input_dim=3, hidden_dim=128, stream=stream,out_dimension=cls_feat_dim)
    else:
        raise RuntimeError('Invalid model name.')

def get_cls_model(model_name='PointNet_Classifier', num_classes=2, cls_feat_dim=1024, dict_feat_size=1024):
    if model_name=='PointNet_Classifier' or model_name=='PointNet':
        return PointNet_Classifier(num_classes=num_classes)  # Remove transformation nets
    elif model_name=='MLP':
        return MLP(cls_feat_dim, num_class)
    else:
        raise RuntimeError('Invalid model name.')
    
class PointNet_SupCon(nn.Module):
    """PointNet Encoder+Linear layers. Trained with contrastive loss"""
    def __init__(self, head='mlp', feat_dim=128):
        super(PointNet_SupCon, self).__init__()
        # encoder
        self.encoder = PointNetfeat()
        # Contrastive learning
        if head == 'linear':
            self.head = nn.Linear(1024, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(1024, 1024),
                nn.ReLU(inplace=True),
                nn.Linear(1024, feat_dim)
            )
        else:
            raise ValueError('Head not supported: {}. Please select from "mlp" or "linear"'.format(head))

    def forward(self, x):
        global_feat = self.encoder(x)
        # contrastive feature
        contra_feat = F.normalize(self.head(global_feat), dim=1)  # normalization is important

        return contra_feat

class LSTM_SupCon(nn.Module):
    """Encoder+Linear layers. Trained with contrastive loss"""
    def __init__(self, head='mlp', encoder_out=1024, stream=15, feat_dim=128):
        super().__init__()
        # encoder
        self.encoder = LSTM_encoder(input_dim=3, hidden_dim=128, stream=stream,out_dimension=encoder_out)
        # Contrastive learning
        if head == 'linear':
            self.head = nn.Linear(encoder_out, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(encoder_out, 1024),
                nn.ReLU(inplace=True),
                nn.Linear(1024, feat_dim)
            )
        else:
            raise ValueError('Head not supported: {}. Please select from "mlp" or "linear"'.format(head))

    def forward(self, x):
        global_feat = self.encoder(x)
        # contrastive feature
        contra_feat = F.normalize(self.head(global_feat), dim=1)  # normalization is important

        return contra_feat

class Transformer_SupCon(nn.Module):
    """Encoder+Linear layers. Trained with contrastive loss"""
    def __init__(self, head='mlp', encoder_out=1024, stream=15, feat_dim=128):
        super().__init__()
        # encoder
        self.encoder = Transformer_encoder(input_dim=3,seq_len=stream,out_dimension=encoder_out)
        # Contrastive learning
        if head == 'linear':
            self.head = nn.Linear(encoder_out, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(encoder_out, 1024),
                nn.ReLU(inplace=True),
                nn.Linear(1024, feat_dim)
            )
        else:
            raise ValueError('Head not supported: {}. Please select from "mlp" or "linear"'.format(head))

    def forward(self, x):
        global_feat = self.encoder(x)
        # contrastive feature
        contra_feat = F.normalize(self.head(global_feat), dim=1)  # normalization is important

        return contra_feat
    
class Single_SigNet_tr_SupCon(nn.Module):
    """Encoder+Linear layers. Trained with contrastive loss"""
    def __init__(self, head='mlp', encoder_out=1024, stream=15, feat_dim=128):
        super().__init__()
        # encoder
        self.encoder = Single_SigNet_tr_encoder(stream=stream, out_dimension=encoder_out)
        # Contrastive learning
        if head == 'linear':
            self.head = nn.Linear(encoder_out, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(encoder_out, 1024),
                nn.ReLU(inplace=True),
                nn.Linear(1024, feat_dim)
            )
        else:
            raise ValueError('Head not supported: {}. Please select from "mlp" or "linear"'.format(head))

    def forward(self, x):
        global_feat = self.encoder(x)
        # contrastive feature
        contra_feat = F.normalize(self.head(global_feat), dim=1)  # normalization is important

        return contra_feat

class PointNetfeat(nn.Module):
    def __init__(self):
        super(PointNetfeat, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

    def forward(self, x):
        x = x.transpose(2, 1) # points [B, 3, N] # newly added beacause train.py discarded it
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        return x


class PointNet_Classifier(nn.Module):
    """The classifier layers in PointNet. Trained with CrossEntropy loss based on the fixed encoder"""
    def __init__(self, num_classes=2):
        super(PointNet_Classifier, self).__init__()
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)

        return F.log_softmax(x, dim=1)

class Single_SigNet_tr_encoder(nn.Module):
    def __init__(self, stream=15, out_dimension=1024, in_channels=3, sig_depth=2, in_Conv_Aug_channels=(), window_stream=True, window_size=7, window_stride=1, dict_feat_size=256):
        super().__init__()
        # input_layer
        self.sig_layer1 = SigLayer(in_channels=in_channels, sig_depth=sig_depth, layer_sizes=in_Conv_Aug_channels, include_original=True, include_time=True, window_stream=window_stream, window_size=window_size, window_stride=window_stride)
        self.sig_layer2 = SigLayer(in_channels=in_channels, sig_depth=sig_depth, layer_sizes=in_Conv_Aug_channels, include_original=True, include_time=True, window_stream=window_stream, window_size=window_size, window_stride=window_stride)
        sig_channels = model._compute_sig_output_channel(in_channels, sig_depth=sig_depth, layer_sizes=in_Conv_Aug_channels, include_original=True, include_time=True)

        window_num = 1
        if window_stream:
            window_num=(stream-window_size)//window_stride + 1

        self.modulate = nn.Linear(2*sig_channels*window_num, out_dimension)

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
        return F.normalize(self.modulate(z),dim=-1)


class LSTM_encoder(nn.Module):
    # cannot run
    def __init__(self, input_dim, hidden_dim, stream, out_dimension, num_layers=3):
        super().__init__()

        # 定义 LSTM 层
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, dropout=0, batch_first=True)
        self.bn = nn.BatchNorm1d(hidden_dim*stream)

        # self.linear = MLP(hidden_dim*stream,out_dimension)
        self.modulate = nn.Linear(hidden_dim*stream, out_dimension)

    def forward(self, x):

        # LSTM 前向传播
        # print(f"x.shape={x.shape}")
        lstm_out, (h_n, c_n) = self.lstm(x)
        y = lstm_out.reshape(lstm_out.shape[0],-1)
        y = self.bn(y)
        
        # output
        return F.normalize(self.modulate(y),dim=-1)

class Transformer_encoder(nn.Module):
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

        # 输入编码
        self.input_embedding = nn.Linear(input_dim, d_model)

        # 位置编码（可学习）
        self.positional_embedding = nn.Parameter(torch.randn(1, seq_len, d_model))
        # 经典的位置编码
        # self.pos_encoder = PositionalEncoding(input_dim, dropout)
        
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
        

        self.modulate = nn.Linear(d_model, out_dimension)

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
        # x = self.self.pos_encoder(x)
        
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


        # output
        return F.normalize(self.modulate(pooled),dim=-1)

class DeepWMA_encoder(nn.Module):
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
        self.modulate = nn.Linear(256*int((2*stream)//2//2//2//2), out_dimension)

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

        x = self.flatten(x)

        return F.normalize(self.modulate(x),dim=-1)

if __name__=='__main__':
    import pickle
    stage1_model = PointNetCls(k=73)
    stage2_encoder = PointNet_SupCon(head='mlp', feat_dim=128)
    stage2_classifer = PointNet_Classifier(num_classes=73)

    # stage1_model = PointNetCls(k=stage1_params['stage1_num_class'])

    pick_path = '/home/rench/code/WMA-related/demo/encoder_params.pickle'
    with open(pick_path, 'rb') as f: # stage 2, encoder with contrastive learning
        encoder_params = pickle.load(f)
        f.close()

    stage2_encoder = PointNet_SupCon(head=encoder_params['head_name'], feat_dim=encoder_params['encoder_feat_num'])
    stage2_classifer = PointNet_Classifier(num_classes=encoder_params['stage2_num_class'])

    stage2_encoder.load_state_dict(torch.load('/home/rench/code/WMA-related/demo/epoch_100_model.pth', weights_only=True, map_location='cpu'))
    stage2_classifer.load_state_dict(torch.load('/home/rench/code/WMA-related/demo/classifier_best_f1_model.pth', weights_only=True, map_location='cpu'))
    test_x = torch.randn(32, 15, 3)
    print(test_x.shape)
    print(stage2_encoder(test_x).shape)
    print(stage2_classifer(stage2_encoder.encoder(test_x)).shape)
    print(hasattr(stage2_encoder,'encoder'))
    # for name,para in stage2_encoder.named_parameters():
    #     print(name)
    