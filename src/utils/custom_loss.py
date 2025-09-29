# SupConLoss function is modified from https://github.com/HobbitLong/SupContrast Yonglong Tian (yonglong@mit.edu)
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_criterion(loss_name='nll_loss', num_classes=None, device='cpu', dict_feat_size=256, temperature=0.07):
    if loss_name=='label_smoothing_classic':
        return LabelSmoothingNllLoss()
    elif loss_name=='LabelConfusion_loss':
        return LabelConfusionCrossEntropy()
    elif loss_name=='LabelDictionaryAndCrossEntropy':
        return LabelDictionaryAndCrossEntropy(num_classes=num_classes, device=device, dict_feat_size=dict_feat_size)
    elif loss_name=='SupConAndCrossEntropy':
        return SupConAndCrossEntropy()
    elif loss_name=='supCon':
        return SupConLoss(temperature=temperature)
    elif loss_name=='nll_loss':
        return safe_nll_loss()
    elif loss_name=='nll_loss_raw':
        return F.nll_loss

class safe_nll_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, targets):
        if not isinstance(logits, torch.Tensor):
            logits = logits[0]
        return F.nll_loss(logits, targets)
    
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, epsilon=0.1):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, logits, targets):
        """
        参数:
            logits: 模型输出 [B, C]
            targets: 真实标签 [B]
        返回:
            平滑后的交叉熵损失
        """
        if not isinstance(logits, torch.Tensor):
            logits = logits[0]
        num_classes = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)
        
        # 构建目标分布
        target_probs = torch.full_like(log_probs, 
                                     self.epsilon / (num_classes - 1))
        target_probs.scatter_(1, targets.unsqueeze(1), 1 - self.epsilon)
        
        # 计算损失
        loss = (-target_probs * log_probs).sum(dim=1).mean()
        return loss

    
class LabelSmoothingNllLoss(nn.Module):
    def __init__(self, epsilon=0.1):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, logits, targets):
        """
        参数:
            logits: 模型输出 [B, C]
            targets: 真实标签 [B]
        返回:
            平滑后的交叉熵损失
        """
        if not isinstance(logits, torch.Tensor):
            logits = logits[0]
        num_classes = logits.size(-1)
        # log_probs = F.log_softmax(logits, dim=-1)
        log_probs = logits
        
        # 构建目标分布
        target_probs = torch.full_like(log_probs, 
                                     self.epsilon / (num_classes - 1))
        target_probs.scatter_(1, targets.unsqueeze(1), 1 - self.epsilon)
        
        # 计算损失
        loss = (-target_probs * log_probs).sum(dim=1).mean()
        return loss

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
    

class LabelDictionaryAndCrossEntropy(nn.Module):
    def __init__(self, dict_feat_size, num_classes, device, loss_epsilon=0.5, mom_epsilon=0.1, simi_type='cos', para_learnable=False):
        super().__init__()
        # self.cls_loss = classic_loss
        self.loss_epsilon = loss_epsilon
        self.mom_epsilon = mom_epsilon
        self.feat_size=dict_feat_size
        self.num_class=num_classes
        self.simi_type = simi_type
        # 各个类别的原型信号，可学习或固定？
        self.dict = nn.Parameter(torch.randn(num_classes, dict_feat_size),requires_grad=para_learnable)
        self.to(device) # self.dict.to(device) cannot function


    def forward(self, logits, targets):
        # 但是梯度更新也有把输入的特征来更新
        """
        参数:
            logits: 模型输出预测概率和与类代表特征相似性 [pred:[B, C], feat:[B, feat_size]]
            targets: 真实标签 [B]
        返回:
            标签混合后的交叉熵损失
        """
        pred, feat = logits
        num_classes = pred.shape[1]
        assert num_classes==self.num_class, "class dictionary size is not the same as number of classes"
        
        with torch.no_grad():  # 确保更新不影响梯度计算
            # 计算 refine 量 ------------------------------------
            # 创建类别掩码 [B, C]

            one_hot = F.one_hot(targets, self.num_class).float()  # [B, C]
            
            # 计算每个类别的特征均值 [C, D]
            sum_feat = torch.mm(one_hot.T, feat)                # [C, D]
            count = one_hot.sum(dim=0, keepdim=True).T          # [C, 1]
            valid_mask = (count > 0).float()
            
            # 防止除零，计算有效类别的均值
            class_mean = sum_feat / (count + 1e-6)              # [C, D]
            
            # 计算 refine 量 (新均值与旧字典的差异)
            refine = (class_mean - self.dict) * valid_mask      # [C, D]
            
            # 动量更新字典 ---------------------------------------
            # 更新出现过的类别
            self.dict += self.mom_epsilon * refine # use"= *+*" cannot function
            

        if self.simi_type=='cos':
            # 扩展维度用于广播计算
            feat = feat.unsqueeze(1)  # [B, 1, feat_size]
            feat_expanded = self.dict.unsqueeze(0)  # [1, num_class, feat_size]
            simi = F.cosine_similarity(feat, feat_expanded, dim=-1)
        elif self.simi_type=='dot':
            simi = torch.matmul(feat, self.dict.transpose(1, 0))
        elif self.simi_type=='L1':
            # 扩展维度用于广播计算
            feat = feat.unsqueeze(1)  # [B, 1, feat_size]
            feat_expanded = self.dict.unsqueeze(0)  # [1, num_class, feat_size]
            simi = torch.abs(feat-feat_expanded).sum(dim=-1)
        elif self.simi_type=='L2':
            # 扩展维度用于广播计算
            feat = feat.unsqueeze(1)  # [B, 1, feat_size]
            feat_expanded = self.dict.unsqueeze(0)  # [1, num_class, feat_size]
            simi = (feat-feat_expanded)**2
            simi = simi.sum(dim=-1)
        targets = F.one_hot(targets, num_classes)
        target_probs = 1/2*(self.loss_epsilon*F.softmax(simi,dim=-1) + (1-self.loss_epsilon)*targets)
        log_probs = F.log_softmax(pred, dim=-1)
        
        # 计算损失
        loss = (-target_probs * log_probs).sum(dim=1).mean()
        return loss

    
class LabelConfusionCrossEntropy(nn.Module):
    def __init__(self, epsilon=0.1):
        super().__init__()
        # self.cls_loss = classic_loss
        self.epsilon = epsilon

    def forward(self, logits, targets):
        # 网络的dictionary参数也动，可能会不稳定
        # 但是梯度更新也有把输入的特征来更新
        """
        参数:
            logits: 模型输出预测概率和与类代表特征相似性 [pred:[B, C], simi:[B, C]]
            targets: 真实标签 [B]
        返回:
            标签混合后的交叉熵损失
        """
        pred,simi = logits
        assert pred.shape==simi.shape, 'Not consistent shape'
        num_classes = pred.shape[1]
        targets = F.one_hot(targets, num_classes)
        target_probs = 1/2*(self.epsilon*F.softmax(simi,dim=-1) + (1-self.epsilon)*targets)
        log_probs = F.log_softmax(pred, dim=-1)
        
        # 计算损失
        loss = (-target_probs * log_probs).sum(dim=1).mean()
        return loss

class SupConAndCrossEntropy(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        # self.cls_loss = classic_loss
        self.alpha = alpha
        self.SupConLoss = SupConLoss()
        self.CE_loss = nn.CrossEntropyLoss()

    def forward(self, logits, targets):
        # 混合Cross Entropy和利用标签进行有监督的对比学习分类
        
        pred, feat = logits

        if len(feat.shape)<3:
            feat_sup_con = feat.unsqueeze(1)
        # pred = F.log_softmax(pred,dim=-1)
        loss = (self.alpha*self.SupConLoss(feat_sup_con, labels=targets) + (1-self.alpha)* self.CE_loss(pred, targets))
        
        return loss

class combined_loss(nn.Module):
    # not finished
    def __init__(self, alpha=0.9, beta=0, num_classes=None, device=None, dict_feat_size=None):
        super().__init__()
        # self.cls_loss = classic_loss
        self.alpha = alpha
        self.SupConLoss = SupConLoss()
        self.CE_loss = nn.CrossEntropyLoss()
        # self.CE_loss = LabelDictionaryAndCrossEntropy(num_classes=num_classes, device=device, dict_feat_size=dict_feat_size)
        # self.CE_loss = LabelSmoothingCrossEntropy(epsilon=beta)

    def forward(self, logits, targets):
        # 混合Cross Entropy和利用标签进行有监督的对比学习分类
        
        pred, feat = logits

        if len(feat.shape)<3:
            sup_feat = sup_feat.unsqueeze(1)
        # pred = F.log_softmax(pred,dim=-1)
        loss = self.alpha*self.SupConLoss(sup_feat, labels=targets) + (1-self.alpha)* self.CE_loss(pred, targets) # (1-self.alpha)* self.CE_loss(pred, targets)
        
        return loss
    

# old SupWMA SupConLoss
# class SupConLoss(nn.Module):
#     """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
#     It also supports the unsupervised contrastive loss in SimCLR"""
#     def __init__(self, temperature=0.07, base_temperature=0.07):
#         super(SupConLoss, self).__init__()
#         self.temperature = temperature
#         self.base_temperature = base_temperature   # Control the loss scalar, does not influence training

#     def forward(self, features, labels=None, mask=None):
#         """Compute loss for model. If both `labels` and `mask` are None,
#         it degenerates to SimCLR unsupervised loss: https://arxiv.org/pdf/2002.05709.pdf

#         Args:
#             features: hidden vector of shape [bsz, n_views, ...].
#             labels: ground truth of shape [bsz].
#             mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
#                 has the same class as sample i. Can be asymmetric.
#         Returns:
#             A loss scalar.
#         """
#         device = (torch.device('cuda') if features.is_cuda else torch.device('cpu'))

#         if len(features.shape) > 3:
#             features = features.view(features.shape[0], features.shape[1], -1)

#         batch_size = features.shape[0]

#         if labels is not None and mask is not None:
#             raise ValueError('Cannot define both `labels` and `mask`')
#         elif labels is None and mask is None:
#             mask = torch.eye(batch_size, dtype=torch.float32).to(device)
#         elif labels is not None:
#             labels = labels.contiguous().view(-1, 1)
#             if labels.shape[0] != batch_size:
#                 raise ValueError('Num of labels does not match num of features')
#             mask = torch.eq(labels, labels.T).float().to(device)
#         else:
#             mask = mask.float().to(device)

#         contrast_count = features.shape[1]
#         contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

#         anchor_feature = contrast_feature
#         anchor_count = contrast_count

#         # compute logits
#         anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), self.temperature)

#         # for numerical stability
#         logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
#         logits = anchor_dot_contrast - logits_max.detach() #这个可以除法约掉，所以没关系

#         # tile mask
#         mask = mask.repeat(anchor_count, contrast_count)
#         # mask-out self-contrast cases
#         logits_mask = torch.scatter(
#             torch.ones_like(mask),
#             1,
#             torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
#             0
#         )
#         # mask = mask * logits_mask # 这个SupWMA漏了

#         # compute log_prob
#         exp_logits = torch.exp(logits) * logits_mask

#         log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        
#         # new save
#         # compute mean of log-likelihood over positive
#         # modified to handle edge cases when there is no positive pair
#         # for an anchor point. 
#         # Edge case e.g.:- 
#         # features of shape: [4,1,...]
#         # labels:            [0,1,1,2]
#         # loss before mean:  [nan, ..., ..., nan] 
#         mask_pos_pairs = mask.sum(1)
#         mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
#         mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs
#         # # compute mean of log-likelihood over positive
#         # mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

#         # loss
#         loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
#         loss = loss.view(anchor_count, batch_size).mean()

#         return loss



"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""

# SupCon github version
class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # modified to handle edge cases when there is no positive pair
        # for an anchor point. 
        # Edge case e.g.:- 
        # features of shape: [4,1,...]
        # labels:            [0,1,1,2]
        # loss before mean:  [nan, ..., ..., nan] 
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss