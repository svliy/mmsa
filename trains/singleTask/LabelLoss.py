import torch
import torch.nn as nn
import numpy as np


class LabelLoss(nn.Module):
    def __init__(self):
        super(LabelLoss, self).__init__()

    def compute_cosine(self, x, y):
        """
        input:
            x: [2304, 100]
            x_norm: [2304]
            y: [2304, 100]
            y_norm: [2304]
        output:
            cosine: [2304]
        """
        # x = self.compute_compact_s(x)
        # y = self.compute_compact_s(y)
        x_norm = torch.sqrt(torch.sum(torch.pow(x, 2), 1)+1e-8)
        x_norm = torch.max(x_norm, 1e-8*torch.ones_like(x_norm))
        y_norm = torch.sqrt(torch.sum(torch.pow(y, 2), 1)+1e-8)
        y_norm = torch.max(y_norm, 1e-8*torch.ones_like(y_norm))
        cosine = torch.sum(x * y, 1) / (x_norm * y_norm)
        return cosine

    def forward(self, ids, feats, temperature=None):
        """
        input:
            ids: [48, 1]
            feats: [48, 100]
            temperature: 温度系数，初始值为 0.07
        """

        # [B=48, D=100]
        B, F = feats.shape

        # [B=48, D=100] -> [B=48, D=4800] -> [B=48^2, D=100]
        s = feats.repeat(1, B).view(-1, F) # B**2 X F
        # [B=48, D=1] -> [B=48, D=48]
        s_ids = ids.view(B, 1).repeat(1, B) # B X B
        # s_ids = torch.round(s_ids)
        
        # [B=48, D=100] -> [B=48^2, D=100]
        t = feats.repeat(B, 1) # B**2 X F
        # [B=48, D=1] -> [B=48, D=48]
        t_ids = ids.view(1, B).repeat(B, 1) # B X B 
        # t_ids = torch.round(t_ids)
        
        # [B = 48^2 = 2304]
        cosine = self.compute_cosine(s, t) # B**2
        equal_mask = torch.eye(B, dtype=torch.bool) # B X B
        # 去除对角线元素
        s_ids = s_ids[~equal_mask].view(B, B-1) # B X (B-1)
        t_ids = t_ids[~equal_mask].view(B, B-1) # B X (B-1)
        cosine = cosine.view(B, B)[~equal_mask].view(B, B-1) # B X (B-1)
        cosine = cosine * temperature

        # 将不同的标签值映射为 [-3, -2, -1, 0, 1, 2, 3] 中的 1 个
        s_ids = torch.round(s_ids)
        t_ids = torch.round(t_ids)
        sim_mask = (s_ids == t_ids) # B X (B-1)
        
        sim_cos = cosine[sim_mask] # 正样本
        dif_cos = cosine[~sim_mask] # 负样本

        # print("The positive cosine: ", sim_cos.shape)
        # print("The negative cosine: ", dif_cos.shape)

        feature = torch.cat([sim_cos, dif_cos], dim=0)
        pos_num = len(sim_cos)
        feature = feature.repeat(pos_num, 1)
        target = torch.arange(pos_num, device=sim_cos.device, dtype=torch.long)

        loss = nn.functional.cross_entropy(feature, target)
        
        # return loss, pos_table.detach().cpu().numpy(), neg_table.detach().cpu().numpy(), pos_label.detach().cpu().numpy(), neg_label.detach().cpu().numpy()
        return loss