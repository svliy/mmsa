import torch
import torch.nn as nn


class HingeLoss(nn.Module):
    def __init__(self):
        super(HingeLoss, self).__init__()

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

    def forward(self, ids, feats, margin=0.1):
        """
        input:
            ids: [48, 1]
            feats: [48, 100]
        """
        # [B=48, D=100]
        B, F = feats.shape

        # [B=48, D=100] -> [B=48, D=4800] -> [B=48^2, D=100]
        s = feats.repeat(1, B).view(-1, F) # B**2 X F
        # [B=48, D=1] -> [B=48, D=48]
        s_ids = ids.view(B, 1).repeat(1, B) # B X B
        
        # [B=48, D=100] -> [B=48^2, D=100]
        t = feats.repeat(B, 1) # B**2 X F
        # [B=48, D=1] -> [B=48, D=48]
        t_ids = ids.view(1, B).repeat(B, 1) # B X B 
        
        # [B = 48^2 = 2304]
        cosine = self.compute_cosine(s, t) # B**2
        equal_mask = torch.eye(B, dtype=torch.bool) # B X B
        # 去除对角线元素
        s_ids = s_ids[~equal_mask].view(B, B-1) # B X (B-1)
        t_ids = t_ids[~equal_mask].view(B, B-1) # B X (B-1)
        cosine = cosine.view(B, B)[~equal_mask].view(B, B-1) # B X (B-1)

        # 将不同的标签值映射为 [-3, -2, -1, 0, 1, 2, 3] 中的 1 个
        # s_ids = torch.round(s_ids)
        # t_ids = torch.round(t_ids)
        sim_mask = (s_ids == t_ids) # B X (B-1)
        margin = 0.15 * abs(s_ids - t_ids)#[~sim_mask].view(B, B - 3)

        loss = 0
        loss_num = 0
        
        for i in range(B):
            sim_num = sum(sim_mask[i])
            dif_num = B - 1 - sim_num
            if not sim_num or not dif_num:
                continue
            sim_cos = cosine[i, sim_mask[i]].reshape(-1, 1).repeat(1, dif_num)
            dif_cos = cosine[i, ~sim_mask[i]].reshape(-1, 1).repeat(1, sim_num).transpose(0, 1)
            t_margin = margin[i, ~sim_mask[i]].reshape(-1, 1).repeat(1, sim_num).transpose(0, 1)

            loss_i = torch.max(torch.zeros_like(sim_cos), t_margin - sim_cos + dif_cos).mean()
            loss += loss_i
            loss_num += 1

        if loss_num == 0:
            loss_num = 1

        loss = loss / loss_num
        return loss