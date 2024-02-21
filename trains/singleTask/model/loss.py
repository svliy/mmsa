import sys

sys.path.append('/workspace/projects/mmsa/trains/singleTask/model')

import torch
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
import torch.nn as nn
import linklink as link

class ClipInfoCELoss(_Loss):
    # def __init__(self, partition_num):
    def __init__(self):
        super(ClipInfoCELoss, self).__init__()
        # self.partition_num = partition_num

    # def forward(self, logits_per_image, logits_per_text, batch):
    #def forward(self, logits):
    #    labels = torch.arange(len(logits)).cuda()
    #    loss_i = F.cross_entropy(logits, labels)
    #    loss_t = F.cross_entropy(logits.t(), labels)
    #    loss = (loss_i+loss_t)/2
    #    return loss
    def forward(self, logits_per_image, logits_per_text):
        bs, l_bs = logits_per_image.shape
        if l_bs == bs:
            labels = torch.arange(len(logits_per_image)).cuda()
        else:
            labels = link.get_rank() * bs + torch.arange(0, bs, dtype=torch.long).cuda()

        loss_i = F.cross_entropy(logits_per_image, labels)
        loss_t = F.cross_entropy(logits_per_text, labels)
        loss = (loss_i+loss_t)/2
        return loss