"""
here is the mian backbone for DMD containing feature decoupling and multimodal transformers
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .clip_fdt import Clip_FDT
from ...subNets import BertTextEncoder
from ...subNets.transformers_encoder.transformer import TransformerEncoder

class DMD(nn.Module):
    def __init__(self, args):
        super(DMD, self).__init__()
        if args.use_bert:
            self.text_model = BertTextEncoder(use_finetune=args.use_finetune,
                                              transformers=args.transformers,
                                              pretrained=args.pretrained)
        self.use_bert = args.use_bert
        # what is dst_feature_dims and nheads? [50, 10]
        # dst_feature_dims: 50, nheads: 10
        dst_feature_dims, nheads = args.dst_feature_dim_nheads
        # l: 语言特征，v: 视觉特征，a: 音频特征
        if args.dataset_name == 'mosi':
            if args.need_data_aligned:
                self.len_l, self.len_v, self.len_a = 50, 50, 50
            else:
                self.len_l, self.len_v, self.len_a = 50, 500, 375
        if args.dataset_name == 'mosei':
            if args.need_data_aligned:
                self.len_l, self.len_v, self.len_a = 50, 50, 50
            else:
                self.len_l, self.len_v, self.len_a = 50, 500, 500
        if args.dataset_name == 'sims':
            self.len_l, self.len_v, self.len_a = 50, 232, 925

        # sims: [768, 25, 177]
        # 语言，声音，图像的特征维度
        # 语言：(batch_size, 50, 768)
        # 声音：(batch_size, 925, 25)
        # 图像：(batch_size, 232, 177)
        self.orig_d_l, self.orig_d_a, self.orig_d_v = args.feature_dims

        # d_l: 语言特征，d_a: 音频特征，d_v: 视觉特征
        # 都是50
        self.d_l = self.d_a = self.d_v = dst_feature_dims
        self.num_heads = nheads
        self.layers = args.nlevels
        self.attn_dropout = args.attn_dropout
        self.attn_dropout_a = args.attn_dropout_a
        self.attn_dropout_v = args.attn_dropout_v
        self.relu_dropout = args.relu_dropout
        self.embed_dropout = args.embed_dropout
        self.res_dropout = args.res_dropout
        self.output_dropout = args.output_dropout
        self.text_dropout = args.text_dropout
        self.attn_mask = args.attn_mask
        combined_dim_low = self.d_a
        combined_dim_high = 2 * self.d_a
        combined_dim = 2 * (self.d_l + self.d_a + self.d_v) + self.d_l * 3
        output_dim = 1
        # fdt
        self.sd_num = args.sd_num
        self.sd_dim = self.d_l
        self.att_func_type = args.att_func_type
        self.sd_temperature = args.sd_temperature # softmax or sparsemax 温度系数
        self.pool_type = args.pool_type

        # 1. Temporal convolutional layers for initial feature
        self.proj_l = nn.Conv1d(self.orig_d_l, self.d_l, kernel_size=args.conv1d_kernel_size_l, padding=0, bias=False)
        self.proj_a = nn.Conv1d(self.orig_d_a, self.d_a, kernel_size=args.conv1d_kernel_size_a, padding=0, bias=False)
        self.proj_v = nn.Conv1d(self.orig_d_v, self.d_v, kernel_size=args.conv1d_kernel_size_v, padding=0, bias=False)

        # 2.1 Modality-specific encoder
        # 私有编码器
        self.encoder_s_l = nn.Conv1d(self.d_l, self.d_l, kernel_size=1, padding=0, bias=False)
        self.encoder_s_v = nn.Conv1d(self.d_v, self.d_v, kernel_size=1, padding=0, bias=False)
        self.encoder_s_a = nn.Conv1d(self.d_a, self.d_a, kernel_size=1, padding=0, bias=False)

        # 2.2 Modality-invariant encoder
        # 共享编码器
        self.encoder_c = nn.Conv1d(self.d_l, self.d_l, kernel_size=1, padding=0, bias=False)

        # 3. Decoder for reconstruct three modalities
        # 私有解码器产生耦合特征
        self.decoder_l = nn.Conv1d(self.d_l * 2, self.d_l, kernel_size=1, padding=0, bias=False)
        self.decoder_v = nn.Conv1d(self.d_v * 2, self.d_v, kernel_size=1, padding=0, bias=False)
        self.decoder_a = nn.Conv1d(self.d_a * 2, self.d_a, kernel_size=1, padding=0, bias=False)

        # for calculate cosine sim between s_x
        # 输入：目标特征维度 * 序列长度
        # 输出：目标特征维度 50
        self.proj_cosine_l = nn.Linear(combined_dim_low * (self.len_l - args.conv1d_kernel_size_l + 1), combined_dim_low)
        self.proj_cosine_v = nn.Linear(combined_dim_low * (self.len_v - args.conv1d_kernel_size_v + 1), combined_dim_low)
        self.proj_cosine_a = nn.Linear(combined_dim_low * (self.len_a - args.conv1d_kernel_size_a + 1), combined_dim_low)

        # for align c_l, c_v, c_a
        self.align_c_l = nn.Linear(combined_dim_low * (self.len_l - args.conv1d_kernel_size_l + 1), combined_dim_low)
        self.align_c_v = nn.Linear(combined_dim_low * (self.len_v - args.conv1d_kernel_size_v + 1), combined_dim_low)
        self.align_c_a = nn.Linear(combined_dim_low * (self.len_a - args.conv1d_kernel_size_a + 1), combined_dim_low)

        self.self_attentions_c_l = self.get_network(self_type='l')
        self.self_attentions_c_v = self.get_network(self_type='v')
        self.self_attentions_c_a = self.get_network(self_type='a')

        self.proj1_c = nn.Linear(self.d_l * 3, self.d_l * 3)
        self.proj2_c = nn.Linear(self.d_l * 3, self.d_l * 3)
        self.out_layer_c = nn.Linear(self.d_l * 3, output_dim)

        # 4.2 Crossmodal Attentions
        self.trans_l_with_a = self.get_network(self_type='la')
        self.trans_l_with_v = self.get_network(self_type='lv')
        self.trans_a_with_l = self.get_network(self_type='al')
        self.trans_a_with_v = self.get_network(self_type='av')
        self.trans_v_with_l = self.get_network(self_type='vl')
        self.trans_v_with_a = self.get_network(self_type='va')
        self.trans_l_mem = self.get_network(self_type='l_mem', layers=3)
        self.trans_a_mem = self.get_network(self_type='a_mem', layers=3)
        self.trans_v_mem = self.get_network(self_type='v_mem', layers=3)

        # 4. fc layers for homogeneous graph distillation
        self.proj1_l_low = nn.Linear(combined_dim_low * (self.len_l - args.conv1d_kernel_size_l + 1), combined_dim_low)
        self.proj2_l_low = nn.Linear(combined_dim_low, combined_dim_low * (self.len_l - args.conv1d_kernel_size_l + 1))
        self.out_layer_l_low = nn.Linear(combined_dim_low * (self.len_l - args.conv1d_kernel_size_l + 1), output_dim)
        self.proj1_v_low = nn.Linear(combined_dim_low * (self.len_v - args.conv1d_kernel_size_v + 1), combined_dim_low)
        self.proj2_v_low = nn.Linear(combined_dim_low, combined_dim_low * (self.len_v - args.conv1d_kernel_size_v + 1))
        self.out_layer_v_low = nn.Linear(combined_dim_low * (self.len_v - args.conv1d_kernel_size_v + 1), output_dim)
        self.proj1_a_low = nn.Linear(combined_dim_low * (self.len_a - args.conv1d_kernel_size_a + 1), combined_dim_low)
        self.proj2_a_low = nn.Linear(combined_dim_low, combined_dim_low * (self.len_a - args.conv1d_kernel_size_a + 1))
        self.out_layer_a_low = nn.Linear(combined_dim_low * (self.len_a - args.conv1d_kernel_size_a + 1), output_dim)
        # based on sd
        self.proj1_l_low_sd = nn.Linear(self.sd_dim, combined_dim_low)
        self.proj2_l_low_sd = nn.Linear(combined_dim_low, self.sd_dim)
        self.out_layer_l_low_sd = nn.Linear(self.sd_dim, output_dim)
        self.proj1_v_low_sd = nn.Linear(self.sd_dim, combined_dim_low)
        self.proj2_v_low_sd = nn.Linear(combined_dim_low, self.sd_dim)
        self.out_layer_v_low_sd = nn.Linear(self.sd_dim, output_dim)
        self.proj1_a_low_sd = nn.Linear(self.sd_dim, combined_dim_low)
        self.proj2_a_low_sd = nn.Linear(combined_dim_low, self.sd_dim)
        self.out_layer_a_low_sd = nn.Linear(self.sd_dim, output_dim)

        # 5. fc layers for heterogeneous graph distillation
        self.proj1_l_high = nn.Linear(combined_dim_high, combined_dim_high)
        self.proj2_l_high = nn.Linear(combined_dim_high, combined_dim_high)
        self.out_layer_l_high = nn.Linear(combined_dim_high, output_dim)
        self.proj1_v_high = nn.Linear(combined_dim_high, combined_dim_high)
        self.proj2_v_high = nn.Linear(combined_dim_high, combined_dim_high)
        self.out_layer_v_high = nn.Linear(combined_dim_high, output_dim)
        self.proj1_a_high = nn.Linear(combined_dim_high, combined_dim_high)
        self.proj2_a_high = nn.Linear(combined_dim_high, combined_dim_high)
        self.out_layer_a_high = nn.Linear(combined_dim_high, output_dim)
        # based on sd
        self.proj1_l_high_sd = nn.Linear(self.sd_dim, self.sd_dim)
        self.proj2_l_high_sd = nn.Linear(self.sd_dim, self.sd_dim)
        self.out_layer_l_high_sd = nn.Linear(self.sd_dim, output_dim)
        self.proj1_v_high_sd = nn.Linear(self.sd_dim, self.sd_dim)
        self.proj2_v_high_sd = nn.Linear(self.sd_dim, self.sd_dim)
        self.out_layer_v_high_sd = nn.Linear(self.sd_dim, output_dim)
        self.proj1_a_high_sd = nn.Linear(self.sd_dim, self.sd_dim)
        self.proj2_a_high_sd = nn.Linear(self.sd_dim, self.sd_dim)
        self.out_layer_a_high_sd = nn.Linear(self.sd_dim, output_dim)

        # 6. Ensemble Projection layers
        # decrease modality dim
        self.proj_l_final = nn.Linear(4 * self.d_l, 2 * self.d_l)
        self.proj_v_final = nn.Linear(4 * self.d_l, 2 * self.d_l)
        self.proj_a_final = nn.Linear(4 * self.d_l, 2 * self.d_l)

        # weight for each modality
        self.weight_l = nn.Linear(2 * self.d_l, 2 * self.d_l)
        self.weight_v = nn.Linear(2 * self.d_v, 2 * self.d_v)
        self.weight_a = nn.Linear(2 * self.d_a, 2 * self.d_a)
        self.weight_c = nn.Linear(3 * self.d_l, 3 * self.d_l)
        # final project
        combined_dim = 300
        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, output_dim)

        # 7. CLIP DICTIONARY
        # self.c_clip_fdt = Clip_FDT(
        #     image_encode=None,
        #     text_encode=None,
        #     use_allgather=None,
        #     sd_num=self.sd_num,
        #     sd_dim=self.sd_dim,
        #     raw_img_ft_dim=self.sd_dim,
        #     raw_txt_ft_dim=self.sd_dim,
        #     raw_acoustic_ft_dim=self.sd_dim,
        #     att_func_type=self.att_func_type,
        #     pool_type=self.pool_type,
        #     sd_temperature=self.sd_temperature
        # )
        self.hetero_dict = Clip_FDT(
            image_encode=None,
            text_encode=None,
            use_allgather=None,
            sd_num=self.sd_num,
            sd_dim=self.sd_dim*2,
            raw_img_ft_dim=self.sd_dim*2,
            raw_txt_ft_dim=self.sd_dim*2,
            raw_acoustic_ft_dim=self.sd_dim*2,
            att_func_type=self.att_func_type,
            pool_type=self.pool_type,
            sd_temperature=self.sd_temperature
        )

    def get_network(self, self_type='l', layers=-1):
        if self_type in ['l', 'al', 'vl']:
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
        elif self_type in ['a', 'la', 'va']:
            embed_dim, attn_dropout = self.d_a, self.attn_dropout_a
        elif self_type in ['v', 'lv', 'av']:
            embed_dim, attn_dropout = self.d_v, self.attn_dropout_v
        elif self_type == 'l_mem':
            embed_dim, attn_dropout = 2 * self.d_l, self.attn_dropout
        elif self_type == 'a_mem':
            embed_dim, attn_dropout = 2 * self.d_a, self.attn_dropout
        elif self_type == 'v_mem':
            embed_dim, attn_dropout = 2 * self.d_v, self.attn_dropout
        else:
            raise ValueError("Unknown network type")

        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=max(self.layers, layers),
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask)

    def forward(self, text, audio, video, is_distill=False):
        # print("Before bert", text.size(), audio.size(), video.size())
        # Before bert torch.Size([16, 3, 50]) torch.Size([16, 925, 25]) torch.Size([16, 232, 177])
        if self.use_bert:
            text = self.text_model(text)
        # After bert torch.Size([16, 50, 768]) torch.Size([16, 925, 25]) torch.Size([16, 232, 177])
        
        # (B, L, D) --> (B, D, L) [B, C, H, W]
        # x_l: (16, 768, 50) x_a: (16, 25, 925) x_v: (16, 177, 232)
        # 改变维度
        x_l = F.dropout(text.transpose(1, 2), p=self.text_dropout, training=self.training)
        x_a = audio.transpose(1, 2)
        x_v = video.transpose(1, 2)

        # print("After proj", proj_x_l.size(), proj_x_a.size(), proj_x_v.size())
        # After proj torch.Size([16, 50, 46]) torch.Size([16, 50, 925]) torch.Size([16, 50, 232])
        # 一维时空卷积：对齐特征维度 50，保留序列长度
        proj_x_l = x_l if self.orig_d_l == self.d_l else self.proj_l(x_l)
        proj_x_a = x_a if self.orig_d_a == self.d_a else self.proj_a(x_a)
        proj_x_v = x_v if self.orig_d_v == self.d_v else self.proj_v(x_v)

        # private feature
        # private feature torch.Size([16, 50, 46]) torch.Size([16, 50, 925]) torch.Size([16, 50, 232])
        s_l = self.encoder_s_l(proj_x_l)
        s_a = self.encoder_s_a(proj_x_a)
        s_v = self.encoder_s_v(proj_x_v)

        # print("private feature", s_l.size(), s_a.size(), s_v.size())

        # common feature
        # Batch_size, feature_dim, seq_len
        # common feature torch.Size([16, 50, 46]) torch.Size([16, 50, 925]) torch.Size([16, 50, 232])
        # c_l = self.encoder_c(proj_x_l)
        # c_a = self.encoder_c(proj_x_a)
        # c_v = self.encoder_c(proj_x_v)
        # # print("common feature", c_l.size(), c_a.size(), c_v.size())
        # c_list = [c_l, c_v, c_a]

        # c_l_sim torch.Size([16, 50]) torch.Size([16, 50]) torch.Size([16, 50])
        # 一个样本由一个维度为 50 的向量组成。
        # input:(16, 50 x 46) ---> (16, 50)
        # 压缩了序列长度
        # c_l_sim = self.align_c_l(c_l.contiguous().view(x_l.size(0), -1))
        # c_a_sim = self.align_c_a(c_a.contiguous().view(x_a.size(0), -1))
        # c_v_sim = self.align_c_v(c_v.contiguous().view(x_v.size(0), -1))

        # print("c_l_sim", c_l_sim.size(), c_a_sim.size(), c_v_sim.size())
       
        # 私有解码器产生耦合特征，堆叠特征
        # torch.Size([16, 50, 46]) torch.Size([16, 50, 925]) torch.Size([16, 50, 232])
        # recon_l = self.decoder_l(torch.cat([s_l, c_list[0]], dim=1))
        # recon_a = self.decoder_a(torch.cat([s_a, c_list[2]], dim=1))
        # recon_v = self.decoder_v(torch.cat([s_v, c_list[1]], dim=1))

        # print("recon_l", recon_l.size(), recon_a.size(), recon_v.size())

        # 私有编码器重新编码耦合特征以回归异质特征
        # s_l_r: torch.Size([16, 50, 46]) s_a_r: torch.Size([16, 50, 925]) s_v_r: torch.Size([16, 50, 232])
        # s_l_r = self.encoder_s_l(recon_l)
        # s_a_r = self.encoder_s_a(recon_a)
        # s_v_r = self.encoder_s_v(recon_v)

        # s_l: torch.Size([46, 16, 50]) s_a: torch.Size([925, 16, 50]) s_v: torch.Size([232, 16, 50])
        s_l = s_l.permute(2, 0, 1)
        s_a = s_a.permute(2, 0, 1)
        s_v = s_v.permute(2, 0, 1)
        
        # c_l: torch.Size([46, 16, 50]) c_a: torch.Size([925, 16, 50]) c_v: torch.Size([232, 16, 50])
        # c_l = c_l.permute(2, 0, 1)
        # c_a = c_a.permute(2, 0, 1)
        # c_v = c_v.permute(2, 0, 1)

        # build featurn based sd(dictionary)
        # logits_c_sd = self.c_clip_fdt(c_l.permute(1, 0, 2).contiguous(), c_v.permute(1, 0, 2).contiguous(), c_a.permute(1, 0, 2).contiguous())

        # hs_l_low: (16, 46 x 50)
        # hs_l_low = c_l.transpose(0, 1).contiguous().view(x_l.size(0), -1)
        # # repr_l_low: (16, 50)
        # # f
        # repr_l_low = self.proj1_l_low(hs_l_low)
        # # hs_proj_l_low: (16, 46 x 50)
        # hs_proj_l_low = self.proj2_l_low(
        #     F.dropout(F.relu(repr_l_low, inplace=True), p=self.output_dropout, training=self.training))
        # # hs_proj_l_low: (16, 46 x 50) + (16, 46 x 50)
        # hs_proj_l_low += hs_l_low
        # # logits_l_low: (16, 1)
        # logits_l_low = self.out_layer_l_low(hs_proj_l_low)
        
        # based on sd
        # hs_l_low = c_l_sd.contiguous()
        # repr_l_low = self.proj1_l_low_sd(hs_l_low)
        # hs_proj_l_low = self.proj2_l_low_sd(
        #     F.dropout(F.relu(repr_l_low, inplace=True),
        #               p=self.output_dropout,
        #               training=self.training)
        # )
        # hs_proj_l_low += hs_l_low
        # logits_l_low = self.out_layer_l_low_sd(hs_proj_l_low)

        # hs_v_low = c_v.transpose(0, 1).contiguous().view(x_v.size(0), -1)
        # repr_v_low = self.proj1_v_low(hs_v_low)
        # hs_proj_v_low = self.proj2_v_low(
        #     F.dropout(F.relu(repr_v_low, inplace=True), p=self.output_dropout, training=self.training))
        # hs_proj_v_low += hs_v_low
        # logits_v_low = self.out_layer_v_low(hs_proj_v_low)
        # based on sd
        # hs_v_low = c_v_sd.contiguous()
        # repr_v_low = self.proj1_v_low_sd(hs_v_low)
        # hs_proj_v_low = self.proj2_v_low_sd(
        #     F.dropout(F.relu(repr_v_low, inplace=True),
        #               p=self.output_dropout,
        #               training=self.training)
        # )
        # hs_proj_v_low += hs_v_low
        # logits_v_low = self.out_layer_v_low_sd(hs_proj_v_low)

        # hs_a_low = c_a.transpose(0, 1).contiguous().view(x_a.size(0), -1)
        # repr_a_low = self.proj1_a_low(hs_a_low)
        # hs_proj_a_low = self.proj2_a_low(
        #     F.dropout(F.relu(repr_a_low, inplace=True), p=self.output_dropout, training=self.training))
        # hs_proj_a_low += hs_a_low
        # logits_a_low = self.out_layer_a_low(hs_proj_a_low)
        # based on sd
        # hs_a_low = c_a_sd.contiguous()
        # repr_a_low = self.proj1_a_low_sd(hs_a_low)
        # hs_proj_a_low = self.proj2_a_low_sd(
        #     F.dropout(F.relu(repr_a_low, inplace=True),
        #               p=self.output_dropout,
        #               training=self.training)
        # )
        # hs_proj_a_low += hs_a_low
        # logits_a_low = self.out_layer_a_low_sd(hs_proj_a_low)

        # s_l: torch.Size([46, 16, 50]) s_a: torch.Size([925, 16, 50]) s_v: torch.Size([232, 16, 50])
        # ---> torch.Size([16, 46 x 50]) torch.Size([16, 925 x 50]) torch.Size([16, 232 x 50])
        # output size: (16, 50)
        # proj_s_l = self.proj_cosine_l(s_l.transpose(0, 1).contiguous().view(x_l.size(0), -1))
        # proj_s_v = self.proj_cosine_v(s_v.transpose(0, 1).contiguous().view(x_l.size(0), -1))
        # proj_s_a = self.proj_cosine_a(s_a.transpose(0, 1).contiguous().view(x_l.size(0), -1))

        # c_l_att = self.self_attentions_c_l(c_l)
        # if type(c_l_att) == tuple:
        #     c_l_att = c_l_att[0]
        # c_l_att = c_l_att[-1]
        # c_v_att = self.self_attentions_c_v(c_v)
        # if type(c_v_att) == tuple:
        #     c_v_att = c_v_att[0]
        # c_v_att = c_v_att[-1]
        # c_a_att = self.self_attentions_c_a(c_a)
        # if type(c_a_att) == tuple:
        #     c_a_att = c_a_att[0]
        # c_a_att = c_a_att[-1]
        # c_fusion = torch.cat([c_l_att, c_v_att, c_a_att], dim=1)

        # c_proj = self.proj2_c(
        #     F.dropout(F.relu(self.proj1_c(c_fusion), inplace=True),
        #               p=self.output_dropout,
        #               training=self.training
        #     )
        # )
        # c_proj += c_fusion
        # # (16, 1)
        # logits_c = self.out_layer_c(c_proj)

        # cross-modal attention
        # (V,A) --> L
        h_l_with_as = self.trans_l_with_a(s_l, s_a, s_a)  # Dimension (L, N, d_l)
        h_l_with_vs = self.trans_l_with_v(s_l, s_v, s_v)  # Dimension (L, N, d_l)
        h_ls = torch.cat([h_l_with_as, h_l_with_vs], dim=2)
        h_ls = self.trans_l_mem(h_ls)
        if type(h_ls) == tuple:
            h_ls = h_ls[0]
        last_h_l = last_hs = h_ls[-1]  # Take the last output for prediction

        # (L,V) --> A
        h_a_with_ls = self.trans_a_with_l(s_a, s_l, s_l)
        h_a_with_vs = self.trans_a_with_v(s_a, s_v, s_v)
        h_as = torch.cat([h_a_with_ls, h_a_with_vs], dim=2)
        h_as = self.trans_a_mem(h_as)
        if type(h_as) == tuple:
            h_as = h_as[0]
        last_h_a = last_hs = h_as[-1]

        # (L,A) --> V
        h_v_with_ls = self.trans_v_with_l(s_v, s_l, s_l)
        h_v_with_as = self.trans_v_with_a(s_v, s_a, s_a)
        h_vs = torch.cat([h_v_with_ls, h_v_with_as], dim=2)
        h_vs = self.trans_v_mem(h_vs)
        if type(h_vs) == tuple:
            h_vs = h_vs[0]
        last_h_v = last_hs = h_vs[-1]
        
        # torch.Size([46, 1, 100]) torch.Size([496, 1, 100]) torch.Size([371, 1, 100])
        s_l_dict, s_v_dict, s_a_dict, logit_scale, att_weight, codebook = self.hetero_dict(h_ls.permute(1, 0, 2).detach(), h_vs.permute(1, 0, 2).detach(), h_as.permute(1, 0, 2).detach())
        
        # torch.Size([1, 512])
        l_att_weight, v_att_weight, a_att_weight = att_weight
        # print(l_att_weight.shape, v_att_weight.shape, a_att_weight.shape)
        
        # print(lva_att_weight.shape)
        # l_att_weight = torch.nonzero(torch.squeeze(l_att_weight, dim=0)).view(1, -1)
        # v_att_weight = torch.nonzero(torch.squeeze(v_att_weight, dim=0)).view(1, -1)
        # a_att_weight = torch.nonzero(torch.squeeze(a_att_weight, dim=0)).view(1, -1)
        lva_att_weight = torch.cat((l_att_weight, v_att_weight, a_att_weight), dim=0)
        # print(lva_att_weight.shape)
        
        # print(lva_att_weight.shape)
        # # 计算需要填充的长度
        # max_len = max(l_att_weight.size(1), v_att_weight.size(1), a_att_weight.size(1))

        # # 使用torch.nn.functional.pad对较短的张量进行0填充
        # if l_att_weight.size(1) < max_len:
        #     l_att_weight = torch.nn.functional.pad(l_att_weight, pad=(0, max_len - l_att_weight.size(1)), mode='constant', value=0)
        # if v_att_weight.size(1) < max_len:
        #     v_att_weight = torch.nn.functional.pad(v_att_weight, pad=(0, max_len - v_att_weight.size(1)), mode='constant', value=0)
        # if a_att_weight.size(1) < max_len:
        #     a_att_weight = torch.nn.functional.pad(a_att_weight, pad=(0, max_len - a_att_weight.size(1)), mode='constant', value=0)
        
        # print(l_att_weight.shape)
        # # print(torch.cat((l_att_weight, v_att_weight, a_att_weight), dim=0))
        # intersection_1_2 = np.intersect1d(l_att_weight.detach().cpu().numpy(), v_att_weight.detach().cpu().numpy())
        # intersection_1_2_3 = np.intersect1d(intersection_1_2, a_att_weight.detach().cpu().numpy())
        # print(intersection_1_2_3)

        # 拼接增强特征和字典重构特征        
        last_h_l =  torch.cat([last_h_l, s_l_dict], dim=1)
        last_h_v =  torch.cat([last_h_v, s_v_dict], dim=1)
        last_h_a =  torch.cat([last_h_a, s_a_dict], dim=1)

        last_h_l = self.proj_l_final(last_h_l)
        last_h_v = self.proj_v_final(last_h_v)
        last_h_a = self.proj_a_final(last_h_a)

        # hs_proj_l_high = self.proj2_l_high(
        #     F.dropout(F.relu(self.proj1_l_high(last_h_l), inplace=True),
        #               p=self.output_dropout,
        #               training=self.training
        #     )
        # )
        # hs_proj_l_high += last_h_l
        # # [16, 1]
        # logits_l_high = self.out_layer_l_high(hs_proj_l_high)

        # hs_proj_v_high = self.proj2_v_high(
        #     F.dropout(F.relu(self.proj1_v_high(last_h_v), inplace=True), p=self.output_dropout, training=self.training))
        # hs_proj_v_high += last_h_v
        # logits_v_high = self.out_layer_v_high(hs_proj_v_high)

        # hs_proj_a_high = self.proj2_a_high(
        #     F.dropout(F.relu(self.proj1_a_high(last_h_a), inplace=True), p=self.output_dropout,
        #               training=self.training))
        # hs_proj_a_high += last_h_a
        # logits_a_high = self.out_layer_a_high(hs_proj_a_high)

        last_h_l = torch.sigmoid(self.weight_l(last_h_l))
        last_h_v = torch.sigmoid(self.weight_v(last_h_v))
        last_h_a = torch.sigmoid(self.weight_a(last_h_a))

        # last_h_l = torch.relu(self.weight_l(last_h_l))
        # last_h_v = torch.relu(self.weight_v(last_h_v))
        # last_h_a = torch.relu(self.weight_a(last_h_a))

        # last_s_l_dict = torch.sigmoid(self.weight_l_dict(s_l_dict))
        # last_s_v_dict = torch.sigmoid(self.weight_v_dict(s_v_dict))
        # last_s_a_dict = torch.sigmoid(self.weight_a_dict(s_a_dict))

        # last_s_l_dict = self.weight_l_dict(s_l_dict)
        # last_s_v_dict = self.weight_v_dict(s_v_dict)
        # last_s_a_dict = self.weight_a_dict(s_a_dict)
        # c_fusion = torch.sigmoid(self.weight_c(c_fusion))

        last_hs = torch.cat([last_h_l, last_h_v, last_h_a], dim=1)
        last_hs_proj = self.proj2(
            F.dropout(F.relu(self.proj1(last_hs), inplace=True), p=self.output_dropout, training=self.training))
        last_hs_proj += last_hs

        output = self.out_layer(last_hs_proj)

        res = {
            'codebook': codebook,
            # 'l_att_weight': l_att_weight,
            # 'v_att_weight': v_att_weight,
            # 'a_att_weight': a_att_weight,
            'lva_att_weight': lva_att_weight,
            's_l_dict': s_l_dict,
            's_v_dict': s_v_dict,
            's_a_dict': s_a_dict,
            'logit_scale': logit_scale,
            # 'logits_l_homo': logits_l_low,
            # 'logits_v_homo': logits_v_low,
            # 'logits_a_homo': logits_a_low,
            # 'repr_l_homo': repr_l_low,
            # 'repr_v_homo': repr_v_low,
            # 'repr_a_homo': repr_a_low,
            'origin_l': proj_x_l,
            'origin_v': proj_x_v,
            'origin_a': proj_x_a,
            's_l': s_l,
            's_v': s_v,
            's_a': s_a,
            # 'proj_s_l': proj_s_l,
            # 'proj_s_v': proj_s_v,
            # 'proj_s_a': proj_s_a,
            # 'c_l': c_l,
            # 'c_v': c_v,
            # 'c_a': c_a,
            # 's_l_r': s_l_r,
            # 's_v_r': s_v_r,
            # 's_a_r': s_a_r,
            # 'recon_l': recon_l,
            # 'recon_v': recon_v,
            # 'recon_a': recon_a,
            # 'c_l_sim': c_l_sim,
            # 'c_v_sim': c_v_sim,
            # 'c_a_sim': c_a_sim,
            # 'logits_l_hetero': logits_l_high,
            # 'logits_v_hetero': logits_v_high,
            # 'logits_a_hetero': logits_a_high,
            # 'repr_l_hetero': hs_proj_l_high,
            # 'repr_v_hetero': hs_proj_v_high,
            # 'repr_a_hetero': hs_proj_a_high,
            'last_h_l': last_h_l,
            'last_h_v': last_h_v,
            'last_h_a': last_h_a,
            # 'h_ls': h_ls,
            # 'h_as': h_as,
            # 'h_vs': h_vs,
            # 'logits_c': logits_c,
            'output_logit': output
        }
        return res