import os
import sys

sys.path.append('/workspace/projects/mmsa/trains/singleTask')

import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from ..utils import MetricsTop, dict_to_str
from .HingeLoss import HingeLoss
from .LabelLoss import LabelLoss
from model.loss import ClipInfoCELoss
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger('MMSA')


def save_grad(name):
	# 返回hook函数 
    def hook(grad):
    # 简单粗暴的直接输出，存到grads中也可以
        # grads[name] = grad
        print(f"name={name}, grad={grad}")
    return hook

class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, pred, real):
        diffs = torch.add(real, -pred)
        n = torch.numel(diffs.data)
        mse = torch.sum(diffs.pow(2)) / n
        return mse

class DMD():
    def __init__(self, args):
        self.args = args
        self.criterion = nn.L1Loss()
        # self.criterion = MSE()
        self.cosine = nn.CosineEmbeddingLoss()
        self.metrics = MetricsTop(args.train_mode).getMetics(args.dataset_name)
        self.MSE = MSE()
        self.sim_loss = HingeLoss()
        # self.hetero_loss = LabelLoss()
        self.hetero_loss = HingeLoss()
        self.writer = SummaryWriter(comment = f'{args.dataset_name}')

    def do_train(self, model, dataloader, return_epoch_results=False):

        # 0: DMD model, 1: Homo GD, 2: Hetero GD

        # params = list(model[0].parameters())
        # print(type(model[0].named_parameters()))
        base_model = []
        hetero_dict_model = []

        # 分离出字典的参数
        for name, p in model[0].named_parameters():
            if "hetero_dict" in name:
                hetero_dict_model += [p]
            else:
                base_model += [p]

        optimizer = optim.Adam([{'params': base_model},
                                {'params': hetero_dict_model, 'lr': self.args.dictionary_lr}], 
                                lr=self.args.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, verbose=True, patience=self.args.patience)

        epochs, best_epoch = 0, 0
        if return_epoch_results:
            epoch_results = {
                'train': [],
                'valid': [],
                'test': []
            }
        min_or_max = 'min' if self.args.KeyEval in ['Loss'] else 'max'
        best_valid = 1e8 if min_or_max == 'min' else 0

        net = []
        net_dmd = model[0]
        # net_distill_homo = model[1]
        # net_distill_hetero = model[2]
        net.append(net_dmd)
        # net.append(net_distill_homo)
        # net.append(net_distill_hetero)
        model = net

        while True:
            epochs += 1
            y_pred, y_true = [], []
            for mod in model:
                mod.train()

            train_loss = 0.0
            loss_hetero = 0.0
            left_epochs = self.args.update_epochs
            
            # 打印学习率
            print("Epoch:{}  Lr:{:.2E}".format(epochs, optimizer.state_dict()['param_groups'][0]['lr']))
            print("Epoch:{}  Lr:{:.2E}".format(epochs, optimizer.state_dict()['param_groups'][1]['lr']))

            # 用于存储每次迭代时的字典
            dict_table = []

            with tqdm(dataloader['train']) as td:
                for batch_data in td:

                    if left_epochs == self.args.update_epochs:
                        optimizer.zero_grad()
                    left_epochs = left_epochs - 1
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    labels = batch_data['labels']['M'].to(self.args.device)
                    labels = labels.view(-1, 1)

                    logits_homo, reprs_homo, logits_hetero, reprs_hetero = [], [], [], []

                    output = model[0](text, audio, vision, is_distill=True)

                    # 保存每次迭代的字典
                    # print(output['codebook'].data[0])
                    dict_table.append(output['codebook'].data.detach().cpu().numpy())
                    
                    # # logits for homo GD
                    # # [3, 16, 1]
                    # logits_homo.append(output['logits_l_homo'])
                    # logits_homo.append(output['logits_v_homo'])
                    # logits_homo.append(output['logits_a_homo'])

                    # # reprs for homo GD
                    # reprs_homo.append(output['repr_l_homo'])
                    # reprs_homo.append(output['repr_v_homo'])
                    # reprs_homo.append(output['repr_a_homo'])

                    # # logits for hetero GD
                    # logits_hetero.append(output['logits_l_hetero'])
                    # logits_hetero.append(output['logits_v_hetero'])
                    # logits_hetero.append(output['logits_a_hetero'])

                    # # reprs for hetero GD
                    # reprs_hetero.append(output['repr_l_hetero'])
                    # reprs_hetero.append(output['repr_v_hetero'])
                    # reprs_hetero.append(output['repr_a_hetero'])

                    # logits_homo = torch.stack(logits_homo)
                    # reprs_homo = torch.stack(reprs_homo)

                    # logits_hetero = torch.stack(logits_hetero)
                    # reprs_hetero = torch.stack(reprs_hetero)

                    # edges for homo distill
                    # edges_homo, edges_origin_homo = model[1](logits_homo, reprs_homo)

                    # edges for hetero distill
                    # edges_hetero, edges_origin_hetero = model[2](logits_hetero, reprs_hetero)

                    # task loss
                    loss_task_all = self.criterion(output['output_logit'], labels)
                    # loss_task_l_homo = self.criterion(output['logits_l_homo'], labels)
                    # loss_task_v_homo = self.criterion(output['logits_v_homo'], labels)
                    # loss_task_a_homo = self.criterion(output['logits_a_homo'], labels)
                    # loss_task_l_hetero = self.criterion(output['logits_l_hetero'], labels)
                    # loss_task_v_hetero = self.criterion(output['logits_v_hetero'], labels)
                    # loss_task_a_hetero = self.criterion(output['logits_a_hetero'], labels)
                    # loss_task_c = self.criterion(output['logits_c'], labels)
                    loss_task = loss_task_all

                    # reconstruction loss
                    # loss_recon_l = self.MSE(output['recon_l'], output['origin_l'])
                    # loss_recon_v = self.MSE(output['recon_v'], output['origin_v'])
                    # loss_recon_a = self.MSE(output['recon_a'], output['origin_a'])
                    # loss_recon = loss_recon_l + loss_recon_v + loss_recon_a

                    # cycle consistency loss between s_x and s_x_r
                    # loss_sl_slr = self.MSE(output['s_l'].permute(1, 2, 0), output['s_l_r'])
                    # loss_sv_slv = self.MSE(output['s_v'].permute(1, 2, 0), output['s_v_r'])
                    # loss_sa_sla = self.MSE(output['s_a'].permute(1, 2, 0), output['s_a_r'])
                    # loss_s_sr = loss_sl_slr + loss_sv_slv + loss_sa_sla

                    # ort loss
                    # cosine_similarity_s_c_l = self.cosine(output['s_l'], output['c_l'],
                    #                                       torch.tensor([-1]).cuda()).mean(0)
                    # cosine_similarity_s_c_v = self.cosine(output['s_v'], output['c_v'],
                    #                                       torch.tensor([-1]).cuda()).mean(0)
                    # cosine_similarity_s_c_a = self.cosine(output['s_a'], output['c_a'],
                    #                                       torch.tensor([-1]).cuda()).mean(0)
                    # loss_ort = cosine_similarity_s_c_l + cosine_similarity_s_c_v + cosine_similarity_s_c_a

                    # margin loss
                    # c_l, c_v, c_a = output['c_l_sim'], output['c_v_sim'], output['c_a_sim']
                    # ids, feats = [], []
                    # for i in range(labels.size(0)):
                    #     feats.append(c_l[i].view(1, -1))
                    #     feats.append(c_v[i].view(1, -1))
                    #     feats.append(c_a[i].view(1, -1))
                    #     ids.append(labels[i].view(1, -1))
                    #     ids.append(labels[i].view(1, -1))
                    #     ids.append(labels[i].view(1, -1))
                    # feats = torch.cat(feats, dim=0)
                    # ids = torch.cat(ids, dim=0)
                    # loss_sim = self.sim_loss(ids, feats)

                    # homo GD loss
                    # loss_reg_homo, loss_logit_homo, loss_repr_homo = \
                    #     model[1].distillation_loss(logits_homo, reprs_homo, edges_homo)
                    # graph_distill_loss_homo = 0.05 * (loss_logit_homo + loss_reg_homo)

                    # # hetero GD loss
                    # loss_reg_hetero, loss_logit_hetero, loss_repr_hetero = \
                    #     model[2].distillation_loss(logits_hetero, reprs_hetero, edges_hetero)
                    # graph_distill_loss_hetero = 0.05 * (loss_logit_hetero + loss_repr_hetero + loss_reg_hetero)

                    # com clip loss
                    # logits_c_sd = output['logits_c_sd']
                    # loss_c_sd_lv = self.c_clip_loss(*logits_c_sd[0])
                    # loss_c_sd_va = self.c_clip_loss(*logits_c_sd[1])
                    # loss_c_sd_la = self.c_clip_loss(*logits_c_sd[2])
                    # loss_c_sd = loss_c_sd_lv + loss_c_sd_va + loss_c_sd_la        

                    # combined_loss = loss_task + \
                    #                 graph_distill_loss_homo + graph_distill_loss_hetero + \
                    #                 (loss_s_sr + loss_recon + (loss_sim+loss_ort) * 0.1) * 0.1
                    
                    # hetero loss
                    s_l_dict = output['s_l_dict']
                    s_v_dict = output['s_v_dict']
                    s_a_dict = output['s_a_dict']
                    logit_scale = output['logit_scale']

                    ids_dict, feats_dict = [], []
                    for i in range(labels.size(0)):
                        feats_dict.append(s_l_dict[i].view(1, -1))
                        feats_dict.append(s_v_dict[i].view(1, -1))
                        feats_dict.append(s_a_dict[i].view(1, -1))
                        ids_dict.append(labels[i].view(1, -1))
                        ids_dict.append(labels[i].view(1, -1))
                        ids_dict.append(labels[i].view(1, -1))
                    feats_dict = torch.cat(feats_dict, dim=0)
                    ids_dict = torch.cat(ids_dict, dim=0)

                    # loss_hetero_dict = self.hetero_loss(ids_dict, feats_dict, logit_scale)
                    loss_hetero_dict = self.hetero_loss(ids_dict, feats_dict)

                    # loss_hetero_dict.register_hook(save_grad('loss_hetero_dict'))
                    # loss_task.register_hook(save_grad('loss_task'))
                    # combined_loss = loss_task
                    combined_loss = loss_task + self.args.loss_factor * loss_hetero_dict
                    
                    combined_loss.backward()

                    if self.args.grad_clip != -1.0:
                        params = list(model[0].parameters())
                        nn.utils.clip_grad_value_(params, self.args.grad_clip)
                    
                    train_loss += combined_loss.item()
                    loss_hetero += loss_hetero_dict.item()
                    # loss_hetero = 0

                    y_pred.append(output['output_logit'].cpu())
                    y_true.append(labels.cpu())
                    if not left_epochs:
                        optimizer.step()
                        left_epochs = self.args.update_epochs
                if not left_epochs:
                    # update
                    optimizer.step()


            np.save(f'/workspace/projects/mmsa/visualization/{self.args.dataset_name}_task/dict_table_{epochs}.npy', np.array(dict_table))
            train_loss = train_loss / len(dataloader['train'])
            loss_hetero = loss_hetero / len(dataloader['train'])
            pred, true = torch.cat(y_pred), torch.cat(y_true)
            train_results = self.metrics(pred, true)

            self.writer.add_scalar('Loss/train', train_loss, epochs)
            self.writer.add_scalar('Loss/train_main', train_loss - self.args.loss_factor * loss_hetero, epochs)
            self.writer.add_scalar('Loss/train_dict', loss_hetero, epochs)
            
            if (self.args.dataset_name == 'sims'):
                self.writer.add_scalar('Train/Acc2', train_results['Mult_acc_2'], epochs)
                self.writer.add_scalar('Train/Acc3', train_results['Mult_acc_3'], epochs)
                self.writer.add_scalar('Train/Acc5', train_results['Mult_acc_5'], epochs)
                self.writer.add_scalar('Train/F1', train_results['F1_score'], epochs)
                self.writer.add_scalar('Train/MAE', train_results['MAE'], epochs)
                self.writer.add_scalar('Train/Corr', train_results['Corr'], epochs)
            else:
                self.writer.add_scalar('Train/Acc2_has0', train_results['Has0_acc_2'], epochs)
                self.writer.add_scalar('Train/F1_has0', train_results['Has0_F1_score'], epochs)
                self.writer.add_scalar('Train/Acc2_non0', train_results['Non0_acc_2'], epochs)
                self.writer.add_scalar('Train/F1_non0', train_results['Non0_F1_score'], epochs)
                self.writer.add_scalar('Train/Acc5', train_results['Mult_acc_5'], epochs)
                self.writer.add_scalar('Train/Acc7', train_results['Mult_acc_7'], epochs)
                self.writer.add_scalar('Train/MAE', train_results['MAE'], epochs)
                self.writer.add_scalar('Train/Corr', train_results['Corr'], epochs)
            

            logger.info(
                f">> Epoch: {epochs} "
                f"TRAIN-({self.args.model_name}) [{epochs - best_epoch}/{epochs}/{self.args.cur_seed}] "
                f">> total_loss: {round(train_loss, 4)} "
                f">> loss_hetero_dict: {round(loss_hetero, 4)} "
                f"{dict_to_str(train_results)}"
            )

            # validation
            val_results = self.do_test(model[0], dataloader['valid'], mode="VAL")
            # print(val_results)
            test_results = self.do_test(model[0], dataloader['test'], mode="TEST")

            # print(val_results)
            # print(test_results)

            self.writer.add_scalar('Loss/val', val_results['Loss'], epochs)
            self.writer.add_scalar('Loss/val_main', val_results['Loss'], epochs)
            self.writer.add_scalar('Loss/test', test_results['Loss'], epochs)
            self.writer.add_scalar('Loss/test_main', test_results['Loss'], epochs)
            
            if (self.args.dataset_name == 'sims'):
                self.writer.add_scalar('Val/Acc2', val_results['Mult_acc_2'], epochs)
                self.writer.add_scalar('Val/Acc3', val_results['Mult_acc_3'], epochs)
                self.writer.add_scalar('Val/Acc5', val_results['Mult_acc_5'], epochs)
                self.writer.add_scalar('Val/F1', val_results['F1_score'], epochs)
                self.writer.add_scalar('Val/MAE', val_results['MAE'], epochs)
                self.writer.add_scalar('Val/Corr', val_results['Corr'], epochs)

                self.writer.add_scalar('Test/Acc2', test_results['Mult_acc_2'], epochs)
                self.writer.add_scalar('Test/Acc3', test_results['Mult_acc_3'], epochs)
                self.writer.add_scalar('Test/Acc5', test_results['Mult_acc_5'], epochs)
                self.writer.add_scalar('Test/F1', test_results['F1_score'], epochs)
                self.writer.add_scalar('Test/MAE', test_results['MAE'], epochs)
                self.writer.add_scalar('Test/Corr', test_results['Corr'], epochs)             
            else:
                self.writer.add_scalar('Val/Acc2_has0', val_results['Has0_acc_2'], epochs)
                self.writer.add_scalar('Val/F1_has0', val_results['Has0_F1_score'], epochs)
                self.writer.add_scalar('Val/Acc2_non0', val_results['Non0_acc_2'], epochs)
                self.writer.add_scalar('Val/F1_non0', val_results['Non0_F1_score'], epochs)
                self.writer.add_scalar('Val/Acc5', val_results['Mult_acc_5'], epochs)
                self.writer.add_scalar('Val/Acc7', val_results['Mult_acc_7'], epochs)
                self.writer.add_scalar('Val/MAE', val_results['MAE'], epochs)
                self.writer.add_scalar('Val/Corr', val_results['Corr'], epochs)

                self.writer.add_scalar('Test/Acc2_has0', test_results['Has0_acc_2'], epochs)
                self.writer.add_scalar('Test/F1_has0', test_results['Has0_F1_score'], epochs)
                self.writer.add_scalar('Test/Acc2_non0', test_results['Non0_acc_2'], epochs)
                self.writer.add_scalar('Test/F1_non0', test_results['Non0_F1_score'], epochs)
                self.writer.add_scalar('Test/Acc5', test_results['Mult_acc_5'], epochs)
                self.writer.add_scalar('Test/Acc7', test_results['Mult_acc_7'], epochs)
                self.writer.add_scalar('Test/MAE', test_results['MAE'], epochs)
                self.writer.add_scalar('Test/Corr', test_results['Corr'], epochs)

            # self.writer.add_scalars('Loss', {
            #     'train': train_loss,
            #     'val': val_results['Loss'],
            #     'test': test_results['Loss']
            # }, epochs)

            cur_valid = val_results[self.args.KeyEval]
            scheduler.step(val_results['Loss'])
            # save each epoch model
            torch.save(model[0].state_dict(), f'/workspace/projects/mmsa/pt/{self.args.dataset_name}/' + str(epochs) + '.pth')
            # save best model
            isBetter = cur_valid <= (best_valid - 1e-6) if min_or_max == 'min' else cur_valid >= (best_valid + 1e-6)
            if isBetter:
                best_valid, best_epoch = cur_valid, epochs
                # save model
                # model_save_path = './pt/dmd.pth'
                model_save_path = f'/workspace/projects/mmsa/pt/{self.args.dataset_name}/dmd_{self.args.dataset_name}.pth'
                
                torch.save(model[0].state_dict(), model_save_path)

            if return_epoch_results:
                train_results["Loss"] = train_loss
                epoch_results['train'].append(train_results)
                epoch_results['valid'].append(val_results)
                test_results = self.do_test(model, dataloader['test'], mode="TEST")
                epoch_results['test'].append(test_results)
            # early stop
            if epochs - best_epoch >= self.args.early_stop:
                return epoch_results if return_epoch_results else None

    def do_test(self, model, dataloader, mode="VAL", return_sample_results=False):

        model.eval()
        y_pred, y_true = [], []

        # pos_label_list = []
        # neg_label_list = []
        # intersection_1_2_3_list = []
        # l_att_weight_list = []
        # v_att_weight_list = []
        # a_att_weight_list = []
        label_list = []
        raw_text_list = []
        ids_list = []
        lva_att_weight_list = []

        test_loss = 0.0
        dict_loss = 0.0
        if return_sample_results:
            ids, sample_results = [], []
            all_labels = []
            features = {
                "Feature_t": [],
                "Feature_a": [],
                "Feature_v": [],
                "Feature_f": [],
            }
        
        with torch.no_grad():
            with tqdm(dataloader) as td:
                for batch_data in td:
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    labels = batch_data['labels']['M'].to(self.args.device)
                    labels = labels.view(-1, 1)

                    output = model(text, audio, vision, is_distill=True)

                    lva_att_weight_list.append(output['lva_att_weight'].detach().cpu().numpy())
                    label_list.append(labels.detach().cpu().numpy())
                    raw_text_list.append(batch_data['raw_text'])
                    ids_list.append(batch_data['id'])
                    
                    # print(lva_att_weight.shape)

                    # hetero loss
                    s_l_dict = output['s_l_dict']
                    s_v_dict = output['s_v_dict']
                    s_a_dict = output['s_a_dict']
                    logit_scale = output['logit_scale']

                    # ids_dict, feats_dict = [], []
                    # for i in range(labels.size(0)):
                    #     feats_dict.append(s_l_dict[i].view(1, -1))
                    #     feats_dict.append(s_v_dict[i].view(1, -1))
                    #     feats_dict.append(s_a_dict[i].view(1, -1))
                    #     ids_dict.append(labels[i].view(1, -1))
                    #     ids_dict.append(labels[i].view(1, -1))
                    #     ids_dict.append(labels[i].view(1, -1))
                    # feats_dict = torch.cat(feats_dict, dim=0)
                    # ids_dict = torch.cat(ids_dict, dim=0)

                    # l_att_weight_list.append(output['l_att_weight'].tolist())
                    # v_att_weight_list.append(output['v_att_weight'].tolist())
                    # a_att_weight_list.append(output['a_att_weight'].tolist())
                    
                    # lva_att_weight = output['lva_att_weight']


                    # loss_dictt, pos, neg, pos_label, neg_label = self.hetero_loss(ids_dict, feats_dict, logit_scale, lva_att_weight)
                    # loss_dictt = self.hetero_loss(ids_dict, feats_dict, logit_scale)
                    # pos_list.append(pos)
                    # neg_list.append(neg)
                    # pos_label_list.append(pos_label)
                    # neg_label_list.append(neg_label)
                    # loss_hetero_dict = self.args.loss_factor *loss_dictt

                    loss = self.criterion(output['output_logit'], labels)
                    
                    # dict_loss = dict_loss + loss_hetero_dict.item()
                    test_loss = test_loss + loss.item()
                    y_pred.append(output['output_logit'].cpu())
                    y_true.append(labels.cpu())

        # dict_loss = dict_loss / len(dataloader)
        np.save(f'/workspace/projects/mmsa/visualization/lva_att_weight_train.npy', np.asarray(lva_att_weight_list, dtype=object))
        np.save(f'/workspace/projects/mmsa/visualization/label_list.npy', np.asarray(label_list, dtype=object))
        np.save(f'/workspace/projects/mmsa/visualization/raw_text_list.npy', np.asarray(raw_text_list, dtype=object))
        np.save(f'/workspace/projects/mmsa/visualization/ids_list.npy', np.asarray(ids_list, dtype=object))
        test_loss = test_loss / len(dataloader)
        pred, true = torch.cat(y_pred), torch.cat(y_true)

        # np.save('/workspace/projects/mmsa/visualization/intersection_1_2_3_list.npy', np.asarray(intersection_1_2_3_list, dtype=object))

        # np.save('/workspace/projects/mmsa/visualization/l_att_weight_list.npy', np.array(l_att_weight_list, dtype=object))
        # np.save('/workspace/projects/mmsa/visualization/v_att_weight_list.npy', np.array(v_att_weight_list, dtype=object))  
        # np.save('/workspace/projects/mmsa/visualization/a_att_weight_list.npy', np.array(a_att_weight_list, dtype=object))

        # np.save('/workspace/projects/mmsa/visualization/pos_list.npy', np.array(pos_list, dtype=object))
        # np.save('/workspace/projects/mmsa/visualization/neg_list.npy', np.array(neg_list, dtype=object))
        # np.save('/workspace/projects/mmsa/visualization/pos_label_list.npy', np.array(pos_label_list, dtype=object))
        # np.save('/workspace/projects/mmsa/visualization/neg_label_list.npy', np.array(neg_label_list, dtype=object))

        eval_results = self.metrics(pred, true)
        eval_results["Loss"] = round(test_loss, 4)
        # eval_results["Loss_dict"] = round(dict_loss, 4)
        logger.info(f"{mode}-({self.args.model_name}) >> {dict_to_str(eval_results)}")

        if return_sample_results:
            eval_results["Ids"] = ids
            eval_results["SResults"] = sample_results
            for k in features.keys():
                features[k] = np.concatenate(features[k], axis=0)
            eval_results['Features'] = features
            eval_results['Labels'] = all_labels

        return eval_results