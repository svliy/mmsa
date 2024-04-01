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
from model.loss import ClipInfoCELoss


logger = logging.getLogger('MMSA')

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
        self.cosine = nn.CosineEmbeddingLoss()
        self.metrics = MetricsTop(args.train_mode).getMetics(args.dataset_name)
        self.MSE = MSE()
        self.sim_loss = HingeLoss()
        self.c_clip_loss = ClipInfoCELoss()
        self.s_clip_loss = ClipInfoCELoss()

    def do_train(self, model, dataloader, return_epoch_results=False):

        # 0: DMD model, 1: Homo GD, 2: Hetero GD
        # params = list(model[0].parameters()) + \
        #          list(model[1].parameters()) + \
        #          list(model[2].parameters())
        
        # print(type(model[0].named_parameters()))
        base_model = []
        clip_model = []
        for name, p in model[0].named_parameters():
            if "clip" in name:
                clip_model += [p]
            else:
                base_model += [p]

        optimizer = optim.Adam([{'params': base_model},
                                {'params': clip_model, 'lr': self.args.dictionary_lr},
                                {'params': model[1].parameters()},
                                {'params': model[2].parameters()}], lr=self.args.learning_rate)
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
        net_distill_homo = model[1]
        net_distill_hetero = model[2]
        net.append(net_dmd)
        net.append(net_distill_homo)
        net.append(net_distill_hetero)
        model = net

        while True:
            epochs += 1
            y_pred, y_true = [], []
            for mod in model:
                mod.train()

            train_loss = 0.0
            loss_sd_c = 0.0
            loss_sd_s = 0.0
            loss_sd_c_lv = 0.0
            loss_sd_c_va = 0.0
            loss_sd_c_la = 0.0
            loss_sd_s_lv = 0.0
            loss_sd_s_va = 0.0
            loss_sd_s_la = 0.0
            left_epochs = self.args.update_epochs
            with tqdm(dataloader['train']) as td:
                for batch_data in td:

                    if left_epochs == self.args.update_epochs:
                        optimizer.zero_grad()
                    left_epochs -= 1
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    labels = batch_data['labels']['M'].to(self.args.device)
                    labels = labels.view(-1, 1)

                    logits_homo, reprs_homo, logits_hetero, reprs_hetero = [], [], [], []

                    output = model[0](text, audio, vision, is_distill=True)

                    # logits for homo GD
                    # [3, 16, 1]
                    logits_homo.append(output['logits_l_homo'])
                    logits_homo.append(output['logits_v_homo'])
                    logits_homo.append(output['logits_a_homo'])

                    # reprs for homo GD
                    reprs_homo.append(output['repr_l_homo'])
                    reprs_homo.append(output['repr_v_homo'])
                    reprs_homo.append(output['repr_a_homo'])

                    # logits for hetero GD
                    logits_hetero.append(output['logits_l_hetero'])
                    logits_hetero.append(output['logits_v_hetero'])
                    logits_hetero.append(output['logits_a_hetero'])

                    # reprs for hetero GD
                    reprs_hetero.append(output['repr_l_hetero'])
                    reprs_hetero.append(output['repr_v_hetero'])
                    reprs_hetero.append(output['repr_a_hetero'])

                    logits_homo = torch.stack(logits_homo)
                    reprs_homo = torch.stack(reprs_homo)

                    logits_hetero = torch.stack(logits_hetero)
                    reprs_hetero = torch.stack(reprs_hetero)

                    # edges for homo distill
                    edges_homo, edges_origin_homo = model[1](logits_homo, reprs_homo)

                    # edges for hetero distill
                    edges_hetero, edges_origin_hetero = model[2](logits_hetero, reprs_hetero)

                    # task loss
                    loss_task_all = self.criterion(output['output_logit'], labels)
                    loss_task_l_homo = self.criterion(output['logits_l_homo'], labels)
                    loss_task_v_homo = self.criterion(output['logits_v_homo'], labels)
                    loss_task_a_homo = self.criterion(output['logits_a_homo'], labels)
                    loss_task_l_hetero = self.criterion(output['logits_l_hetero'], labels)
                    loss_task_v_hetero = self.criterion(output['logits_v_hetero'], labels)
                    loss_task_a_hetero = self.criterion(output['logits_a_hetero'], labels)
                    loss_task_c = self.criterion(output['logits_c'], labels)
                    loss_task = loss_task_all + loss_task_l_homo + loss_task_v_homo + loss_task_a_homo + loss_task_l_hetero + loss_task_v_hetero + loss_task_a_hetero + loss_task_c

                    # reconstruction loss
                    loss_recon_l = self.MSE(output['recon_l'], output['origin_l'])
                    loss_recon_v = self.MSE(output['recon_v'], output['origin_v'])
                    loss_recon_a = self.MSE(output['recon_a'], output['origin_a'])
                    loss_recon = loss_recon_l + loss_recon_v + loss_recon_a

                    # cycle consistency loss between s_x and s_x_r
                    loss_sl_slr = self.MSE(output['s_l'].permute(1, 2, 0), output['s_l_r'])
                    loss_sv_slv = self.MSE(output['s_v'].permute(1, 2, 0), output['s_v_r'])
                    loss_sa_sla = self.MSE(output['s_a'].permute(1, 2, 0), output['s_a_r'])
                    loss_s_sr = loss_sl_slr + loss_sv_slv + loss_sa_sla

                    # ort loss
                    cosine_similarity_s_c_l = self.cosine(output['s_l'], output['c_l'],
                                                          torch.tensor([-1]).cuda()).mean(0)
                    cosine_similarity_s_c_v = self.cosine(output['s_v'], output['c_v'],
                                                          torch.tensor([-1]).cuda()).mean(0)
                    cosine_similarity_s_c_a = self.cosine(output['s_a'], output['c_a'],
                                                          torch.tensor([-1]).cuda()).mean(0)
                    loss_ort = cosine_similarity_s_c_l + cosine_similarity_s_c_v + cosine_similarity_s_c_a

                    # margin loss
                    c_l, c_v, c_a = output['c_l_sim'], output['c_v_sim'], output['c_a_sim']
                    ids, feats = [], []
                    for i in range(labels.size(0)):
                        feats.append(c_l[i].view(1, -1))
                        feats.append(c_v[i].view(1, -1))
                        feats.append(c_a[i].view(1, -1))
                        ids.append(labels[i].view(1, -1))
                        ids.append(labels[i].view(1, -1))
                        ids.append(labels[i].view(1, -1))
                    feats = torch.cat(feats, dim=0)
                    ids = torch.cat(ids, dim=0)
                    loss_sim = self.sim_loss(ids, feats)

                    # homo GD loss
                    loss_reg_homo, loss_logit_homo, loss_repr_homo = \
                        model[1].distillation_loss(logits_homo, reprs_homo, edges_homo)
                    graph_distill_loss_homo = 0.05 * (loss_logit_homo + loss_reg_homo)

                    # hetero GD loss
                    loss_reg_hetero, loss_logit_hetero, loss_repr_hetero = \
                        model[2].distillation_loss(logits_hetero, reprs_hetero, edges_hetero)
                    graph_distill_loss_hetero = 0.05 * (loss_logit_hetero + loss_repr_hetero + loss_reg_hetero)

                    # com clip loss
                    logits_c_sd = output['logits_c_sd']
                    loss_c_sd_lv = self.c_clip_loss(*logits_c_sd[0])
                    loss_c_sd_va = self.c_clip_loss(*logits_c_sd[1])
                    loss_c_sd_la = self.c_clip_loss(*logits_c_sd[2])
                    loss_c_sd = loss_c_sd_lv + loss_c_sd_va + loss_c_sd_la

                    # logits_s_sd = output['logits_s_sd']
                    # loss_s_sd_lv = self.s_clip_loss(*logits_s_sd[0])
                    # loss_s_sd_va = self.s_clip_loss(*logits_s_sd[1])
                    # loss_s_sd_la = self.s_clip_loss(*logits_s_sd[2])
                    # loss_s_sd = loss_s_sd_lv + loss_s_sd_va + loss_s_sd_la

                    # loss_c_sd_lv = self.c_clip_loss(*output['logits_c_lv_sd'])
                    # loss_c_sd_va = self.c_clip_loss(*output['logits_c_la_sd'])
                    # loss_c_sd_la = self.c_clip_loss(*output['logits_c_va_sd'])
                    # loss_c_sd = loss_c_sd_lv + loss_c_sd_va + loss_c_sd_la

                    # loss_s_sd_lv = self.s_clip_loss(*output['logits_s_lv_sd'])
                    # loss_s_sd_va = self.s_clip_loss(*output['logits_s_la_sd'])
                    # loss_s_sd_la = self.s_clip_loss(*output['logits_s_va_sd'])
                    # loss_s_sd = loss_s_sd_lv + loss_s_sd_va + loss_s_sd_la

                    combined_loss = loss_task + \
                                    graph_distill_loss_homo + graph_distill_loss_hetero + \
                                    (loss_s_sr + loss_recon + (loss_sim+loss_ort) * 0.1) * 0.1

                    combined_loss += self.args.loss_factor * loss_c_sd

                    combined_loss.backward()

                    if self.args.grad_clip != -1.0:
                        params = base_model + clip_model + list(model[1].parameters()) + \
                                 list(model[2].parameters())
                        nn.utils.clip_grad_value_(params, self.args.grad_clip)

                    train_loss += combined_loss.item()
                    loss_sd_c += loss_c_sd.item()
                    # loss_sd_s += loss_s_sd.item()
                    loss_sd_c_lv += loss_c_sd_lv.item()
                    loss_sd_c_va += loss_c_sd_va.item()
                    loss_sd_c_la += loss_c_sd_la.item()
                    # loss_sd_s_lv += loss_s_sd_lv.item()
                    # loss_sd_s_va += loss_s_sd_va.item()
                    # loss_sd_s_la += loss_s_sd_la.item()

                    y_pred.append(output['output_logit'].cpu())
                    y_true.append(labels.cpu())
                    if not left_epochs:
                        optimizer.step()
                        left_epochs = self.args.update_epochs
                if not left_epochs:
                    # update
                    optimizer.step()

            train_loss = train_loss / len(dataloader['train'])
            loss_sd_c = loss_sd_c / len(dataloader['train'])
            loss_sd_s = loss_sd_s / len(dataloader['train'])
            loss_sd_c_lv = loss_sd_c_lv / len(dataloader['train'])
            loss_sd_c_va = loss_sd_c_va / len(dataloader['train'])
            loss_sd_c_la = loss_sd_c_la / len(dataloader['train'])
            loss_sd_s_lv = loss_sd_s_lv / len(dataloader['train'])
            loss_sd_s_va = loss_sd_s_va / len(dataloader['train'])
            loss_sd_s_la = loss_sd_s_la / len(dataloader['train'])
            pred, true = torch.cat(y_pred), torch.cat(y_true)
            train_results = self.metrics(pred, true)

            logger.info(
                f">> Epoch: {epochs} "
                f"TRAIN-({self.args.model_name}) [{epochs - best_epoch}/{epochs}/{self.args.cur_seed}] "
                f">> total_loss: {round(train_loss, 4)} "
                f">> origin_loss: {round(train_loss - 0.1 * (loss_sd_c + loss_sd_s), 4)} "
                f">> loss_sd_c: {round(loss_sd_c, 4)} "
                f">> loss_sd_s: {round(loss_sd_s, 4)} "
                f">> loss_sd_c_lv: {round(loss_sd_c_lv, 4)} "
                f">> loss_sd_c_va: {round(loss_sd_c_va, 4)} "
                f">> loss_sd_c_la: {round(loss_sd_c_la, 4)} "
                # f">> loss_sd_s_lv: {round(loss_sd_s_lv, 4)} "
                # f">> loss_sd_s_va: {round(loss_sd_s_va, 4)} "
                # f">> loss_sd_s_la: {round(loss_sd_s_la, 4)} "
                f"{dict_to_str(train_results)}"
            )

            # validation
            val_results = self.do_test(model[0], dataloader['valid'], mode="VAL")
            test_results = self.do_test(model[0], dataloader['test'], mode="TEST")
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

        eval_loss = 0.0
        if return_sample_results:
            ids, sample_results = [], []
            all_labels = []
            features = {
                "Feature_t": [],
                "Feature_a": [],
                "Feature_v": [],
                "Feature_f": [],
            }
        
        # # attention weight
        # att_weight_c_l = []
        # att_weight_c_v = []
        # att_weight_s_l = []
        # att_weight_s_v = []
        # # 字典
        # c_dict = []
        # s_dict = []

        # df = pd.read_csv('/home/omnisky/Documents/ry/projects/result/visualization/a_mosei_testset.csv')
        # index = 0
        
        with torch.no_grad():
            with tqdm(dataloader) as td:
                for batch_data in td:
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    labels = batch_data['labels']['M'].to(self.args.device)
                    labels = labels.view(-1, 1)

                    # file_name = df.iloc[index]['ID']
                    # folder_name = '/home/omnisky/Documents/ry/projects/result/visualization/' + file_name
                    # os.mkdir(folder_name)

                    output = model(text, audio, vision, is_distill=True)
                    
                    # c_dict = output['c_dict']
                    # s_dict = output['s_dict']
                    # pd.DataFrame(output['att_weight_c_l'].cpu().numpy()).to_csv(folder_name+ '/att_weight_c_l.csv', index=False)
                    # pd.DataFrame(output['att_weight_c_v'].cpu().numpy()).to_csv(folder_name+ '/att_weight_c_v.csv', index=False)
                    # pd.DataFrame(output['att_weight_s_l'].cpu().numpy()).to_csv(folder_name+ '/att_weight_s_l.csv', index=False)
                    # pd.DataFrame(output['att_weight_s_v'].cpu().numpy()).to_csv(folder_name+ '/att_weight_s_v.csv', index=False)

                    loss = self.criterion(output['output_logit'], labels)
                    eval_loss += loss.item()
                    y_pred.append(output['output_logit'].cpu())
                    y_true.append(labels.cpu())

                    # index += 1
        
        # pd.DataFrame(c_dict.detach().cpu().numpy()).to_csv('/home/omnisky/Documents/ry/projects/result/dict/c_dict.csv', index=False)
        # pd.DataFrame(s_dict.detach().cpu().numpy()).to_csv('/home/omnisky/Documents/ry/projects/result/dict/s_dict.csv', index=False)


        eval_loss = eval_loss / len(dataloader)
        pred, true = torch.cat(y_pred), torch.cat(y_true)

        eval_results = self.metrics(pred, true)
        eval_results["Loss"] = round(eval_loss, 4)
        logger.info(f"{mode}-({self.args.model_name}) >> {dict_to_str(eval_results)}")

        if return_sample_results:
            eval_results["Ids"] = ids
            eval_results["SResults"] = sample_results
            for k in features.keys():
                features[k] = np.concatenate(features[k], axis=0)
            eval_results['Features'] = features
            eval_results['Labels'] = all_labels

        return eval_results