import gc
import logging
import os
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import random
from easydict import EasyDict as edict
from config import get_config_regression, get_config_tune
from data_loader import MMDataLoader
from trains import ATIO
from utils import assign_gpu, setup_seed
from trains.singleTask.model import dmd
from trains.singleTask.distillnets import get_distillation_kernel, get_distillation_kernel_homo
from trains.singleTask.misc import softmax
from torch.utils.tensorboard import SummaryWriter
import sys
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:2"
logger = logging.getLogger('MMSA')

def _set_logger(log_dir, model_name, dataset_name, verbose_level):

    # base logger
    log_file_path = Path(log_dir) / f"{model_name}-{dataset_name}.log"
    logger = logging.getLogger('MMSA')
    logger.setLevel(logging.DEBUG)

    # file handler
    fh = logging.FileHandler(log_file_path)
    fh_formatter = logging.Formatter('%(asctime)s - %(name)s [%(levelname)s] - %(message)s')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fh_formatter)
    logger.addHandler(fh)

    # stream handler
    stream_level = {0: logging.ERROR, 1: logging.INFO, 2: logging.DEBUG}
    ch = logging.StreamHandler()
    ch.setLevel(stream_level[verbose_level])
    ch_formatter = logging.Formatter('%(name)s - %(message)s')
    ch.setFormatter(ch_formatter)
    logger.addHandler(ch)

    return logger


def DMD_run(
    model_name, dataset_name, config=None, config_file="", seeds=[], is_tune=False,
    tune_times=500, feature_T="", feature_A="", feature_V="",
    model_save_dir="", res_save_dir="", log_dir="",
    gpu_ids=[0], num_workers=4, verbose_level=1, mode = '', is_distill = False
):
    # Initialization
    model_name = model_name.lower() # dmd
    dataset_name = dataset_name.lower() # mosei
    
    # 读取配置文件
    if config_file != "":
        config_file = Path(config_file)
    else: # use default config files
        if is_tune:
            config_file = Path(__file__).parent / "config" / "config_tune.json"
        else:
            config_file = Path(__file__).parent / "config" / "config.json"
    if not config_file.is_file():
        raise ValueError(f"Config file {str(config_file)} not found.")
    
    # 模型权重保存目录
    if model_save_dir == "":
        model_save_dir = Path.home() / "MMSA" / "saved_models"
    Path(model_save_dir).mkdir(parents=True, exist_ok=True)

    # 训练结果保存目录
    if res_save_dir == "":
        res_save_dir = Path.home() / "MMSA" / "results"
    Path(res_save_dir).mkdir(parents=True, exist_ok=True)

    # 日志保存目录
    if log_dir == "":
        log_dir = Path.home() / "MMSA" / "logs"
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    seeds = seeds if seeds != [] else [1111, 1112, 1113, 1114, 1115]
    logger = _set_logger(log_dir, model_name, dataset_name, verbose_level)

    # 可视化工具
    writer = SummaryWriter()
    
    if is_tune: # run tune
        setup_seed(seeds[0])
        logger.info(f"Tuning with seed {seeds[0]}")
        initial_args = get_config_tune(model_name, dataset_name, config_file)
        initial_args['model_save_path'] = Path(model_save_dir) / f"{initial_args['model_name']}-{initial_args['dataset_name']}.pth"
        initial_args['device'] = assign_gpu(gpu_ids)
        initial_args['train_mode'] = 'regression' # backward compatibility. TODO: remove all train_mode in code
        initial_args['feature_T'] = feature_T
        initial_args['feature_A'] = feature_A
        initial_args['feature_V'] = feature_V

        torch.cuda.set_device(initial_args['device'])

        res_save_dir = Path(res_save_dir) / "tune"
        res_save_dir.mkdir(parents=True, exist_ok=True)
        has_debuged = [] # save used params
        csv_file = res_save_dir / f"{dataset_name}-{model_name}.csv"
        if csv_file.is_file():
            df = pd.read_csv(csv_file)
            for i in range(len(df)):
                has_debuged.append([df.loc[i,k] for k in initial_args['d_paras']])

        for i in range(tune_times):
            args = edict(**initial_args)
            args.mode = mode # train or test
            args.is_distill = is_distill # is or not distill for cat123
            # if args.is_distill == False:
            #     args.modalities = []
            random.seed(time.time())
            new_args = get_config_tune(model_name, dataset_name, config_file)
            args.update(new_args)
            args['cur_seed'] = i + 1
            logger.info(f"{'-'*30} Tuning [{i + 1}/{tune_times}] {'-'*30}")
            logger.info(f"Args: {args}")
            # check if this param has been run
            cur_param = [args[k] for k in args['d_paras']]
            if cur_param in has_debuged:
                logger.info(f"This set of parameters has been run. Skip.")
                time.sleep(1)
                continue
            has_debuged.append(cur_param)
            # actual running
            result = _run(args, num_workers, is_tune)
            print(f'result: {result}')
            # save result to csv file
            if Path(csv_file).is_file():
                df2 = pd.read_csv(csv_file)
            else:
                df2 = pd.DataFrame(columns = [k for k in args.d_paras] + [k for k in result.keys()])
            res = [args[c] for c in args.d_paras]
            for col in result.keys():
                value = result[col]
                res.append(value)
            df2.loc[len(df2)] = res
            df2.to_csv(csv_file, index=None)
            logger.info(f"Results saved to {csv_file}.")
    else:
        # 从配置文件中获取给定数据集dataset和模型model的回归配置config_file。
        args = get_config_regression(model_name, dataset_name, config_file)
        args.is_distill = is_distill  # use or not use distill, train use, test not use
        args.mode = mode # train or test
        # 没有用到啊!!!!!!!!!
        args['model_save_path'] = Path(model_save_dir) / f"{args['model_name']}_{args['dataset_name']}.pth"
        # 设置单卡GPU
        args['device'] = assign_gpu(gpu_ids)
        # 有用到吗？
        args['train_mode'] = 'regression'
        args['feature_T'] = feature_T
        args['feature_A'] = feature_A
        args['feature_V'] = feature_V
        # 训练的时候加入的配置
        if config:
            args.update(config)

        # 结果保存路径 
        res_save_dir = Path(res_save_dir) / "normal"
        res_save_dir.mkdir(parents=True, exist_ok=True)
        model_results = []
        for i, seed in enumerate(seeds):
            setup_seed(seed)
            args['cur_seed'] = i + 1
            # pass in parameters from config file
            result = _run(args, num_workers, is_tune)

            model_results.append(result)
        if args.is_distill:
            criterions = list(model_results[0].keys())
            # save result to csv
            csv_file = res_save_dir / f"{dataset_name}.csv"
            if csv_file.is_file():
                df = pd.read_csv(csv_file)
            else:
                df = pd.DataFrame(columns=["Model"] + criterions)
            # save results
            res = [model_name]
            for c in criterions:
                values = [r[c] for r in model_results]
                mean = round(np.mean(values)*100, 2)
                std = round(np.std(values)*100, 2)
                res.append((mean, std))
            df.loc[len(df)] = res
            df.to_csv(csv_file, index=None)
            logger.info(f"Results saved to {csv_file}.")
            # logger.info(f"Config file: \n {args}")
    
    writer.flush()


def _run(args, num_workers=4, is_tune=False, from_sena=False):

    # build dataloader
    dataloader = MMDataLoader(args, num_workers)

    # TODO: logger arges
    logger.info(f"Args: {args}")

    if args.is_distill:
        print("training for DMD")

        # param of homogeneous graph distillation
        args.gd_size_low = 64  # hidden size of graph distillation
        args.w_losses_low = [1, 10]  # weights for losses: [logit, repr]
        args.metric_low = 'l1'  # distance metric for distillation loss

        # param of heterogeneous graph distillation
        args.gd_size_high = 32  # hidden size of graph distillation
        args.w_losses_high = [1, 10]  # weights for losses: [logit, repr]
        args.metric_high = 'l1'  # distance metric for distillation loss

        to_idx = [0, 1, 2]  # all modalities can be distilled from each other simultaneously
        from_idx = [0, 1, 2]  # all modalities can be distilled from each other simultaneously
        assert len(from_idx) >= 1

        model = []
        # 初始化模型架构
        model_dmd = getattr(dmd, 'DMD')(args)
        # print(model_dmd)
        model_distill_homo = getattr(get_distillation_kernel_homo, 'DistillationKernel')(n_classes=1,
                                                                               hidden_size=
                                                                               args.dst_feature_dim_nheads[0],
                                                                               gd_size=args.gd_size_low,
                                                                               to_idx=to_idx, from_idx=from_idx,
                                                                               gd_prior=softmax([0, 0, 1, 0, 1, 0], 0.25),
                                                                               gd_reg=10,
                                                                               w_losses=args.w_losses_low,
                                                                               metric=args.metric_low,
                                                                               alpha=1 / 8,
                                                                               hyp_params=args)

        model_distill_hetero = getattr(get_distillation_kernel, 'DistillationKernel')(n_classes=1,
                                                                                   hidden_size=
                                                                                   args.dst_feature_dim_nheads[0] * 2,
                                                                                   gd_size=args.gd_size_high,
                                                                                   to_idx=to_idx, from_idx=from_idx,
                                                                                   gd_prior=softmax([0, 0, 1, 0, 1, 1], 0.25),
                                                                                   gd_reg=10,
                                                                                   w_losses=args.w_losses_high,
                                                                                   metric=args.metric_high,
                                                                                   alpha=1 / 8,
                                                                                   hyp_params=args)

        model_dmd, model_distill_homo, model_distill_hetero = model_dmd.cuda(), model_distill_homo.cuda(), model_distill_hetero.cuda()

        model = [model_dmd, model_distill_homo, model_distill_hetero]
    else:
        print("testing phase for DMD")
        model = getattr(dmd, 'DMD')(args)
        model = model.cuda()

    # trainning loop
    trainer = ATIO().getTrain(args)

    # 训练不会执行这个测试部分
    if args.mode == 'test':
        model.load_state_dict(torch.load(f'./pt/{args.dataset_name}/dmd_{args.dataset_name}.pth'))

        results = trainer.do_test(model, dataloader['test'], mode="TEST")
        sys.stdout.flush()
        input('[Press Any Key to start another run]')
    else:
        # 训练部分
        epoch_results = trainer.do_train(model, dataloader, return_epoch_results=from_sena)
        # last save model
        # 每训练1个epoch，保存一个模型权重
        model[0].load_state_dict(torch.load(f'/workspace/projects/mmsa/pt/{args.dataset_name}/dmd_{args.dataset_name}.pth'))
        # 测试部分
        # 只测试一次
        results = trainer.do_test(model[0], dataloader['test'], mode="TEST")

        del model
        torch.cuda.empty_cache()
        gc.collect()
        time.sleep(1)
    return results