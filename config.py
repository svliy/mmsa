import json
import os
from pathlib import Path
from easydict import EasyDict as edict
import random


def get_config_regression(model_name, dataset_name, config_file=""):
    """
    Get the regression config of given dataset and model from config file.

    Parameters:
        config_file (str): Path to config file, if given an empty string, will use default config file.
        model_name (str): Name of model.
        dataset_name (str): Name of dataset.

    Returns:
        config (dict): config of the given dataset and model
    """
    if config_file == "":
        config_file = Path(__file__).parent / "config" / "config_regression.json"
    with open(config_file, 'r') as f:
        config_all = json.load(f)
    
    model_common_args = config_all[model_name]['commonParams']
    model_dataset_args = config_all[model_name]['datasetParams'][dataset_name]
    dataset_args = config_all['datasetCommonParams'][dataset_name]
    # use aligned feature if the model requires it, otherwise use unaligned feature
    dataset_args = dataset_args['aligned'] if (model_common_args['need_data_aligned'] and 'aligned' in dataset_args) else dataset_args['unaligned']

    config = {}
    config['model_name'] = model_name
    config['dataset_name'] = dataset_name
    # 数据集配置
    config.update(dataset_args)
    # 训练配置
    config.update(model_common_args)
    # 模型架构配置
    config.update(model_dataset_args)
    config['featurePath'] = os.path.join(config_all['datasetCommonParams']['dataset_root_dir'], config['featurePath'])
    config = edict(config)

    return config


def get_config_tune(model_name, dataset_name, config_file="", random_choice=True):
    """
    Get the tuning config of given dataset and model from config file.

    Parameters:
        model_name (str): Name of model.
        dataset_name (str): Name of dataset.
        config_file (str): Path to config file, if given an empty string, will use default config file.
        random_choice (bool): If True, will randomly choose a config from the list of configs.

    Returns:
        config (dict): config of the given dataset and model
    """
    if config_file == "":
        config_file = Path(__file__).parent / "config" / "config_tune.json"
    with open(config_file, 'r') as f:
        config_all = json.load(f)
    model_common_args = config_all[model_name]['commonParams']
    model_dataset_args = config_all[model_name]['datasetParams'][dataset_name] if 'datasetParams' in config_all[model_name] else {}
    model_debug_args = config_all[model_name]['debugParams']
    dataset_args = config_all['datasetCommonParams'][dataset_name]
    # use aligned feature if the model requires it, otherwise use unaligned feature
    dataset_args = dataset_args['aligned'] if (model_common_args['need_data_aligned'] and 'aligned' in dataset_args) else dataset_args['unaligned']

    # random choice of args
    if random_choice:
        for item in model_debug_args['d_paras']:
            if type(model_debug_args[item]) == list:
                model_debug_args[item] = random.choice(model_debug_args[item])
            elif type(model_debug_args[item]) == dict: # nested params, 2 levels max
                for k, v in model_debug_args[item].items():
                    model_debug_args[item][k] = random.choice(v)

    config = {}
    config['model_name'] = model_name
    config['dataset_name'] = dataset_name
    config.update(dataset_args)
    config.update(model_common_args)
    config.update(model_dataset_args)
    config.update(model_debug_args)
    config['featurePath'] = os.path.join(config_all['datasetCommonParams']['dataset_root_dir'], config['featurePath'])

    config = edict(config)

    return config