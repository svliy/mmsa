"""
Testing script for DMD
"""
from run import DMD_run

DMD_run(model_name='dmd',
        dataset_name='mosei',
        config_file='/workspace/projects/mmsa/config/config_regression.json',
        is_tune=False,
        seeds=[1111],
        model_save_dir="./pt",
        res_save_dir="./result",
        log_dir="./log/test",
        mode='test',
        is_distill=False)
