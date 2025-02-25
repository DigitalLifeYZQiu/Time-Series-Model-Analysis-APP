import argparse
import os
import torch
from exp.exp_large_few_shot_roll_demo import Exp_Large_Few_Shot_Roll_Demo
import random
import numpy as np

argsDict={
    # 需要配置的参数
    'root_path': './dataset/Gas/',
    'data_path': '919368_data.csv', # '134312_data.csv' | '134361_data.csv' | '919368_data.csv'
    'ckpt_path': 'checkpoints/Timer_forecast_1.0.ckpt',
    # 其他可以默认的参数
    'task_name': 'large_finetune',
    'is_training': 0,
    'model_id': '2G_{672}_{96}_{96}_',
    'model': 'Timer',
    'seed': 1,
    'data': 'Gas',
    'features': 'M',
    'target': 'OT',
    'freq': 'h',
    'checkpoints': './ckpt/',
    'seq_len': 672,
    'label_len': 576,
    'pred_len': 96,
    'seasonal_patterns': 'Monthly',
    'mask_rate': 0.25,
    'anomaly_ratio': 0.25,
    'top_k': 3,
    'num_kernels': 6,
    'enc_in': 1,
    'dec_in': 1,
    'c_out': 1,
    'd_model': 1024,
    'n_heads': 8,
    'e_layers': 8,
    'd_layers': 1,
    'd_ff': 2048,
    'moving_avg': 25,
    'factor': 3,
    'distil': True,
    'dropout': 0.1,
    'embed': 'timeF',
    'activation': 'gelu',
    'output_attention': False,
    'num_workers': 4,
    'itr': 1,
    'train_epochs': 10,
    'batch_size': 32,
    'patience': 3,
    'learning_rate': 3e-05,
    'des': 'Exp',
    'loss': 'MSE',
    'lradj': 'type1',
    'use_amp': False,
    'use_gpu': True,
    'gpu': 0,
    'use_multi_gpu': False,
    'devices': '0,1,2,3',
    'p_hidden_dims': [128, 128],
    'p_hidden_layers': 2,
    'exp_name': 'None',
    'partial_part': 0,
    'random_train': False,
    'channel_independent': False,
    'inverse': False,
    'class_strategy': 'projection',
    'target_root_path': './data/ETT-small/',
    'target_data_path': 'ETTh1.csv',
    'target_data': 'custom',
    'target_root_path_list': None,
    'target_data_path_list': None,
    'target_data_list': None,
    'exchange_attention': False,
    'root_path_list': None,
    'data_path_list': None,
    'stride_list': None,
    'stride': 1,
    'decompose_order': 'last',
    'decompose_strategy': 'one',
    'loss_type': 'mse',
    'long_embed': False,
    'decompose_type': 'default',
    'ckpt_output_path': './tmp_ckpt/large_debug_default.pth',
    'finetune_epochs': 50,
    'finetune_rate': 0.1,
    'patch_len': 96,
    'subset_rand_ratio': 1.0,
    'subset_rand_rand_ratio': 1,
    'data_type': 'custom',
    'subset_ratio': 1,
    'split': 0.9,
    'decay_fac': 0.75,
    'cos_warm_up_steps': 100,
    'cos_max_decay_steps': 60000,
    'cos_max_decay_epoch': 10,
    'cos_max': 0.0001,
    'cos_min': 2e-06,
    'freeze': 0,
    'unfrozen_list': ['norm', 'value_embedding', 'proj.'],
    'use_weight_decay': 0,
    'weight_decay': 0.01,
    'use_post_data': 0,
    'roll': True,
    'output_len': 96,
    'train_test': 1,
    'is_finetuning': 0,
    'finetuned_ckpt_path': '',
    'train_offset': 96,
    'datetime_col': None,
    'feature_cols': None,
    'target_cols': None,
    'timestamp_feature': None,
    'show_demo': True
}

args = argparse.Namespace(**argsDict)

Exp = Exp_Large_Few_Shot_Roll_Demo
exp = Exp(args)

# forecasting_model = exp.model.load_state_dict(torch.load('./checkpoints/checkpoint-example.pth'))

def load_model(args, exp):
    return exp.model

model  = load_model(args, exp)

# 将输入的list转换为tensor形式
def list_to_input_array(input_list):
    input = np.array(input_list)
    # if len(input.shape)<3:
    #     input = np.expand_dims(input, axis=0)
    #     input = np.array(input)
    # else:
    #     input = np.array(input)
    return input

def output_tensor_to_list(output):
    if len(output.shape)==3:
        output = output.reshape(output.shape[1],output.shape[2])
    return output.tolist()

# 数据集导入(dataset \ dataloader)
inference_data, inference_loader = exp._get_data(flag='test')

# 获取数据
#   输入数据：sample（672,1）
#   输出数据：outputs（768,）
#   回归标签：label（672,1）

index=1
sample_list = inference_data[index][0].tolist()
label_list = inference_data[index][1].tolist()
sample = list_to_input_array(sample_list)
label = list_to_input_array(label_list)

outputs = exp.inference(data=sample)

predict_outputs = output_tensor_to_list(outputs)



import pdb
pdb.set_trace()