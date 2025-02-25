import argparse
import random

import numpy as np
import torch

from exp.exp_large_few_shot_roll_demo import Exp_Large_Few_Shot_Roll_Demo


def test_inference(test_data,pred_len):
    parser = argparse.ArgumentParser(description='TimesNet')

    # basic config
    parser.add_argument('--task_name', type=str, default='large_finetune')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--model', type=str, default='Timer')
    parser.add_argument('--ckpt_path', type=str, default='checkpoints/Timer_forecast_1.0.ckpt')

    # model define
    parser.add_argument('--patch_len', type=int, default=96)
    parser.add_argument('--d_model', type=int, default=1024, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=16, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=8, help='num of encoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--factor', type=int, default=3, help='attn factor')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true')
    parser.add_argument('--output_reconstruction', action='store_true')
    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')


    args = parser.parse_args()
    fix_seed = args.seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    # # 设置序列的大小
    # size = (700, 1)
    # # 生成0到2π的序列，用于正弦波的x轴值
    # x = np.linspace(0, 50 * np.pi, size[0])
    # # 将生成的数据转换为指定的大小
    # test_data = (0.5 * np.sin(x)).reshape(size)
    
    # * Change default model loading directory
    args.ckpt_path='/data/qiuyunzhong/CKPT/Building_timegpt_d1024_l8_p96_n64_new_full.ckpt'

    exp = Exp_Large_Few_Shot_Roll_Demo(args)
    dir = exp.inference(test_data, pred_len)
    print(dir)

    # pass
    return dir

def generate_sin(seq_len, period):
    x = np.linspace(0, seq_len / period * np.pi * 2, seq_len)
    return (0.5 * np.sin(x)).reshape(seq_len, 1)

def generate_linear(seq_len):
    return np.linspace(0, 1, seq_len).reshape(seq_len, 1)

def generate_zero(seq_len):
    return np.zeros((seq_len, 1))

def generate_random(seq_len):
    return np.random.rand(seq_len, 1)

def generate_reminder_1(seq_len):
    #     -           -           -           -
    #   /   \       /   \       /   \       /   \
    # --     -------     -------     -------     -----
    data = np.zeros((seq_len, 1))
    interval = 384
    width = 24
    for i in range(width, seq_len - width, interval):
        # 以 -i 为中心生成一个正弦波
        x = np.linspace(0, np.pi, 2 * width)
        wave = (0.5 * np.sin(x)).reshape(2 * width, 1)
        data[i - width:i + width, :] = 1
    return data

if __name__ == '__main__':
    np.random.seed(14)
    seq_len = 2000
    # test_data = generate_linear(seq_len)
    #
    pred_len=600
    test_data = generate_reminder_1(seq_len)
    test_inference(test_data,pred_len)