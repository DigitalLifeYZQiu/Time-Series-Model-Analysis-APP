import os
import time
import numpy as np
import argparse
import random
import torch
import gradio as gr
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pandas as pd
from io import StringIO
from test_inference import test_inference
from exceptions import MineException
from exp.exp_large_few_shot_roll_demo import Exp_Large_Few_Shot_Roll_Demo
import altair as alt
import copy

# ! Note: Name model label by ${model_name}_${seq_len}_${pred_len}
model_list={
    'P_90_15': 'ckpt_default/long_term_forecast_ETTh1_sl90_pl15_PatchTST_ETTh1_ftMS_sl90_ll48_pl15_dm512_nh8_el3_dl1_df1024_expand2_dc4_fc3_ebtimeF_dtTrue_test_0/checkpoint.pth',
    'P_180_15': 'ckpt_default/long_term_forecast_ETTh1_sl180_pl15_PatchTST_ETTh1_ftMS_sl180_ll48_pl15_dm512_nh8_el3_dl1_df1024_expand2_dc4_fc3_ebtimeF_dtTrue_test_0/checkpoint.pth'
}

'''
load checkpoint once
'''
parser = argparse.ArgumentParser(description='TimesNet')

# basic config
parser.add_argument('--task_name', type=str, default='long_term_forecast')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--model', type=str, default='Timer')
parser.add_argument('--ckpt_path', type=str, default='checkpoints/Building_timegpt_d1024_l8_p96_n64_new_full.ckpt')

# model define
parser.add_argument('--patch_len', type=int, default=15)
parser.add_argument('--d_model', type=int, default=1024, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=8, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--factor', type=int, default=3, help='attn factor')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true')
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length')
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
parser.add_argument('--enc_in', type=int, default=1, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=1, help='decoder input size')
parser.add_argument('--c_out', type=int, default=1, help='output size')
parser.add_argument('--stride', type=int, default=15, help='stride')

# GPU
parser.add_argument('--use_gpu', type=bool, default=False, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')


args = parser.parse_args()
args.model = 'PatchTST'
args.patch_len= 15
args.stride = 15
args.seq_len = 90
args.label_len = 48
args.pred_len = 15
args.e_layers = 3
args.d_model = 512
args.d_ff = 1024
fix_seed = args.seed
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)
args.ckpt_path='random'
exp = Exp_Large_Few_Shot_Roll_Demo(args)
print(f"args.seq_len={args.seq_len}, args.pred_len={args.pred_len}")
# args = parser.parse_args()
# args.model = 'PatchTST'
# args.seq_len = 180
# args.label_len = 48
# args.pred_len = 15
# args.e_layers = 3
# args.d_model = 512
# args.d_ff = 1024
# args.label_len
# exp = Exp_Large_Few_Shot_Roll_Demo(args)

def update_model_selection(choice):
    args.ckpt_path=model_list.get(choice)
    
    print("model selection:", choice)
    print("ckpt path:",args.ckpt_path)
    args.seq_len = int(choice.split("_")[1])
    args.pred_len = int(choice.split("_")[2])
    exp = Exp_Large_Few_Shot_Roll_Demo(args)
    print(f"seq_len={args.seq_len}, pred_len={args.pred_len}")
    return f"load model {choice} from directory {model_list.get(choice)}", args.seq_len, args.pred_len


def load_data(temp_file,progress=gr.Progress()):
    '''
    Descriptions:
        Original data loading entrance to get original data and data columns.
    Inputs: 
        .csv file path -> string format
    Outputs: 
        csv file content -> pd.Dataframe format
        csv file column list -> gr.Dropdown() format
    Remarks:
        I downloaded the input data to "temp_file.csv" during this function for easier data accessing.
    '''
    progress(0, desc="Starting...")
    df = pd.read_csv(temp_file)
    df.to_csv('temp_file.csv', index=False)
    columns = df.columns.tolist()
    plot_target=gr.Dropdown(choices=columns,allow_custom_value=True,interactive=True)
    return plot_target

def load_table():
    df = pd.read_csv("temp_file.csv")
    return df

def plot_column(choice):
    '''
    Descriptions:
        Plot function for target column using Matplotlib. (beta)
    Inputs: 
        choice of target column -> string format
    Outputs: 
        Matplotlib image format
    '''
    df = pd.read_csv("temp_file.csv")
    df_choice = df[choice]
    image=plt.figure()
    plt.title('Value over Time:{}'.format(choice))
    plt.plot(df_choice)
    return image

def plot_column_lineplt(choice):
    '''
    Descriptions:
        Plot function for target column using gr.LinePlot(). (beta)
    Inputs: 
        choice of target column -> string format
    Outputs: 
        gr.LinePlot() input -> pd.DataFrame format
    Remarks:
        Remember to use "alt.data_transformers.disable_max_rows()" to disable the 5000 rows checking mechanism
    '''
    df = pd.read_csv("temp_file.csv")
    df_choice = np.array(df[choice])
    length = len(df_choice)
    alt.data_transformers.disable_max_rows()
    df_new = pd.DataFrame({
        'x' : np.arange(length),
        'y': df_choice[:length]
    })
    return df_new

def plot_column_plotly(choice):
    '''
    Descriptions:
        Plot function for target column using Plotly.
    Inputs: 
        choice of target column -> string format
    Outputs: 
        Plotly image format
    '''
    df = pd.read_csv("temp_file.csv")
    df_choice = np.array(df[choice])
    length = len(df_choice)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.arange(length),y=df_choice,mode='lines',name="Ground Truth"))
    fig.update_layout(title='Value over Time:{}'.format(choice),xaxis_title='Timepoint',yaxis_title='{} value'.format(choice))
    return fig

def inference_plotly_fast(choice,start_index,length_index,pred_len,show_GT,model_choice):
    '''
    Descriptions:
        Inference function for target column using Plotly.
    Input: 
        choice of target column -> string format
        start_index -> string format
        length_index -> string format
        pred_len -> string format
        show_GT -> bool format
    Output: 
        result_dir -> string format
        Matplotlib image format
    '''
    start_index = int(start_index)
    length_index = int(length_index)
    pred_len = int(pred_len)
    df = pd.read_csv("temp_file.csv")
    df_choice = df[choice]
    length = len(df_choice)
    if model_choice in model_list:
        args.ckpt_path=args.ckpt_path=model_list.get(model_choice)
    else:
        args.ckpt_path=model_choice

    exp = Exp_Large_Few_Shot_Roll_Demo(args)

    if start_index<0 or start_index>length-1:
        raise MineException("起始位置错误")
    if length_index<0:
        raise MineException("长度错误")
    if length_index>length-1:
        raise MineException("终止位置错误")
    if pred_len<0:
        raise MineException("预测长度错误")
    test_data = df_choice.to_numpy().reshape((length,1))
    result_dir = exp.inference(test_data[int(start_index):int(start_index)+int(length_index),:],int(pred_len))
    original_data = np.load(os.path.join(result_dir, 'original_data.npy')).squeeze()
    pred_data = np.load(os.path.join(result_dir, 'pred_data.npy')).squeeze()
    seq_len = int(length_index)
    pred_len = int(pred_len)
    end_index = int(start_index)+int(length_index)
    fig = go.Figure()
    if show_GT:
        GT_len = pred_len
        try:
            if end_index + pred_len > length:
                GT_len = length-end_index
                raise MineException("预测范围超出GroundTruth限制")
        except Exception as result:
            GT_len = length-end_index
            print(result)
        groundtruth_data = test_data[end_index-1:end_index+GT_len].squeeze()
        fig.add_trace(go.Scatter(x=np.arange(seq_len + pred_len),y=pred_data,mode='lines',name="Prediction",line = dict(color='royalblue', width=1.5)))
        fig.add_trace(go.Scatter(x=np.arange(seq_len),y=original_data,mode='lines',name="Original Data",line = dict(color='firebrick', width=1.5)))
        if GT_len>0:
            fig.add_trace(go.Scatter(x=np.arange(seq_len-1,seq_len + GT_len),y=groundtruth_data,mode='lines',name="Ground Truth Data",line = dict(color='purple', width=1.5)))
        fig.update_layout(title='Two Line Plots', xaxis_title='Timepoint', yaxis_title='{} value'.format(choice))
    else:
        fig.add_trace(go.Scatter(x=np.arange(seq_len + pred_len),y=pred_data,mode='lines',name="Prediction",line = dict(color='royalblue', width=1.5)))
        fig.add_trace(go.Scatter(x=np.arange(seq_len),y=original_data,mode='lines',name="Original Data",line = dict(color='firebrick', width=1.5)))
        fig.update_layout(title='Two Line Plots', xaxis_title='Timepoint', yaxis_title='{} value'.format(choice))
    return result_dir,fig

def getIOlen(seq_len, pred_len):
    return seq_len, pred_len
    
    


'''
The Timer Demo App in Gradio
'''
with gr.Blocks() as demo:
    gr.Markdown("<div align='center' ><font size='70'><b>时间序列模型应用平台</b></font></div>")
    with gr.Tab("接口调用"):
        gr.Markdown("# 数据上传（仅csv格式）")
        gr.Markdown("请在此处上传需要分析的文件，上传后的文件将自动在下面的表格中展示。")
        upload_button = gr.File(label="请在此上传文件")
        table_button = gr.Button(value = "展示表格")
        table = gr.Dataframe(interactive=True)

        gr.Markdown("# 目标变量选择")
        gr.Markdown("上传文件中的所有变量如下，请选择要分析的变量。你可以在图中查看所选变量的变化趋势。")
        plot_target = gr.Dropdown(allow_custom_value=True,interactive=True,label="请选择推理目标变量")
        file = upload_button.upload(fn=load_data, inputs=upload_button, outputs=[plot_target], api_name="upload_csv")
        table_button.click(fn=load_table, inputs=None, outputs=table)

        colplot= gr.Plot(label="目标变量可视化结果")
        plot_target.change(fn=plot_column_plotly, inputs=[plot_target],outputs=colplot)

        gr.Markdown("# 目标变量推理")
        gr.Markdown("请指定目标变量分析范围起始位置与终止位置，并给出想要预测的长度。准备妥当后，点击按钮进行推理。推理结果将在图中展示。")
        gr.Markdown("当使用 PatchTST 时，仅可指定起始位置，输入长度为模型seq_len，输出长度为模型pred_len。更换模型后输入长度与推理长度将**自动更新**。")
        with gr.Row():
            start = gr.Textbox(value=0,label="起始位置")
            length = gr.Textbox(value=90,label="输入长度")
            pred = gr.Textbox(value=15,label="推理长度")
            show_GT = gr.Checkbox(value=True,label="选中以展示真值")

            # start.change(
            #     fn = getIOlen,
            #     inputs=[args.seq_len, args.pred_len],
            #     outputs = [length, pred]
            # )
        
        # gr.Markdown("# Timer 模型选择")
        # gr.Markdown("## 目前支持的模型\n - Timer1\n - Timer-UTSD\n - Timer-LOTSA")
        # model_selection=gr.Dropdown(choices=list(model_list.keys()),allow_custom_value=True,interactive=True,value='random',label="请选择内置模型")
        # model_selection_text = gr.Textbox(label="模型选择结果")
        # model_selection.change(fn=update_model_selection, inputs=[model_selection], outputs=model_selection_text)
        
        # inference_button = gr.Button(value="Timer进行推理")
        # result_dir = gr.Textbox(label="结果文件位置")
        # result_img = gr.Plot(label="推理结果可视化")
        # inference_button.click(
        #     fn=inference_plotly_fast,
        #     inputs=[plot_target, start, length, pred, show_GT, model_selection],
        #     outputs=[result_dir, result_img]
        # )
        
        gr.Markdown("# PatchTST 模型选择")
        gr.Markdown("## 目前支持的模型\n "
                    "- PatchTST1: P_90_15\n" \
                    "- PatchTST2: P_180_15\n "
                    "- PatchTST3: Other PatchTST Checkpoint")
        patch_model_selection = gr.Dropdown(choices=list(model_list.keys()), allow_custom_value=True, interactive=True,
                                      value='random', label="请选择内置模型")
        patch_model_selection_text = gr.Textbox(label="模型选择结果")
        patch_model_selection.change(fn=update_model_selection, inputs=[patch_model_selection], outputs=[patch_model_selection_text, length, pred])

        
        patch_inference_button = gr.Button(value = "PatchTST进行推理")
        patch_result_dir = gr.Textbox(label="结果文件位置")
        patch_result_img = gr.Plot(label="推理结果可视化")
        patch_inference_button.click(
            fn=inference_plotly_fast,
            inputs=[plot_target,start,length,pred,show_GT,patch_model_selection],
            outputs=[patch_result_dir,patch_result_img]
        )

    # with gr.Tab("可视化展示"):
    #     with gr.Column():
    #         with gr.Row():
    #             all = gr.Textbox(len(file_list),label="测试集样本数量")
    #     map = gr.Plot()
    #     btn = gr.Button("RUN")
    #     btn.click(fn=draw_all,inputs=[],outputs=map)
    
demo.queue()
demo.launch(root_path="http://anylearn.nelbds.cn:81/hdemo/tsanalysis")
