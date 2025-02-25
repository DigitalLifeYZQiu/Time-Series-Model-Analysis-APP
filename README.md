# Time-Series-Model-Analysis-APP

This is a hosted demo for time series model analysis.

## Usage

### 1. Prepare Environment.
Install Python $\ge$ 3.10 with necessary requirements. If you are using `Anaconda`, here is an example:

```shell
conda create -n alumina python=3.10 jupyter notebook
pip install -r requirements.txt
```

### 2. Prepare Checkpoint. 
Please download the pretrained LTM checkpoints: ([Tsinghua Cloud](https://cloud.tsinghua.edu.cn/d/7539c66be03f49c3afe4/))
and specify route of `model_list` in `./app.py` . Feel free to add your own checkpoint in this route.
```python
model_list={
    'Timer1': '/data/qiuyunzhong/CKPT/Timer_forecast_1.0.ckpt',
    'Timer-UTSD': '/data/qiuyunzhong/CKPT/Building_timegpt_d1024_l8_p96_n64_new_full.ckpt',
    'Timer-LOTSA': '/data/qiuyunzhong/CKPT/Large_timegpt_d1024_l8_p96_n64_new_full.ckpt'
}
```
### 3. Prepare Data. 
The default data loading route is `dataset`.

### 4. Run inference. 
The inference process `exp.inference` can be tested by `./test_inference.py`. Note that the default parameter set `args` might not fit in
to all scenarios, users need to modify the `args` before creating the experiment class `exp`. Here is a brief code example:
```python
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
    
    # * Change default model loading directory (Other params can be changed similarly)
    args.ckpt_path='/data/qiuyunzhong/CKPT/Building_timegpt_d1024_l8_p96_n64_new_full.ckpt'
    
    # * Use the specified args to create experiment class
    exp = Exp_Large_Few_Shot_Roll_Demo(args)
    
    # * Perform inference (All inference results are saved to target directory as npy matrix format)
    dir = exp.inference(test_data, pred_len)
    print(dir)
```
Meanwhile, there is another fast inference process `exp.fast_inference` for directly getting the inference result.
An using example is provided in `fast_inference_example.py`. You need to specify the arguments of model and data before running:
```python
'root_path': '<your-root-path>',
'data_path': '<your-data-path>',
'ckpt_path': '<your-checkpoint-path>',
'<other-necessary-arguments>': '<your-other-necessary-arguments>'
```
### 5. Run APP. 
An interface is created by `Gradio`, run the following scripts to use the interface.
```shell
# Method 1 (vanilla start-up)
python app.py

# Method 2 (Dynamic start-up)
gradio app.py
```
## Contact

If you have any questions or suggestions, feel free to contact us:

- Yunzhong Qiu (Master student, qiuyz24@mails.tsinghua.edu.cn)