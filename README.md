# Gas Forcasting Inference API

This is an Inference API repo for Gas forecasting mission.

## Usage

1. Install Python 3.10 with necessary requirements. If you are using `Anaconda`, here is an example:

```shell
conda create -n alumina python=3.10 jupyter notebook
pip install -r requirements.txt
```

2. Prepare Checkpoint. Please download the pretrained checkpoint: [Timer_forecast_1.0.ckpt](https://pan.baidu.com/s/1Wj_1_qMgyLNLOSUFZK3weg?pwd=r8i1#list/path=%2F)
and place in route `checkpoints`. Feel free to add your own checkpoint in this route.
3. Prepare Data. Place the `Gas dataset` in route `dataset`.
4. Run inference. The forecasting script is briefed in file `inference_example.py`

## Contact

If you have any questions or suggestions, feel free to contact us:

- Yunzhong Qiu (Master student, qiuyz24@mails.tsinghua.edu.cn)