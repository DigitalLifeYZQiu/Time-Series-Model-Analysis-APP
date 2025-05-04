import time

from matplotlib import pyplot as plt
from numpy import ndarray

from data_provider.data_factory import data_provider, target_data_provider
from exp.exp_basic import Exp_Basic
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import warnings
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP
from datetime import datetime

from utils.tools import attn_map

warnings.filterwarnings('ignore')


class Exp_Large_Few_Shot_Roll_Demo(Exp_Basic):

    def __init__(self, args):
        super(Exp_Large_Few_Shot_Roll_Demo, self).__init__(args)


    def _build_model(self):
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = self.model_dict[self.args.model].Model(self.args)
            model = DDP(model.cuda(), device_ids=[self.args.local_rank], find_unused_parameters=True)
        else:
            self.args.device = self.device
            model = self.model_dict[self.args.model].Model(self.args)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        if self.args.use_weight_decay:
            model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        else:
            model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion
    def test(self, setting, test=0):
        if not self.args.is_finetuning and self.args.finetuned_ckpt_path:
            print('loading model: ', self.args.finetuned_ckpt_path)
            if self.args.finetuned_ckpt_path.endswith('.pth'):
                sd = torch.load(self.args.finetuned_ckpt_path, map_location="cpu")
                self.model.load_state_dict(sd, strict=True)

            elif self.args.finetuned_ckpt_path.endswith('.ckpt'):
                if self.args.use_multi_gpu:
                    sd = torch.load(self.args.finetuned_ckpt_path, map_location="cpu")["state_dict"]
                    sd = {'module.' + k: v for k, v in sd.items()}
                    self.model.load_state_dict(sd, strict=True)
                else:
                    sd = torch.load(self.args.finetuned_ckpt_path, map_location="cpu")["state_dict"]
                    self.model.load_state_dict(sd, strict=True)
            else:
                raise NotImplementedError

        print('Model parameters: ', sum(param.numel() for param in self.model.parameters()))
        target_root_path = self.args.root_path
        target_data_path = self.args.data_path
        target_data = self.args.data


        print("=====================Testing: {}=====================".format(target_root_path + target_data_path))
        print("=====================Demo: {}=====================".format(target_root_path + target_data_path))
        test_data, test_loader = target_data_provider(self.args, target_root_path, target_data_path, target_data,
                                                      flag='test')

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/' + target_data_path + '/'
        if not os.path.exists(folder_path) and int(os.environ.get("LOCAL_RANK", "0")) == 0:
            os.makedirs(folder_path)
        mae_val = torch.tensor(0., device="cuda")
        mse_val = torch.tensor(0., device="cuda")
        count = torch.tensor(1e-5, device="cuda")
        self.model.eval()
        
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                inference_steps = self.args.output_len // self.args.pred_len
                dis = self.args.output_len - inference_steps * self.args.pred_len
                if dis != 0:
                    inference_steps += 1
                pred_y = []
                # encoder - decoder
                for j in range(inference_steps):
                    if len(pred_y) != 0:
                        batch_x = torch.cat([batch_x[:, self.args.pred_len:, :], pred_y[-1]], dim=1)
                        tmp = batch_y_mark[:, j - 1:j, :]
                        batch_x_mark = torch.cat([batch_x_mark[:, 1:, :], tmp], dim=1)

                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    outputs = outputs[:, -self.args.pred_len:, -1:]
                    f_dim = -1 if self.args.features == 'MS' else 0
                    pred_y.append(outputs[:, -self.args.pred_len:, :])
                pred_y = torch.cat(pred_y, dim=1)
                if dis != 0:
                    pred_y = pred_y[:, :-dis, :]
                # import pdb
                # pdb.set_trace()
                
                # batch_y = batch_y[:, self.args.label_len:self.args.label_len + self.args.output_len, :].to(self.device)
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                outputs = pred_y.detach().cpu()
                batch_y = batch_y.detach().cpu()
                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)

                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)

                if i % 10 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, -self.args.pred_len:, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, -self.args.pred_len:, -1], pred[0, :, -1]), axis=0)
                    
                    dir_path = folder_path + f'{self.args.output_len}/'
                    if not os.path.exists(dir_path):
                        os.makedirs(dir_path)
                    # np.save(os.path.join(dir_path, f'gt_{i}.npy'), np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0))
                    # np.save(os.path.join(dir_path, f'pd_{i}.npy'), np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0))
                    np.savez(os.path.join(dir_path, f'res_{i}.npz'), groundtruth=gt,predict=pd)
                    print(os.path.join(dir_path, f'res_{i}.npz'),"saved")

                    # if self.args.use_multi_gpu:
                    #     visual(gt, pd, os.path.join(dir_path, f'{i}_{self.args.local_rank}.pdf'))
                    # else:
                    #     visual(gt, pd, os.path.join(dir_path, f'{i}_.pdf'))
        preds = torch.cat(preds, dim=0).numpy()
        trues = torch.cat(trues, dim=0).numpy()

        # # result save
        # folder_path = './results/' + setting + '/'
        # if not os.path.exists(folder_path):
        #     os.makedirs(folder_path)


        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))

        return

    # def inference(self, data: ndarray, pred_len: int = 96) -> str:
    #     # 以该函数被调用的时间为准，创建一个文件夹
    #     folder_path = './inference_results/' + datetime.now().strftime("%Y%m%d_%H%M%S%f") + '/'
    #     if not os.path.exists(folder_path):
    #         os.makedirs(folder_path)

    #     # print('Model parameters: ', sum(param.numel() for param in self.model.parameters()))
    #     # 存储一个data的副本
    #     # original_data = data.copy() # [seq_len, 1]
    #     # import pdb
    #     # pdb.set_trace()

    #     patch_len = self.args.patch_len
    #     # data的形状为[L, 1]，将L补齐到patch_len的整数倍
    #     seq_len = data.shape[0]
    #     pad_len = (patch_len - seq_len % patch_len) % patch_len
    #     if seq_len % patch_len != 0:
    #         data = np.concatenate((np.zeros((pad_len, 1)), data), axis=0)

    #     # 在data前面挤压一维，变成[1, L, 1]
    #     data = data[np.newaxis, :, :] # [1, seq_len + pad_len, 1]
    #     data = torch.tensor(data, dtype=torch.float32).to(self.device)
    #     inference_steps = pred_len // patch_len
    #     dis = inference_steps * patch_len - pred_len
    #     if dis != 0:
    #         inference_steps += 1
    #         dis = dis + patch_len

    #     # encoder - decoder
    #     self.model.eval()
    #     with torch.no_grad():
    #         for j in range(inference_steps):
    #             if self.args.output_attention:
    #                 outputs, attns = self.model(data, None, None, None)
    #             else:
    #                 outputs = self.model(data, None, None, None)
    #             data = torch.cat([data, outputs[:, -patch_len:, :]], dim=1)

    #     outputs = torch.cat([data[:, :patch_len, :], outputs], dim=1)
    #     if dis != 0:
    #         data = data[:, :-dis, :]
    #         outputs = outputs[:, :-dis, :]

    #     data = data.detach().cpu().numpy()

    #     data = data.squeeze(0) # [seq_len + pad_len + pred_len, 1]

    #     pred_data = data[pad_len:, 0]
    #     return pred_data

    def inference(self, data: ndarray, pred_len: int = 96) -> str:
            # 以该函数被调用的时间为准，创建一个文件夹
            folder_path = './inference_results/' + datetime.now().strftime("%Y%m%d_%H%M%S%f") + '/'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            # print('Model parameters: ', sum(param.numel() for param in self.model.parameters()))
            # 存储一个data的副本
            original_data = data.copy() # [seq_len, 1]

            patch_len = self.args.patch_len
            # data的形状为[L, 1]，将L补齐到patch_len的整数倍
            seq_len = data.shape[0]
            pad_len = (patch_len - seq_len % patch_len) % patch_len
            if seq_len % patch_len != 0:
                data = np.concatenate((np.zeros((pad_len, 1)), data), axis=0)

            # 在data前面挤压一维，变成[1, L, 1]
            data = data[np.newaxis, :, :] # [1, seq_len + pad_len, 1]
            data = torch.tensor(data, dtype=torch.float32).to(self.device)
            print(f"data shape:{data.shape}")
            inference_steps = pred_len // patch_len
            dis = inference_steps * patch_len - pred_len
            if dis != 0:
                inference_steps += 1
                dis = dis + patch_len

            # encoder - decoder
            self.model.eval()
            with torch.no_grad():
                for j in range(inference_steps):
                    if self.args.output_attention:
                        outputs, attns = self.model(data, None, None, None)
                    else:
                        outputs = self.model(data, None, None, None)
                    data = torch.cat([data, outputs[:, -patch_len:, :]], dim=1)

            outputs = torch.cat([data[:, :patch_len, :], outputs], dim=1)
            if dis != 0:
                data = data[:, :-dis, :]
                outputs = outputs[:, :-dis, :]

            data = data.detach().cpu().numpy()

            data = data.squeeze(0) # [seq_len + pad_len + pred_len, 1]

            pred_data = data[pad_len:, 0]

            # 保存原始数据和预测数据
            np.save(os.path.join(folder_path, 'original_data.npy'), original_data)
            np.save(os.path.join(folder_path, 'pred_data.npy'), pred_data)

            # plt.figure(figsize=((seq_len + pred_len) // patch_len * 5, 5))

            # # # 如果绘制重构部分
            # # if self.args.output_reconstruction:
            # #     pred_data = np.concatenate((original_data, pred_data), axis=0)

            # # 绘制图像
            # if self.args.output_reconstruction:
            #     outputs = outputs.detach().cpu().numpy().squeeze(0)[pad_len:, 0]
            #     plt.plot(np.arange(seq_len + pred_len), outputs, label='Prediction', c='dodgerblue', linewidth=2)
            # else:
            #     plt.plot(np.arange(seq_len + pred_len), pred_data, label='Prediction', c='dodgerblue', linewidth=2)
            # plt.plot(np.arange(seq_len), original_data, label='GroundTruth', c='tomato', linewidth=2)

            # # 添加图例
            # plt.legend()

            # # 保存图片
            # plt.savefig(os.path.join(folder_path, 'inference_result.pdf'), bbox_inches='tight')

            # if self.args.output_attention:
            #     attn = attns[0].cpu().numpy()[0, 0, :, :]
            #     attn_map(attn, os.path.join(folder_path, f'attn.pdf'))

            return folder_path
    
    def fast_inference(self, data: ndarray, pred_len: int = 96) -> str:
        # 以该函数被调用的时间为准，创建一个文件夹
        folder_path = './inference_results/' + datetime.now().strftime("%Y%m%d_%H%M%S%f") + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        # print('Model parameters: ', sum(param.numel() for param in self.model.parameters()))
        # 存储一个data的副本
        # original_data = data.copy() # [seq_len, 1]
        # import pdb
        # pdb.set_trace()
        
        patch_len = self.args.patch_len
        # data的形状为[L, 1]，将L补齐到patch_len的整数倍
        seq_len = data.shape[0]
        pad_len = (patch_len - seq_len % patch_len) % patch_len
        if seq_len % patch_len != 0:
            data = np.concatenate((np.zeros((pad_len, 1)), data), axis=0)
        
        # 在data前面挤压一维，变成[1, L, 1]
        data = data[np.newaxis, :, :]  # [1, seq_len + pad_len, 1]
        data = torch.tensor(data, dtype=torch.float32).to(self.device)
        inference_steps = pred_len // patch_len
        dis = inference_steps * patch_len - pred_len
        if dis != 0:
            inference_steps += 1
            dis = dis + patch_len
        
        # encoder - decoder
        self.model.eval()
        with torch.no_grad():
            for j in range(inference_steps):
                if self.args.output_attention:
                    outputs, attns = self.model(data, None, None, None)
                else:
                    outputs = self.model(data, None, None, None)
                data = torch.cat([data, outputs[:, -patch_len:, :]], dim=1)
        
        outputs = torch.cat([data[:, :patch_len, :], outputs], dim=1)
        if dis != 0:
            data = data[:, :-dis, :]
            outputs = outputs[:, :-dis, :]
        
        data = data.detach().cpu().numpy()
        
        data = data.squeeze(0)  # [seq_len + pad_len + pred_len, 1]
        
        pred_data = data[pad_len:, 0]
        return pred_data

