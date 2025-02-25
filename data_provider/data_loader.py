import os
import numpy as np
import pandas as pd
import glob
import re
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings

warnings.filterwarnings('ignore')


class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0) 

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_PEMS(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        # info
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        data_file = os.path.join(self.root_path, self.data_path)
        data = np.load(data_file, allow_pickle=True)
        data = data['data'][:, :, 0]

        train_ratio = 0.6
        valid_ratio = 0.2
        train_data = data[:int(train_ratio * len(data))]
        valid_data = data[int(train_ratio * len(data)): int((train_ratio + valid_ratio) * len(data))]
        test_data = data[int((train_ratio + valid_ratio) * len(data)):]
        total_data = [train_data, valid_data, test_data]
        data = total_data[self.set_type]

        if self.scale:
            self.scaler.fit(train_data)
            data = self.scaler.transform(data)

        df = pd.DataFrame(data)
        df = df.fillna(method='ffill', limit=len(df)).fillna(method='bfill', limit=len(df)).values

        self.data_x = df
        self.data_y = df

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = torch.zeros((seq_x.shape[0], 4))
        seq_y_mark = torch.zeros((seq_x.shape[0], 4))

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Solar(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        # info
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = []
        with open(os.path.join(self.root_path, self.data_path), "r", encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip('\n').split(',')
                data_line = np.stack([float(i) for i in line])
                df_raw.append(data_line)
        df_raw = np.stack(df_raw, 0)
        df_raw = pd.DataFrame(df_raw)

        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_valid = int(len(df_raw) * 0.1)
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_valid, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        df_data = df_raw.values

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data)
            data = self.scaler.transform(df_data)
        else:
            data = df_data

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = torch.zeros((seq_x.shape[0], 4))
        seq_y_mark = torch.zeros((seq_x.shape[0], 4))

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class UCRAnomalyloader(Dataset):
    def __init__(self, root_path, data_path, seq_len, patch_len, flag="train"):
        self.root_path = root_path
        self.data_path = data_path
        self.seq_len = seq_len
        self.input_len = seq_len - patch_len
        self.patch_len = patch_len
        self.flag = flag
        self.stride = 1 if self.flag == "train" else self.seq_len - 2 * self.patch_len
        self.dataset_file_path = os.path.join(self.root_path, self.data_path)
        data_list = []
        assert self.dataset_file_path.endswith('.txt')
        try:
            with open(self.dataset_file_path, "r", encoding='utf-8') as f:
                for line in f.readlines():
                    line = line.strip('\n').split(',')
                    data_line = np.stack([float(i) for i in line])
                    data_list.append(data_line)
            self.data = np.stack(data_list, 0)
        except ValueError:
            # the dataset is a single line
            with open(self.dataset_file_path, "r", encoding='utf-8') as f:
                for line in f.readlines():
                    line = line.strip('\n').split(',')
                    data_line = np.stack([float(i) for i in line[0].split()])
            self.data = data_line
            self.data = np.expand_dims(self.data, axis=1)

        self.border = self.find_border_number(self.data_path)
        self.scaler = StandardScaler()
        self.scaler.fit(self.data[:self.border])
        self.data = self.scaler.transform(self.data)
        if self.flag == "train":
            self.data = self.data[:self.border]
        else:
            self.data = self.data[self.border - self.patch_len:]

    def find_border_number(self, input_string):
        parts = input_string.split('_')

        if len(parts) < 3:
            return None

        border_str = parts[-3]

        try:
            border = int(border_str)
            return border
        except ValueError:
            return None

    def __len__(self):
        return (self.data.shape[0] - self.seq_len) // self.stride + 1

    def __getitem__(self, index):
        index = index * self.stride
        return self.data[index:index + self.seq_len, :]


class CIDatasetBenchmark(Dataset):
    # Fair Benckmarks from Time-Series-Library: https://github.com/thuml/Time-Series-Library
    # And iTransformer: https://github.com/thuml/iTransformer/blob/main/data_provider/data_loader.py
    # The Split Ratio for Testing is Fixed to Keep Consistent with Previous Benchmarks
    # Including: ETT, ECL, Weather, Traffic, PEMS, SolarEnergy
    def __init__(self, root_path='dataset', flag='train', input_len=None, pred_len=None,
                 data_type='custom', scale=True, timeenc=1, freq='h', stride=1, subset_ratio=1.0, subset_rand_ratio=1.0,
                 use_post_data=False, train_offset=0):
        self.subset_rand_ratio = subset_rand_ratio
        # size [seq_len, label_len, pred_len]
        # info
        self.input_len = input_len
        self.pred_len = pred_len
        self.seq_len = input_len + pred_len
        self.timeenc = timeenc
        self.scale = scale
        self.subset_ratio = subset_ratio
        self.use_post_data = use_post_data
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.data_type = data_type
        if self.set_type == 0:
            self.internal = int(1 // self.subset_rand_ratio)
        else:
            self.internal = 1

        self.root_path = root_path
        self.dataset_name = self.root_path.split('/')[-1].split('.')[0]
        self.train_offset = train_offset

        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        dataset_file_path = self.root_path
        if dataset_file_path.endswith('.csv'):
            # df_raw = pd.read_csv(dataset_file_path)
            df_raw = pd.read_csv(dataset_file_path,encoding='GB2312')
            # import pdb
            # pdb.set_trace()
        elif dataset_file_path.endswith('.txt'):
            df_raw = []
            with open(dataset_file_path, "r", encoding='utf-8') as f:
                for line in f.readlines():
                    line = line.strip('\n').split(',')
                    data_line = np.stack([float(i) for i in line])
                    df_raw.append(data_line)
            df_raw = np.stack(df_raw, 0)
            df_raw = pd.DataFrame(df_raw)

        elif dataset_file_path.endswith('.npz'):
            data = np.load(dataset_file_path, allow_pickle=True)
            data = data['data'][:, :, 0]
            df_raw = pd.DataFrame(data)
        else:
            raise ValueError('Unknown data format: {}'.format(dataset_file_path))

        if self.data_type == 'custom':
            data_len = len(df_raw)
            num_train = int(data_len * 0.7)
            num_test = int(data_len * 0.2)
            num_vali = data_len - num_train - num_test
            border1s = [0, num_train - self.input_len, data_len - num_test - self.input_len]
            border2s = [num_train, num_train + num_vali, data_len]
        elif self.data_type == 'ETTh' or self.data_type == 'ETTh1' or self.data_type == 'ETTh2':
            border1s = [0, 12 * 30 * 24 - self.input_len, 12 * 30 * 24 + 4 * 30 * 24 - self.input_len]
            border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        elif self.data_type == 'ETTm' or self.data_type == 'ETm1' or self.data_type == 'ETTm2':
            border1s = [0, 12 * 30 * 24 * 4 - self.input_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.input_len]
            border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        elif self.data_type == 'Solar':
            num_train = int(len(df_raw) * 0.7)
            num_test = int(len(df_raw) * 0.2)
            num_valid = int(len(df_raw) * 0.1)
            border1s = [0, num_train - self.input_len, len(df_raw) - num_test - self.input_len]
            border2s = [num_train, num_train + num_valid, len(df_raw)]
        elif self.data_type == 'PEMS':
            data_len = len(df_raw)
            num_train = int(data_len * 0.6)
            num_test = int(data_len * 0.2)
            num_vali = data_len - num_train - num_test
            border1s = [0, num_train - self.input_len, data_len - num_test - self.input_len]
            border2s = [num_train, num_train + num_vali, data_len]
        else:
            data_len = len(df_raw)
            num_train = int(data_len * 0.7)
            num_test = int(data_len * 0.2)
            num_vali = data_len - num_train - num_test
            border1s = [0, num_train - self.input_len, data_len - num_test - self.input_len]
            border2s = [num_train, num_train + num_vali, data_len]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        if self.set_type == 0:
            if self.use_post_data:
                border1 = int((border2 - self.input_len - self.pred_len - self.train_offset) * (1 - self.subset_ratio))
            else:
                border2 = int((border2 - self.input_len - self.pred_len - self.train_offset) * self.subset_ratio + self.input_len + self.pred_len)

            border1 += self.train_offset

        if isinstance(df_raw[df_raw.columns[0]][2], str):
            data = df_raw[df_raw.columns[1:]].values
        else:
            data = df_raw.values

        if self.scale:
            train_data = data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data)
            data = self.scaler.transform(data)

        if self.timeenc == 0:
            df_stamp = df_raw[['date']][border1:border2]
            df_stamp['date'] = pd.to_datetime(df_stamp.date)
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            if isinstance(df_raw[df_raw.columns[0]][2], str):
                data_stamp = time_features(pd.to_datetime(pd.to_datetime(df_raw.date).values), freq='h')
                data_stamp = data_stamp.transpose(1, 0)
            else:
                data_stamp = np.zeros((len(df_raw), 4))
        else:
            raise ValueError('Unknown timeenc: {}'.format(self.timeenc))

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
        self.n_var = self.data_x.shape[-1]
        self.n_timepoint = len(self.data_x) - self.input_len - self.pred_len + 1

    def __getitem__(self, index):
        if self.set_type == 0:
            index = index * self.internal
        c_begin = index // self.n_timepoint  # select variable
        s_begin = index % self.n_timepoint  # select start time
        s_end = s_begin + self.input_len
        r_begin = s_end
        r_end = r_begin + self.pred_len
        seq_x = self.data_x[s_begin:s_end, c_begin:c_begin + 1]
        seq_y = self.data_y[r_begin:r_end, c_begin:c_begin + 1]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        if self.set_type == 0:
            return int(self.n_var * self.n_timepoint * self.subset_rand_ratio)
        else:
            return int(self.n_var * self.n_timepoint)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class AluminaMSDataset(Dataset):
    # Fair Benckmarks from Time-Series-Library: https://github.com/thuml/Time-Series-Library
    # And iTransformer: https://github.com/thuml/iTransformer/blob/main/data_provider/data_loader.py
    # The Split Ratio for Testing is Fixed to Keep Consistent with Previous Benchmarks
    # Including: ETT, ECL, Weather, Traffic, PEMS, SolarEnergy

    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()


    def __init__(self, feature_cols, target_cols, root_path='dataset', flag='train', input_len=None, label_len=None, pred_len=None,
                 data_type='custom', scale=True, timeenc=1, freq='h', stride=1, subset_ratio=1.0, subset_rand_ratio=1.0,
                 use_post_data=False, train_offset=0):
        self.subset_rand_ratio = subset_rand_ratio
        # size [seq_len, label_len, pred_len]
        # info
        self.input_len = input_len
        self.pred_len = pred_len
        self.seq_len = input_len + pred_len
        self.label_len = label_len
        self.timeenc = timeenc
        self.scale = scale
        self.subset_ratio = subset_ratio
        self.use_post_data = use_post_data
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.data_type = data_type
        if self.set_type == 0:
            self.internal = int(1 // self.subset_rand_ratio)
        else:
            self.internal = 1

        self.root_path = root_path
        self.dataset_name = self.root_path.split('/')[-1].split('.')[0]
        self.train_offset = train_offset

        self.feature_cols = feature_cols.split(',')
        self.target_cols = target_cols.split(',')

        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()

        dataset_file_path = self.root_path
        if dataset_file_path.endswith('.csv'):
            df_raw = pd.read_csv(dataset_file_path)
            # df_raw = pd.read_csv(dataset_file_path,encoding='GB2312')
            df_raw = df_raw[self.feature_cols]
            data_features = df_raw[self.feature_cols]
            data_target = df_raw[self.target_cols]
            # self.feature_scaler.fit(data_features)
            # self.target_scaler.fit(data_target)
            # self.data_features = self.feature_scaler.transform(data_features)
            # self.data_target = self.target_scaler.transform(data_target)
            # import pdb
            # pdb.set_trace()

        elif dataset_file_path.endswith('.txt'):
            df_raw = []
            with open(dataset_file_path, "r", encoding='utf-8') as f:
                for line in f.readlines():
                    line = line.strip('\n').split(',')
                    data_line = np.stack([float(i) for i in line])
                    df_raw.append(data_line)
            df_raw = np.stack(df_raw, 0)
            df_raw = pd.DataFrame(df_raw)

        elif dataset_file_path.endswith('.npz'):
            data = np.load(dataset_file_path, allow_pickle=True)
            data = data['data'][:, :, 0]
            df_raw = pd.DataFrame(data)
        else:
            raise ValueError('Unknown data format: {}'.format(dataset_file_path))

        if self.data_type == 'custom':
            data_len = len(df_raw)
            num_train = int(data_len * 0.7)
            num_test = int(data_len * 0.2)
            num_vali = data_len - num_train - num_test
            border1s = [0, num_train - self.input_len, data_len - num_test - self.input_len]
            border2s = [num_train, num_train + num_vali, data_len]
        elif self.data_type == 'ETTh' or self.data_type == 'ETTh1' or self.data_type == 'ETTh2':
            border1s = [0, 12 * 30 * 24 - self.input_len, 12 * 30 * 24 + 4 * 30 * 24 - self.input_len]
            border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        elif self.data_type == 'ETTm' or self.data_type == 'ETm1' or self.data_type == 'ETTm2':
            border1s = [0, 12 * 30 * 24 * 4 - self.input_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.input_len]
            border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        elif self.data_type == 'Solar':
            num_train = int(len(df_raw) * 0.7)
            num_test = int(len(df_raw) * 0.2)
            num_valid = int(len(df_raw) * 0.1)
            border1s = [0, num_train - self.input_len, len(df_raw) - num_test - self.input_len]
            border2s = [num_train, num_train + num_valid, len(df_raw)]
        elif self.data_type == 'PEMS':
            data_len = len(df_raw)
            num_train = int(data_len * 0.6)
            num_test = int(data_len * 0.2)
            num_vali = data_len - num_train - num_test
            border1s = [0, num_train - self.input_len, data_len - num_test - self.input_len]
            border2s = [num_train, num_train + num_vali, data_len]
        else:
            data_len = len(df_raw)
            num_train = int(data_len * 0.7)
            num_test = int(data_len * 0.2)
            num_vali = data_len - num_train - num_test
            border1s = [0, num_train - self.input_len, data_len - num_test - self.input_len]
            border2s = [num_train, num_train + num_vali, data_len]
            # import pdb
            # pdb.set_trace()

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        if self.set_type == 0:
            if self.use_post_data:
                border1 = int((border2 - self.input_len - self.pred_len - self.train_offset) * (1 - self.subset_ratio))
            else:
                border2 = int((border2 - self.input_len - self.pred_len - self.train_offset) * self.subset_ratio + self.input_len + self.pred_len)

            border1 += self.train_offset
        
        if isinstance(df_raw[df_raw.columns[0]][2], str):
            data = df_raw[df_raw.columns[1:]].values
        else:
            data = df_raw.values
        
        
        # if self.scale:
        #     train_data = data[border1s[0]:border2s[0]]
        #     self.scaler.fit(train_data)
        #     data = self.scaler.transform(data)

        if self.timeenc == 0:
            df_stamp = df_raw[['date']][border1:border2]
            df_stamp['date'] = pd.to_datetime(df_stamp.date)
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            if isinstance(df_raw[df_raw.columns[0]][2], str):
                data_stamp = time_features(pd.to_datetime(pd.to_datetime(df_raw.date).values), freq='h')
                data_stamp = data_stamp.transpose(1, 0)
            else:
                data_stamp = np.zeros((len(df_raw), 4))
        else:
            raise ValueError('Unknown timeenc: {}'.format(self.timeenc))
        
        # target_data = data[self.target_cols] 
        # self.data_x = data[border1:border2]
        self.data_x = data_features[border1:border2]
        # self.data_y = data[border1:border2]
        self.data_y = data_target[border1:border2]
        # self.data_y = target_data[border1:border2]
        # self.data_y = self.data_y[:,-1].reshape(self.data_x.shape[0],1)
        self.feature_scaler.fit(self.data_x)
        self.target_scaler.fit(self.data_y)
        self.data_x = self.feature_scaler.transform(self.data_x)
        self.data_y = self.target_scaler.transform(self.data_y)
        
        self.data_stamp = data_stamp
        self.n_var = self.data_x.shape[-1]
        self.n_timepoint = len(self.data_x) - self.input_len - self.pred_len + 1

        data_len = len(self.data_x)
 
        start_indices = [i for i in range(int(data_len)-(self.pred_len + self.seq_len) + 1)]
        self.start_indices = start_indices
       

    # def __getitem__(self, index):
    #     if self.set_type == 0:
    #         index = index * self.internal
    #     c_begin = index // self.n_timepoint  # select variable
    #     s_begin = index % self.n_timepoint  # select start time
    #     s_end = s_begin + self.input_len
    #     r_begin = s_end
    #     r_end = r_begin + self.pred_len
    #     seq_x = self.data_x[s_begin:s_end, c_begin:c_begin + 1]
    #     seq_y = self.data_y[r_begin:r_end, c_begin:c_begin + 1]
    #     seq_x_mark = self.data_stamp[s_begin:s_end]
    #     seq_y_mark = self.data_stamp[r_begin:r_end]

    #     return seq_x, seq_y, seq_x_mark, seq_y_mark
        
    # def __getitem__(self, index):
    #     if self.set_type == 0:
    #         index = index * self.internal
    #     c_begin = index // self.n_timepoint  # select variable
    #     s_begin = index % self.n_timepoint  # select start time
    #     s_end = s_begin + self.input_len
    #     r_begin = s_end - self.label_len
    #     r_end = r_begin + self.label_len + self.pred_len
    #     seq_x = self.data_x[s_begin:s_end, c_begin:c_begin + 1]
    #     seq_y = self.data_y[r_begin:r_end, c_begin:c_begin + 1]
    #     seq_x_mark = self.data_stamp[s_begin:s_end]
    #     seq_y_mark = self.data_stamp[r_begin:r_end]
    #     # print("seq_x shape:",seq_x.shape)
    #     # s_begin = self.stride * s_begin
    #     # s_end = s_begin + self.input_len
    #     # p_begin = s_begin + self.pred_len
    #     # p_end = p_begin + self.input_len
        
    #     return seq_x, seq_y, seq_x_mark, seq_y_mark
    def __getitem__(self, index):
        posi = int(self.start_indices[index])
        seq_x = self.data_x[posi : posi + self.seq_len]
        seq_y = self.data_y[posi + self.seq_len : posi + self.seq_len + self.pred_len]
        seq_x_mark = self.data_stamp[posi : posi + self.seq_len]
        seq_y_mark = self.data_stamp[posi + self.seq_len : posi + self.seq_len + self.pred_len]
        seq_x = np.array(seq_x)
        seq_y = np.array(seq_y)
        seq_x_mark = np.array(seq_x_mark)
        seq_y_mark = np.array(seq_y_mark)

        return seq_x, seq_y, seq_x_mark, seq_y_mark


    # def __len__(self):
    #     if self.set_type == 0:
    #         return int(self.n_var * self.n_timepoint * self.subset_rand_ratio)
    #     else:
    #         return int(self.n_var * self.n_timepoint)
    def __len__(self):
        return len(self.start_indices)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class CIAutoRegressionDatasetBenchmark(CIDatasetBenchmark):
    # Fair Benckmarks from Time-Series-Library: https://github.com/thuml/Time-Series-Library
    # And iTransformer: https://github.com/thuml/iTransformer/blob/main/data_provider/data_loader.py
    # The Split Ratio for Testing is Fixed to Keep Consistent with Previous Benchmarks
    # Including: ETT, ECL, Weather, Traffic, PEMS, SolarEnergy
    def __init__(self, root_path='dataset', flag='train', input_len=None, label_len=None, pred_len=None,
                 data_type='custom', scale=True, timeenc=1, freq='h', stride=1, subset_ratio=1.0, subset_rand_ratio=1.0,
                 use_post_data=False, train_offset=0):
        self.label_len = label_len
        super().__init__(root_path=root_path, flag=flag, input_len=input_len, pred_len=pred_len,
                         data_type=data_type, scale=scale, timeenc=timeenc, freq=freq, stride=stride,
                         subset_ratio=subset_ratio, subset_rand_ratio=subset_rand_ratio, use_post_data=use_post_data,
                         train_offset=train_offset)

    def __getitem__(self, index):
        if self.set_type == 0:
            index = index * self.internal
        c_begin = index // self.n_timepoint  # select variable
        s_begin = index % self.n_timepoint  # select start time
        s_end = s_begin + self.input_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end, c_begin:c_begin + 1]
        seq_y = self.data_y[r_begin:r_end, c_begin:c_begin + 1]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        # s_begin = self.stride * s_begin
        # s_end = s_begin + self.input_len
        # p_begin = s_begin + self.pred_len
        # p_end = p_begin + self.input_len
        return seq_x, seq_y, seq_x_mark, seq_y_mark

# class AluminaMSDataset(Dataset):

#     feature_scaler = StandardScaler()
#     target_scaler = StandardScaler()

#     def __init__(self, csv_path, segment_len,seq_len,pred_len, feature_cols, target_cols, 
#                  datetime_col='Time', interval='none', timestamp_feature='none', init_scaler=True, flag='train'):

#         # init
#         self.csv_path = csv_path
#         self.segment_len = segment_len
#         self.seq_len = seq_len
#         self.pred_len = pred_len
#         self.feature_cols = feature_cols.split(',')
#         self.flag = flag
#         self.target_cols = target_cols.split(',')
#         self.datetime_col = datetime_col
#         self.interval = interval
#         self.timestamp_feature = timestamp_feature
#         self.init_scaler = init_scaler

#         # init
#         assert flag in ['train', 'test', 'val']
#         type_map = {'train': 0, 'val': 1, 'test': 2}
#         self.set_type = type_map[flag]
#         self.data_type = data_type
#         if self.set_type == 0:
#             self.internal = int(1 // self.subset_rand_ratio)
#         else:
#             self.internal = 1
        

#         self.__read_data__()

#     def __read_data__(self):

#         data = pd.read_csv(self.csv_path,encoding="GB2312")

#         data = data.dropna()
#         data = data.reset_index(drop=True)
#         num_samples = data.shape[0]
#         train_num = int(num_samples * 0.7)
#         valid_num = int(num_samples * 0.1)
#         test_num = num_samples - train_num - valid_num

#         train_data = data[:train_num]
#         valid_data = data[train_num:-test_num]
#         test_data = data[-test_num:]
        



#         data_features = data[self.feature_cols]
#         data_target = data[self.target_cols]
#         # import pdb
#         # pdb.set_trace()
#         if self.init_scaler:
#             self.feature_scaler.fit(data_features)
#             self.target_scaler.fit(data_target)
#         self.data_features = self.feature_scaler.transform(data_features)
#         self.data_target = self.target_scaler.transform(data_target)

#         # get temporally length-fixed samples from data_features
#         data_len = len(data_features)
 
#         start_indices = [i for i in range(int(data_len)-self.segment_len + 1)]
#         self.start_indices = start_indices



#     def __getitem__(self, index):
#         posi = int(self.start_indices[index])
#         targets=[]
#         samples = self.data_features[posi : posi + self.seq_len]

#         targets = self.data_target[posi + self.seq_len : posi + self.seq_len + self.pred_len]
        
#         return samples, np.array(targets, dtype=float), 0
        

#     def __len__(self):
#         return len(self.start_indices)
