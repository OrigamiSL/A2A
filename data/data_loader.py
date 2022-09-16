import os
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import MaxAbsScaler
from torch.utils.data import Dataset
from utils.tools import StandardScaler

warnings.filterwarnings('ignore')


class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', criterion='Standard',
                 aug_num=2, jitter=0.2, order_num=1):
        # size [label_len, pred_len]
        # info
        if size is None:
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.label_len = size[0]
            self.pred_len = size[1]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.root_path = root_path
        self.data_path = data_path
        self.criterion = criterion
        self.aug_num = aug_num
        self.order_num = order_num
        self.jitter = jitter
        self.feature_num = 1
        self.__read_data__()

    def __read_data__(self):
        if str(self.criterion) == 'Standard':
            self.scaler = StandardScaler()
        else:
            self.scaler = MaxAbsScaler()

        self.scaler2 = StandardScaler()

        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.label_len, 12 * 30 * 24 + 4 * 30 * 24 - self.label_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + [self.target] + cols]

        if self.features == 'M':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        df_value = df_data.values

        # data standardization
        train_data = df_value[border1s[0]:border2s[0]]
        self.scaler.fit(train_data)
        self.scaler2.fit(train_data)
        data = self.scaler.transform(df_value)

        self.feature_num = data.shape[-1]
        self.data_x = data[border1:border2]

    def __getitem__(self, index):
        if self.pred_len:
            r_begin = index
            r_end = r_begin + self.label_len + self.pred_len
            seq_x = self.data_x[r_begin:r_end]
        else:
            r_begin = index
            r_end = r_begin + self.label_len
            seq_x = self.data_x[r_begin:r_end]
            seq_x = np.expand_dims(seq_x, axis=0)
            seq_x = seq_x.repeat(self.aug_num + self.order_num, axis=0)
            _, L, D = seq_x.shape
            for i in range(self.aug_num - 1):
                rand_aug = int(np.rint(np.random.rand(1) * 3)[0])
                if 0 <= rand_aug < 1:
                    seq_x[i + 1, :, :] *= 1 + (np.random.rand(L, D) - 0.5) * self.jitter
                elif 1 <= rand_aug < 2:
                    seq_x[i + 1, :, :] *= 1 + (np.random.rand(1) - 0.5)[0] * self.jitter
                else:
                    seq_x[i + 1, :, :] += (np.random.rand(L, D) - 0.5) * self.jitter
            for i in range(self.order_num):
                seq_x[i + self.aug_num, :, :] = seq_x[i + self.aug_num, np.random.permutation(L), :]
        return seq_x

    def __len__(self):
        return len(self.data_x) - self.label_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def standard_transformer(self, data):
        return self.scaler2.transform(data)

    def target_feature(self):
        return self.feature_num


class Dataset_ETT_min(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', criterion='Standard',
                 aug_num=2, jitter=0.2, order_num=1):
        # size [label_len, pred_len]
        # info
        if size is None:
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.label_len = size[0]
            self.pred_len = size[1]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.root_path = root_path
        self.data_path = data_path
        self.criterion = criterion
        self.aug_num = aug_num
        self.order_num = order_num
        self.jitter = jitter
        self.feature_num = 1
        self.__read_data__()

    def __read_data__(self):
        if str(self.criterion) == 'Standard':
            self.scaler = StandardScaler()
        else:
            self.scaler = MaxAbsScaler()

        self.scaler2 = StandardScaler()

        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.label_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.label_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + [self.target] + cols]

        if self.features == 'M':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        df_value = df_data.values

        # data standardization
        train_data = df_value[border1s[0]:border2s[0]]
        self.scaler.fit(train_data)
        self.scaler2.fit(train_data)
        data = self.scaler.transform(df_value)

        self.feature_num = data.shape[-1]
        self.data_x = data[border1:border2]

    def __getitem__(self, index):
        if self.pred_len:
            r_begin = index
            r_end = r_begin + self.label_len + self.pred_len
            seq_x = self.data_x[r_begin:r_end]
        else:
            r_begin = index
            r_end = r_begin + self.label_len
            seq_x = self.data_x[r_begin:r_end]
            seq_x = np.expand_dims(seq_x, axis=0)
            seq_x = seq_x.repeat(self.aug_num + self.order_num, axis=0)
            _, L, D = seq_x.shape
            for i in range(self.aug_num - 1):
                rand_aug = int(np.rint(np.random.rand(1) * 3)[0])
                if 0 <= rand_aug < 1:
                    seq_x[i + 1, :, :] *= 1 + (np.random.rand(L, D) - 0.5) * self.jitter
                elif 1 <= rand_aug < 2:
                    seq_x[i + 1, :, :] *= 1 + (np.random.rand(1) - 0.5)[0] * self.jitter
                else:
                    seq_x[i + 1, :, :] += (np.random.rand(L, D) - 0.5) * self.jitter
            for i in range(self.order_num):
                seq_x[i + self.aug_num, :, :] = seq_x[i + self.aug_num, np.random.permutation(L), :]
        return seq_x

    def __len__(self):
        return len(self.data_x) - self.label_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def standard_transformer(self, data):
        return self.scaler2.transform(data)

    def target_feature(self):
        return self.feature_num


class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ECL.csv',
                 target='OT', criterion='Standard',
                 aug_num=2, jitter=0.2, order_num=1):
        # size [label_len, pred_len]
        # info
        if size is None:
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.label_len = size[0]
            self.pred_len = size[1]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features

        self.target = target
        self.root_path = root_path
        self.data_path = data_path
        self.criterion = criterion
        self.aug_num = aug_num
        self.order_num = order_num
        self.jitter = jitter
        self.feature_num = 1
        self.__read_data__()

    def __read_data__(self):
        if str(self.criterion) == 'Standard':
            self.scaler = StandardScaler()
        else:
            self.scaler = MaxAbsScaler()

        self.scaler2 = StandardScaler()

        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + [self.target] + cols]

        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.label_len, len(df_raw) - num_test - self.label_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        df_value = df_data.values

        # data standardization
        train_data = df_value[border1s[0]:border2s[0]]
        self.scaler.fit(train_data)
        self.scaler2.fit(train_data)
        data = self.scaler.transform(df_value)

        self.feature_num = data.shape[-1]
        self.data_x = data[border1:border2]

    def __getitem__(self, index):
        if self.pred_len:
            r_begin = index
            r_end = r_begin + self.label_len + self.pred_len
            seq_x = self.data_x[r_begin:r_end]
        else:
            r_begin = index
            r_end = r_begin + self.label_len
            seq_x = self.data_x[r_begin:r_end]
            seq_x = np.expand_dims(seq_x, axis=0)
            seq_x = seq_x.repeat(self.aug_num + self.order_num, axis=0)
            _, L, D = seq_x.shape
            for i in range(self.aug_num - 1):
                rand_aug = int(np.rint(np.random.rand(1) * 3)[0])
                if 0 <= rand_aug < 1:
                    seq_x[i + 1, :, :] *= 1 + (np.random.rand(L, D) - 0.5) * self.jitter
                elif 1 <= rand_aug < 2:
                    seq_x[i + 1, :, :] *= 1 + (np.random.rand(1) - 0.5)[0] * self.jitter
                else:
                    seq_x[i + 1, :, :] += (np.random.rand(L, D) - 0.5) * self.jitter
            for i in range(self.order_num):
                seq_x[i + self.aug_num, :, :] = seq_x[i + self.aug_num, np.random.permutation(L), :]
        return seq_x

    def __len__(self):
        return len(self.data_x) - self.label_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def standard_transformer(self, data):
        return self.scaler2.transform(data)

    def target_feature(self):
        return self.feature_num
