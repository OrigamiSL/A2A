from data.data_loader import Dataset_ETT_hour, Dataset_ETT_min, Dataset_Custom
from exp.exp_basic import Exp_Basic
from A2A.model import A2A

from utils.tools import EarlyStopping, adjust_learning_rate, loss_process, Cost_loss
from utils.metrics import metric

import numpy as np

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import os
import time

import warnings

warnings.filterwarnings('ignore')


class Exp_Model(Exp_Basic):
    def __init__(self, args):
        super(Exp_Model, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'A2A': A2A
        }
        if self.args.model == 'A2A':
            model = model_dict[self.args.model](
                self.args.label_len,
                self.args.pred_list,
                self.args.enc_in,
                self.args.d_model,
                self.args.repr_dim,
                self.args.activation,
                self.args.dropout,
                self.args.kernel,
                self.args.attn_nums,
                self.args.pyramid
            ).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        total_num = sum(p.numel() for p in model.parameters())
        print("Total parameters: {0}MB".format(total_num / 1024 / 1024))
        return model

    def _get_data(self, flag, forecasting_form=None, pred_len=0):
        args = self.args
        data_set = None

        data_dict = {
            'ETTh1': Dataset_ETT_hour,
            'ETTm2': Dataset_ETT_min,
            'ECL': Dataset_Custom,
            'Exchange': Dataset_Custom,
            'Traffic': Dataset_Custom,
            'weather': Dataset_Custom
        }
        Data = data_dict[self.args.data]

        if flag == 'test':
            shuffle_flag = False
            drop_last = True
            batch_size = args.batch_size
        else:
            shuffle_flag = True
            drop_last = True
            batch_size = args.batch_size
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.label_len, pred_len],
            features=args.features,
            target=args.target,
            criterion=args.criterion,
            aug_num=args.aug_num,
            jitter=args.jitter,
            order_num=args.order_num
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size if forecasting_form == 'MLP' else self.args.cost_batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)

        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data=None, vali_loader=None, criterion=None, index=0):
        self.model.eval()
        total_loss = []
        with torch.no_grad():
            for i, batch_x in enumerate(vali_loader):
                pred, true = self._process_one_batch(
                    batch_x, index=index)
                loss = loss_process(self.args, pred, true, criterion, flag=1)
                total_loss.append(loss)

            total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        #  Contrastive learning
        train_data, train_loader = self._get_data(flag='train')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        model_optim = self._select_optimizer()

        for epoch in range(self.args.cost_epochs):

            if epoch > 0:
                train_data, train_loader = self._get_data(flag='train')
            iter_count = 0

            self.model.train()
            epoch_time = time.time()
            for i, batch_x in enumerate(train_loader):
                B, A, L, D = batch_x.shape
                batch_x = batch_x.contiguous().view(B * A, L, D)
                model_optim.zero_grad()
                iter_count += 1
                hidden_features = self._process_one_batch_cons(
                    batch_x)

                loss = Cost_loss(hidden_features, aug_num=self.args.aug_num,
                                 order_num=self.args.order_num, instance_loss=self.args.instance_loss)

                loss.backward(torch.ones_like(loss))
                model_optim.step()

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1,
                                                                            torch.mean(loss).item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.cost_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
            torch.save(self.model.state_dict(), path + '/' + 'checkpoint.pth')
            print("Stage: COST |Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        # MLP
        if self.args.regressor == 'MLP':
            lr = self.args.learning_rate
            for lens in self.args.pred_list:
                self.args.learning_rate = lr
                train_data, train_loader = self._get_data(flag='train', forecasting_form='MLP', pred_len=lens)
                vali_data, vali_loader = self._get_data(flag='val', forecasting_form='MLP', pred_len=lens)
                test_data, test_loader = self._get_data(flag='test', forecasting_form='MLP', pred_len=lens)

                time_now = time.time()
                train_steps = len(train_loader)

                early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
                criterion = self._select_criterion()
                for epoch in range(self.args.train_epochs):
                    iter_count = 0
                    self.model.train()
                    epoch_time = time.time()
                    for i, batch_x in enumerate(train_loader):
                        model_optim.zero_grad()
                        iter_count += 1
                        pred, true = self._process_one_batch(
                            batch_x, index=self.args.pred_list.index(lens))
                        loss = loss_process(self.args, pred, true, criterion, flag=0, dataset=train_data)

                        loss.backward(torch.ones_like(loss))
                        model_optim.step()

                        if (i + 1) % 100 == 0:
                            print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1,
                                                                                    torch.mean(loss).item()))
                            speed = (time.time() - time_now) / iter_count
                            left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                            print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                            iter_count = 0
                            time_now = time.time()

                    print("Pred:{}|Epoch: {} cost time: {}".format(lens, epoch + 1, time.time() - epoch_time))

                    vali_loss = self.vali(vali_data, vali_loader, criterion,
                                          index=self.args.pred_list.index(lens))
                    test_loss = self.vali(test_data, test_loader, criterion,
                                          index=self.args.pred_list.index(lens))

                    print("Pred_len: {0} Stage: MLP| Epoch: {1}, Steps: {2} | Vali Loss: {3:.7f} Test Loss: {4:.7f}".
                          format(lens, epoch + 1, train_steps, vali_loss, test_loss))
                    early_stopping(vali_loss, self.model, path)
                    if early_stopping.early_stop:
                        print("Early stopping")
                        break

                    adjust_learning_rate(model_optim, epoch + 1, self.args)

                best_model_path = path + '/' + 'checkpoint.pth'
                self.model.load_state_dict(torch.load(best_model_path))
        else:
            pass
        return self.model

    def test(self, setting, load=False, write_loss=True, save_loss=True, return_loss=False):
        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))
        self.model.eval()

        mse_list = []
        mae_list = []
        for index, lens in enumerate(self.args.pred_list):
            test_data, test_loader = self._get_data(flag='test', forecasting_form='MLP', pred_len=lens)
            criterion = self._select_criterion()
            mse = 0
            mae = 0
            if save_loss:
                preds = []
                trues = []
                with torch.no_grad():
                    for i, batch_x in enumerate(test_loader):
                        pred, true = self._process_one_batch(
                            batch_x, index=self.args.pred_list.index(lens))
                        if self.args.test_inverse:
                            pred = loss_process(self.args, pred, true, criterion, flag=2, dataset=test_data)
                            pred = pred.reshape(self.args.batch_size, lens, self.args.c_out)
                            true = true.reshape(-1, pred.shape[-1])
                            true = test_data.inverse_transform(true.detach().cpu().numpy())
                            true = test_data.standard_transformer(true)
                            true = true.reshape(self.args.batch_size, lens, self.args.c_out)
                        else:
                            pred = loss_process(self.args, pred, true, criterion, flag=2)
                            pred = pred.detach().cpu().numpy()
                            true = true.detach().cpu().numpy()
                        preds.append(pred)
                        trues.append(true)
                    preds = np.array(preds)
                    trues = np.array(trues)
                    print('test shape:', preds.shape, trues.shape)
                    preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
                    trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
                    print('test shape:', preds.shape, trues.shape)

                mae, mse = metric(preds, trues)
                print('|{}_{}|pred_len{}|mse:{}, mae:{}'.
                      format(self.args.data, self.args.features, lens, mse, mae) + '\n')
                # result save
                folder_path = './results/' + setting + '/'
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
                np.save(folder_path + f'metrics_{lens}.npy', np.array([mae, mse]))
                np.save(folder_path + f'pred_{lens}.npy', preds)
                np.save(folder_path + f'true_{lens}.npy', trues)
            else:
                mse_list = []
                mae_list = []
                infer_first = 0
                infer_second = 0
                with torch.no_grad():
                    for i, batch_x in enumerate(test_loader):
                        pred, true = self._process_one_batch(
                            batch_x, index=self.args.pred_list.index(lens))
                        if self.args.test_inverse:
                            pred = loss_process(self.args, pred, true, criterion, flag=2, dataset=test_data)
                            pred = pred.reshape(self.args.batch_size, lens, self.args.c_out)
                            true = true.reshape(-1, pred.shape[-1])
                            true = test_data.inverse_transform(true.detach().cpu().numpy())
                            true = test_data.standard_transformer(true)
                            true = true.reshape(self.args.batch_size, lens, self.args.c_out)
                        else:
                            pred = loss_process(self.args, pred, true, criterion, flag=2)
                            pred = pred.detach().cpu().numpy()
                            true = true.detach().cpu().numpy()
                        t_mae, t_mse = metric(pred, true)
                        mse_list.append(t_mse)
                        mae_list.append(t_mae)
                    mse = np.average(mse_list)
                    mae = np.average(mae_list)

                print('|{}_{}|pred_len{}|mse:{}, mae:{}'.
                      format(self.args.data, self.args.features, lens, mse, mae) + '\n')

            if write_loss:
                path = './result.log'
                with open(path, "a") as f:
                    f.write('inference time:{},{}'.format(infer_first, infer_second) + '\n')
                    f.write(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))
                    f.write('|{}_{}|pred_len{}|mse: {} , mae: {}'.
                            format(self.args.data, self.args.features, lens, mse,
                                   mae) + '\n')
                    f.flush()
                    f.close()
            else:
                pass

            mse_list.append(mse)
            mae_list.append(mae)

        if not save_loss:
            dir_path = os.path.join(self.args.checkpoints, setting)
            check_path = dir_path + '/' + 'checkpoint.pth'
            if os.path.exists(check_path):
                os.remove(check_path)
                os.removedirs(dir_path)
        if return_loss:
            return mse_list, mae_list
        else:
            return

    def _process_one_batch(self, batch_x, index=0):
        t = time.time()
        batch_x = batch_x.float().to(self.device)

        input_seq = batch_x[:, :self.args.label_len, :]
        outputs = self.model(input_seq, flag='MLP', index=index)
        batch_y = batch_x[:, -self.args.pred_list[index]:, :].to(self.device)

        return outputs, batch_y

    def _process_one_batch_cons(self, batch_x, batch_y=None):
        batch_x = batch_x.float().to(self.device)
        if batch_y is None:
            outputs = self.model(batch_x, flag='first stage')
        else:
            batch_y = batch_y.float().to(self.device)
            outputs = self.model(batch_x, batch_y, flag='first stage')
        return outputs
